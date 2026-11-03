# spec/rlcd_spec.cr
require "spec"
require "../src/rlcd"

private def softmax_of(logits : Array(Float64)) : Array(Float64)
  max = logits.max
  exps = logits.map { |v| Math.exp(v - max) }
  sum = exps.sum
  exps.map { |v| v / sum }
end

describe RLCD::Policy do
  it "produces a probability distribution" do
    policy = RLCD::Policy.new(input_dim: 4, hidden_dim: 8, option_count: 3, rng: Random.new(1))
    probs = policy.probs([0.5, -0.2, 0.1, 0.9])
    probs.sum.should be_close(1.0, 1e-9)
    probs.each { |p| p.should be > 0.0 }
  end

  it "samples valid option indices" do
    policy = RLCD::Policy.new(input_dim: 4, hidden_dim: 8, option_count: 3, rng: Random.new(2))
    100.times do
      a = policy.sample([0.5, -0.2, 0.1, 0.9])
      (0...3).should contain(a)
    end
  end
end

describe RLCD::ScoringRule do
  it "prefers the true distribution (propriety)" do
    truth = [0.1, 0.7, 0.15, 0.05]
    expected = ->(q : Array(Float64)) do
      truth.each_with_index.sum { |p_y, y| p_y * RLCD.score(RLCD::ScoringRule::Brier, q, y) }
    end
    honest = expected.call(truth)
    overconfident = expected.call([0.01, 0.97, 0.01, 0.01])
    hedged = expected.call([0.25, 0.25, 0.25, 0.25])
    honest.should be > overconfident
    honest.should be > hedged
  end

  it "brier logit gradient matches finite differences" do
    rng = Random.new(7)
    logits = Array.new(4) { rng.next_float * 2.0 - 1.0 }
    label = 2
    probs = softmax_of(logits)
    grad = RLCD.score_grad_logits(RLCD::ScoringRule::Brier, probs, label)
    eps = 1e-6
    logits.size.times do |k|
      plus = logits.dup
      minus = logits.dup
      plus[k] += eps
      minus[k] -= eps
      numerical = (RLCD.score(RLCD::ScoringRule::Brier, softmax_of(plus), label) -
                   RLCD.score(RLCD::ScoringRule::Brier, softmax_of(minus), label)) / (2.0 * eps)
      grad[k].should be_close(numerical, 1e-6)
    end
  end

  it "log-score logit gradient matches finite differences" do
    rng = Random.new(8)
    logits = Array.new(5) { rng.next_float * 2.0 - 1.0 }
    label = 0
    probs = softmax_of(logits)
    grad = RLCD.score_grad_logits(RLCD::ScoringRule::LogScore, probs, label)
    eps = 1e-6
    logits.size.times do |k|
      plus = logits.dup
      minus = logits.dup
      plus[k] += eps
      minus[k] -= eps
      numerical = (RLCD.score(RLCD::ScoringRule::LogScore, softmax_of(plus), label) -
                   RLCD.score(RLCD::ScoringRule::LogScore, softmax_of(minus), label)) / (2.0 * eps)
      grad[k].should be_close(numerical, 1e-6)
    end
  end
end

describe RLCD do
  it "ece is zero for perfect confidence and accuracy" do
    predictions = Array(RLCD::Prediction).new
    50.times { predictions << {[0.0, 1.0, 0.0], 1} }
    ece, _ = RLCD.ece(predictions)
    ece.should be_close(0.0, 1e-12)
  end

  it "ece is zero when confidence matches the hit rate" do
    rng = Random.new(3)
    predictions = Array(RLCD::Prediction).new
    20000.times do
      hit = rng.next_float < 0.25
      probs = [0.25, 0.25, 0.25, 0.25]
      predictions << {probs, hit ? 0 : 1}
    end
    ece, _ = RLCD.ece(predictions)
    ece.should be < 0.01
  end

  it "training improves the proper score on a verifiable task" do
    rng = Random.new(11)
    task = RLCD::SyntheticTask.new(input_dim: 8, class_count: 3, noise: 0.8, rng: rng)
    train_set = task.generate(200)
    policy = RLCD::Policy.new(input_dim: 8, hidden_dim: 16, option_count: 3, rng: rng)

    mean_score = -> do
      train_set.sum { |e| RLCD.score(RLCD::ScoringRule::Brier, policy.probs(e.features), e.label) } / train_set.size
    end

    before = mean_score.call
    trainer = RLCD::Trainer.new(policy: policy, lr: 0.01, group_size: 8, rng: rng)
    trainer.train(train_set, 30)
    mean_score.call.should be > before
  end
end

describe RLCD::Featurizer do
  it "is deterministic and normalizes non-empty input" do
    f = RLCD::Featurizer.new(dim: 64, seed: 99_u64)
    a = f.features("The payment failed", "Should I escalate?", "critical severity")
    b = f.features("The payment failed", "Should I escalate?", "critical severity")
    a.should eq(b)
    a.size.should eq(64)
    norm = Math.sqrt(a.sum { |v| v * v })
    norm.should be_close(1.0, 1e-9)
  end

  it "returns a zero vector for empty input" do
    f = RLCD::Featurizer.new(dim: 32, seed: 99_u64)
    f.features("", "", "").all?(&.zero?).should be_true
  end

  it "keeps word-only tokenization by default" do
    f = RLCD::Featurizer.new(dim: 256, seed: 99_u64)
    f.features("payment failed", "", "").should eq(f.features("payment-failed", "", ""))
    f.features("{}", "", "").all?(&.zero?).should be_true
  end

  it "normalizes deterministic code features and preserves empty input" do
    f = RLCD::Featurizer.new(dim: 4096, seed: 99_u64, code: true)
    a = f.features("def add(a, b):\r\n    return a + b", "", "")
    b = f.features("def add(a, b):\n    return a + b", "", "")
    a.should eq(b)
    a.size.should eq(4096)
    Math.sqrt(a.sum { |v| v * v }).should be_close(1.0, 1e-9)
    f.features("", "", "").all?(&.zero?).should be_true
  end

  it "retains punctuation and distinguishes operators in code mode" do
    f = RLCD::Featurizer.new(dim: 4096, seed: 99_u64, code: true)
    f.features("{}", "", "").any? { |v| !v.zero? }.should be_true
    f.features("value -> other", "", "").should_not eq(f.features("value = other", "", ""))
  end

  it "shares character n-grams across similar identifiers" do
    f = RLCD::Featurizer.new(dim: 4096, seed: 99_u64, code: true)
    a = f.features("abcdefgh", "", "")
    b = f.features("abcdefxy", "", "")
    c = f.features("12345678", "", "")
    shared = a.each_with_index.sum { |v, i| v * b[i] }
    disjoint = a.each_with_index.sum { |v, i| v * c[i] }
    shared.should be > disjoint
  end

  it "applies code tokenization after resolving structured state" do
    f = RLCD::Featurizer.new(dim: 4096, seed: 99_u64, code: true)
    state = JSON.parse(%q({"snippet":"value -> other"}))
    ref = f.features(state, "classify `snippet`", "")
    lit = f.features("snippet: value -> other", "classify value -> other", "")
    ref.should eq(lit)
  end
end

describe RLCD::NoulTask do
  it "produces binary labeled examples with the featurizer dimension" do
    task = RLCD::NoulTask.new(dim: 64, seed: 42_u64)
    exs = task.examples
    exs.size.should be > 0
    exs.each do |e|
      (0..1).should contain(e.label)
      e.features.size.should eq(64)
    end
  end
end

describe RLCD::IntentTask do
  it "produces 4-class labeled examples with the featurizer dimension" do
    task = RLCD::IntentTask.new(dim: 64, seed: 42_u64)
    exs = task.examples
    exs.size.should be > 0
    exs.each do |e|
      (0..3).should contain(e.label)
      e.features.size.should eq(64)
    end
  end
end

describe "RLCD::Policy (linear head)" do
  it "produces a probability distribution" do
    policy = RLCD::Policy.new(input_dim: 4, hidden_dim: 0, option_count: 3, rng: Random.new(5))
    probs = policy.probs([0.5, -0.2, 0.1, 0.9])
    probs.sum.should be_close(1.0, 1e-9)
    probs.each { |p| p.should be > 0.0 }
  end

  it "improves the proper score on a verifiable task" do
    rng = Random.new(12)
    task = RLCD::SyntheticTask.new(input_dim: 8, class_count: 3, noise: 0.8, rng: rng)
    train_set = task.generate(200)
    policy = RLCD::Policy.new(input_dim: 8, hidden_dim: 0, option_count: 3, rng: rng)

    mean_score = -> do
      train_set.sum { |e| RLCD.score(RLCD::ScoringRule::Brier, policy.probs(e.features), e.label) } / train_set.size
    end

    before = mean_score.call
    trainer = RLCD::Trainer.new(policy: policy, lr: 0.01, group_size: 8, batch_size: 32, rng: rng)
    trainer.train(train_set, 30)
    mean_score.call.should be > before
  end
end

describe "RLCD temperature scaling" do
  it "returns a positive finite temperature" do
    rng = Random.new(13)
    logits = Array.new(50) { Array.new(3) { rng.next_float * 4.0 - 2.0 } }
    labels = Array.new(50) { rng.rand(3) }
    t = RLCD.fit_temperature(logits, labels)
    t.should be > 0.0
    t.finite?.should be_true
  end

  it "softens overconfident logits" do
    logits = [] of Array(Float64)
    labels = [] of Int32
    50.times { logits << [10.0, 0.0, 0.0]; labels << 0 }
    50.times { logits << [10.0, 0.0, 0.0]; labels << 1 }
    RLCD.fit_temperature(logits, labels).should be > 1.0
  end
end

describe "RLCD::Policy persistence" do
  it "save/load round-trips identical probabilities" do
    policy = RLCD::Policy.new(input_dim: 4, hidden_dim: 8, option_count: 3, rng: Random.new(9))
    path = "/tmp/rlcd_spec_model.json"
    policy.save(path)
    loaded = RLCD::Policy.load(path)
    x = [0.5, -0.2, 0.1, 0.9]
    p1 = policy.probs(x)
    p2 = loaded.probs(x)
    p1.size.times { |i| p2[i].should be_close(p1[i], 1e-9) }
    File.delete(path) if File.exists?(path)
  end
end

describe "RLCD mini-batch training" do
  it "improves the proper score with batching" do
    rng = Random.new(14)
    task = RLCD::SyntheticTask.new(input_dim: 8, class_count: 3, noise: 0.8, rng: rng)
    train_set = task.generate(200)
    policy = RLCD::Policy.new(input_dim: 8, hidden_dim: 16, option_count: 3, rng: rng)

    mean_score = -> do
      train_set.sum { |e| RLCD.score(RLCD::ScoringRule::Brier, policy.probs(e.features), e.label) } / train_set.size
    end

    before = mean_score.call
    trainer = RLCD::Trainer.new(policy: policy, lr: 0.01, group_size: 8, batch_size: 64, rng: rng)
    trainer.train(train_set, 30)
    mean_score.call.should be > before
  end
end

describe RLCD::Mat do
  it "indexes and mutates values" do
    m = RLCD::Mat.new(2, 3)
    m[0, 1] = 5.0
    m[1, 2] = -2.5
    m[0, 1].should eq(5.0)
    m[1, 2].should eq(-2.5)
    m[0, 0].should eq(0.0)
  end

  it "mul_vec computes the matrix-vector product" do
    m = RLCD::Mat.new(2, 3)
    m[0, 0] = 1.0; m[0, 1] = 2.0; m[0, 2] = 3.0
    m[1, 0] = 4.0; m[1, 1] = 5.0; m[1, 2] = 6.0
    m.mul_vec([1.0, 10.0, 100.0]).should eq([321.0, 654.0])
  end

  it "t_mul_vec computes the transposed product" do
    m = RLCD::Mat.new(2, 3)
    m[0, 0] = 1.0; m[0, 1] = 2.0; m[0, 2] = 3.0
    m[1, 0] = 4.0; m[1, 1] = 5.0; m[1, 2] = 6.0
    m.t_mul_vec([1.0, 10.0]).should eq([41.0, 52.0, 63.0])
  end

  it "t_mul_vec_into writes into a reused buffer" do
    m = RLCD::Mat.new(2, 3)
    m[0, 0] = 1.0; m[0, 1] = 2.0; m[0, 2] = 3.0
    m[1, 0] = 4.0; m[1, 1] = 5.0; m[1, 2] = 6.0
    dest = Array(Float64).new(3, 0.0)
    m.t_mul_vec_into([1.0, 10.0], dest).should eq([41.0, 52.0, 63.0])
    m.t_mul_vec_into([0.0, 1.0], dest).should eq([4.0, 5.0, 6.0])
  end

  it "add_outer! accumulates the outer product" do
    m = RLCD::Mat.new(2, 2)
    m.add_outer!([1.0, 2.0], [10.0, 100.0])
    m.add_outer!([1.0, 2.0], [10.0, 100.0], 0.5)
    m[0, 0].should eq(15.0)
    m[0, 1].should eq(150.0)
    m[1, 0].should eq(30.0)
    m[1, 1].should eq(300.0)
  end
end

describe RLCD::Adam do
  it "leaves parameters unchanged on a zero gradient" do
    rng = Random.new(30)
    policy = RLCD::Policy.new(input_dim: 2, hidden_dim: 3, option_count: 2, rng: rng)
    adam = RLCD::Adam.new(policy, lr: 0.01)
    before = policy.param_slices.map(&.to_a)
    grads = policy.param_slices.map { |s| Slice(Float64).new(s.size, 0.0) }
    adam.step!(grads)
    policy.param_slices.each_with_index do |s, i|
      s.size.times { |j| s[j].should be_close(before[i][j], 1e-12) }
    end
  end

  it "descends against a constant gradient" do
    rng = Random.new(31)
    policy = RLCD::Policy.new(input_dim: 2, hidden_dim: 0, option_count: 2, rng: rng)
    adam = RLCD::Adam.new(policy, lr: 0.01)
    before = policy.param_slices[0][0]
    grads = policy.param_slices.map { |s| Slice(Float64).new(s.size, 1.0) }
    adam.step!(grads)
    policy.param_slices[0][0].should be < before
  end
end

describe "RLCD::Trainer backprop" do
  it "matches finite differences for MLP and linear heads" do
    {4, 0}.each do |hidden|
      rng = Random.new(21 + hidden)
      policy = RLCD::Policy.new(input_dim: 3, hidden_dim: hidden, option_count: 3, rng: rng)
      trainer = RLCD::Trainer.new(policy: policy, rng: rng)
      x = [0.4, -0.7, 0.2]
      label = 1
      fwd = policy.forward(x)
      sgrad = RLCD.score_grad_logits(RLCD::ScoringRule::Brier, fwd.probs, label)
      dlogits = sgrad.map { |g| -g }
      acc = policy.param_slices.map { |s| Array(Float64).new(s.size, 0.0) }
      trainer.backprop(x, fwd, dlogits, acc)

      eps = 1e-6
      policy.param_slices.each_with_index do |slice, pi|
        slice.size.times do |j|
          orig = slice[j]
          slice[j] = orig + eps
          plus = -RLCD.score(RLCD::ScoringRule::Brier, policy.probs(x), label)
          slice[j] = orig - eps
          minus = -RLCD.score(RLCD::ScoringRule::Brier, policy.probs(x), label)
          slice[j] = orig
          acc[pi][j].should be_close((plus - minus) / (2.0 * eps), 1e-5)
        end
      end
    end
  end
end

describe "RLCD::Policy categorical sampling" do
  it "samples in proportion to the given distribution" do
    policy = RLCD::Policy.new(input_dim: 2, hidden_dim: 0, option_count: 3, rng: Random.new(40))
    counts = [0, 0, 0]
    3000.times { counts[policy.categorical([0.1, 0.2, 0.7])] += 1 }
    counts[0].should be_close(300, 80)
    counts[1].should be_close(600, 100)
    counts[2].should be_close(2100, 120)
  end
end

describe "RLCD.split_examples" do
  it "partitions into disjoint, complete splits" do
    rng = Random.new(50)
    examples = Array.new(100) { |i| RLCD::Example.new([i.to_f], i % 4) }
    train, val, test = RLCD.split_examples(examples, 0.7, 0.2, rng)
    train.size.should eq(70)
    val.size.should eq(20)
    test.size.should eq(10)
    (train + val + test).map(&.features[0]).sort.should eq((0...100).map(&.to_f).sort)
  end
end

describe "task determinism" do
  it "NoulTask is deterministic for a fixed seed" do
    RLCD::NoulTask.new(dim: 32, seed: 42_u64).examples.should eq(RLCD::NoulTask.new(dim: 32, seed: 42_u64).examples)
  end

  it "IntentTask is deterministic for a fixed seed" do
    RLCD::IntentTask.new(dim: 32, seed: 42_u64).examples.should eq(RLCD::IntentTask.new(dim: 32, seed: 42_u64).examples)
  end
end

describe "RLCD::Policy persistence errors" do
  it "raises a clear error for truncated saved weights" do
    path = "/tmp/rlcd_spec_bad_model.json"
    File.write(path, "{\"input_dim\": 4, \"hidden_dim\": 8, \"option_count\": 3, \"w1\": [0.0], \"b1\": [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0], \"w2\": [0.0], \"b2\": [0.0,0.0,0.0]}")
    expect_raises(ArgumentError) { RLCD::Policy.load(path) }
    File.delete(path) if File.exists?(path)
  end
end

describe "RLCD.ece edge cases" do
  it "returns zero ece and empty bins for no predictions" do
    ece, bins = RLCD.ece(Array(RLCD::Prediction).new)
    ece.should eq(0.0)
    bins.size.should eq(10)
    bins.all?(&.count.zero?).should be_true
  end

  it "resolves argmax ties to the first maximum" do
    preds = Array(RLCD::Prediction).new
    10.times { preds << {[0.5, 0.5, 0.0], 0} }
    _, bins = RLCD.ece(preds)
    bin = bins[5]
    bin.count.should eq(10)
    bin.avg_confidence.should be_close(0.5, 1e-9)
    bin.accuracy.should be_close(1.0, 1e-9)
  end
end

describe "task curated injection" do
  it "NoulTask accepts injected curated examples" do
    custom = [{"custom escalation", 1}, {"custom non-escalation", 0}]
    task = RLCD::NoulTask.new(dim: 32, seed: 42_u64, curated: custom)
    exs = task.examples
    exs.size.should eq(custom.size + 1600)
    exs[0].label.should eq(1)
    exs[1].label.should eq(0)
  end

  it "IntentTask accepts injected curated examples" do
    custom = [{"custom intent", 2}]
    task = RLCD::IntentTask.new(dim: 32, seed: 42_u64, curated: custom)
    exs = task.examples
    exs.size.should eq(custom.size + 1600)
    exs[0].label.should eq(2)
  end
end

describe "RLCD answer helpers" do
  it "computes noul/choice confidence" do
    RLCD.confidence([0.5, 0.5]).should be_close(0.0, 1e-9)
    RLCD.confidence([1.0, 0.0]).should be_close(1.0, 1e-9)
    RLCD.confidence([0.7, 0.3]).should be_close(0.4, 1e-9)
    RLCD.confidence([0.25, 0.25, 0.25, 0.25]).should be_close(0.0, 1e-9)
  end

  it "computes the probability-weighted score" do
    RLCD.expected_score([0.1, 0.2, 0.3, 0.4]).should be_close(2.0, 1e-9)
    RLCD.expected_score([1.0, 0.0, 0.0]).should be_close(0.0, 1e-9)
  end

  it "computes score confidence" do
    RLCD.score_confidence([1.0, 0.0, 0.0, 0.0]).should be_close(1.0, 1e-9)
    RLCD.score_confidence([0.25, 0.25, 0.25, 0.25]).should be_close(0.0, 1e-9)
  end
end

describe "RLCD structured state featurization" do
  it "flattens JSON state and resolves field references" do
    f = RLCD::Featurizer.new(dim: 128, seed: 99_u64)
    state = JSON.parse(%q({"ticket": {"subject": "Duplicate charge", "messages": [{"text": "I was charged twice"}]}}))
    ref = f.features(state, "does `ticket.messages[0].text` request a refund?", "refund")
    lit = f.features("ticket: subject: Duplicate charge messages: text: I was charged twice", "does I was charged twice request a refund?", "refund")
    ref.should eq(lit)
    ref.size.should eq(128)
  end

  it "keeps unresolvable field references unchanged" do
    f = RLCD::Featurizer.new(dim: 64, seed: 99_u64)
    state = JSON.parse(%q({"a": 1}))
    ref = f.features(state, "is `nope.missing` here?", "")
    lit = f.features("a: 1", "is `nope.missing` here?", "")
    ref.should eq(lit)
  end
end

describe RLCD::ScoreTask do
  it "produces ordered-level labeled examples with the featurizer dimension" do
    task = RLCD::ScoreTask.new(dim: 64, seed: 42_u64)
    exs = task.examples
    exs.size.should be > 0
    task.option_count.should eq(4)
    task.legend.size.should eq(4)
    exs.each { |e| (0..3).should contain(e.label) }
  end

  it "accepts injected curated examples" do
    custom = [{"very angry message", 3}]
    task = RLCD::ScoreTask.new(dim: 32, seed: 42_u64, curated: custom)
    exs = task.examples
    exs.size.should eq(custom.size + 1600)
    exs[0].label.should eq(3)
  end
end

describe RLCD::SystemOne do
  it "answers mixed questions for a shared state" do
    f = RLCD::Featurizer.new(dim: 64, seed: 99_u64)
    p2 = RLCD::Policy.new(input_dim: 64, hidden_dim: 8, option_count: 2, rng: Random.new(70))
    p4 = RLCD::Policy.new(input_dim: 64, hidden_dim: 8, option_count: 4, rng: Random.new(71))
    sys = RLCD::SystemOne.new(f, {2 => p2, 4 => p4})
    qs = [
      RLCD::Question.new("urgent", RLCD::Primitive::Noul, "Is this urgent?", ["no", "yes"]),
      RLCD::Question.new("team", RLCD::Primitive::Choice, "Which team?", ["billing", "technical", "sales", "other"]),
    ]
    answers = sys.answer("The API is down", qs)
    answers.size.should eq(2)
    a0 = answers[0].as(RLCD::NoulAnswer)
    a0.id.should eq("urgent")
    a0.noul.should be_close(a0.probabilities[1], 1e-9)
    a1 = answers[1].as(RLCD::ChoiceAnswer)
    a1.id.should eq("team")
    a1.probabilities.size.should eq(4)
    a1.choice.should eq(a1.probabilities.index(a1.probabilities.max).not_nil!)
  end

  it "raises when no policy matches an option count" do
    f = RLCD::Featurizer.new(dim: 64, seed: 99_u64)
    p2 = RLCD::Policy.new(input_dim: 64, hidden_dim: 8, option_count: 2, rng: Random.new(72))
    sys = RLCD::SystemOne.new(f, {2 => p2})
    qs = [RLCD::Question.new("team", RLCD::Primitive::Choice, "Which team?", ["a", "b", "c"])]
    expect_raises(ArgumentError) { sys.answer("state", qs) }
  end
end

private def spec_policy(hidden : Int32, seed : Int32 = 80) : RLCD::Policy
  RLCD::Policy.new(input_dim: 6, hidden_dim: hidden, option_count: 3, rng: Random.new(seed))
end

private def spec_model(optimizer : Bool = false) : RLCD::Model
  featurizer = RLCD::Featurizer.new(dim: 64, seed: 0xcbf29ce484222325_u64, code: true)
  policy     = RLCD::Policy.new(input_dim: 64, hidden_dim: 8, option_count: 3, rng: Random.new(81))
  examples   = [{"def add(a, b)\n  a + b\nend", 0}, {"fn add(a: i32) -> i32 { a }", 1}, {"{\"a\": [1, 2]}", 2}].map do |(text, label)|
    RLCD::Example.new(featurizer.features(text, "", ""), label)
  end
  trainer = RLCD::Trainer.new(policy: policy, rng: Random.new(82))
  trainer.train(examples, 5)
  RLCD::Model.new(featurizer, policy, ["crystal", "rust", "json"], temperature: 1.5, optimizer: optimizer ? trainer.optimizer : nil)
end

private def binary(value, precision : RLCD::Precision = RLCD::Precision::F64, format : IO::ByteFormat = IO::ByteFormat::LittleEndian) : Bytes
  io = IO::Memory.new
  value.to_io(io, format, precision)
  io.to_slice
end

private def max_param_delta(a : RLCD::Policy, b : RLCD::Policy) : Float64
  delta = 0.0
  a.param_slices.each_with_index do |slice, i|
    other = b.param_slices[i]
    slice.size.times { |j| delta = {delta, (slice[j] - other[j]).abs}.max }
  end
  delta
end

describe RLCD::Checksum do
  it "computes the standard CRC-32 check value" do
    sum = RLCD::Checksum.new(IO::Memory.new)
    sum.write("123456789".to_slice)
    sum.value.should eq(0xCBF43926_u32)
  end
end

describe "RLCD::Policy serialization" do
  it "round-trips JSON exactly for MLP and linear heads" do
    {8, 0}.each do |hidden|
      policy = spec_policy(hidden)
      max_param_delta(policy, RLCD::Policy.from_json(policy.to_json)).should eq(0.0)
    end
  end

  it "loads JSON written with empty w1/b1 for a linear head" do
    json = %({"input_dim": 2, "hidden_dim": 0, "option_count": 2, "w1": [], "b1": [], "w2": [1.0, 2.0, 3.0, 4.0], "b2": [0.5, -0.5]})
    RLCD::Policy.from_json(json).param_slices.map(&.to_a).should eq([[1.0, 2.0, 3.0, 4.0], [0.5, -0.5]])
  end

  it "round-trips binary F64 exactly in both byte orders" do
    {IO::ByteFormat::LittleEndian, IO::ByteFormat::BigEndian}.each do |format|
      {8, 0}.each do |hidden|
        policy = spec_policy(hidden)
        io     = IO::Memory.new
        io.write_bytes(policy, format)
        io.rewind
        max_param_delta(policy, io.read_bytes(RLCD::Policy)).should eq(0.0)
      end
    end
  end

  it "shrinks with lower precision while keeping predictions close" do
    policy = spec_policy(8)
    x      = [0.3, -0.1, 0.8, 0.0, -0.6, 0.2]
    sizes  = {RLCD::Precision::F64, RLCD::Precision::F32, RLCD::Precision::I8}.map do |precision|
      bytes  = binary(policy, precision)
      loaded = RLCD::Policy.from_io(IO::Memory.new(bytes))
      tolerance = precision.i8? ? 0.05 : (precision.f32? ? 1e-6 : 0.0)
      max_param_delta(policy, loaded).should be <= tolerance
      policy.probs(x).zip(loaded.probs(x)) { |a, b| b.should be_close(a, 0.02) }
      bytes.size
    end
    sizes[0].should be > sizes[1]
    sizes[1].should be > sizes[2]
  end

  it "saves by extension and detects the format on load" do
    policy = spec_policy(8)
    {".json" => '{'.ord, ".rlcd" => 'R'.ord}.each do |extension, first|
      path = File.tempname("rlcd_spec", extension)
      policy.save(path)
      File.open(path) { |file| file.read_byte.should eq(first) }
      max_param_delta(policy, RLCD::Policy.load(path)).should eq(0.0)
      File.delete(path)
    end
  end

  it "rejects a non-F64 precision for JSON" do
    expect_raises(ArgumentError, /binary format only/) do
      spec_policy(8).save(File.tempname("rlcd_spec", ".json"), precision: RLCD::Precision::F32)
    end
  end
end

describe "RLCD binary container errors" do
  it "rejects corrupted bytes with a checksum mismatch" do
    bytes = binary(spec_policy(8)).dup
    bytes[40] ^= 0xFF_u8
    expect_raises(RLCD::FormatError, /checksum mismatch/) { RLCD::Policy.from_io(IO::Memory.new(bytes)) }
  end

  it "rejects truncated input" do
    bytes = binary(spec_policy(8))
    expect_raises(RLCD::FormatError, /truncated/) { RLCD::Policy.from_io(IO::Memory.new(bytes[0, bytes.size - 10])) }
  end

  it "rejects huge declared dimensions before allocating" do
    io = IO::Memory.new
    RLCD::Container.write(io, IO::ByteFormat::LittleEndian) do |writer|
      writer.add(RLCD::Container::TAG[:policy]) do |payload|
        {40_000, 40_000, 2}.each { |dim| payload.write_bytes(dim, IO::ByteFormat::LittleEndian) }
        payload.write_byte(0_u8)
      end
    end
    io.rewind
    expect_raises(RLCD::FormatError, /policy weights holds 0 bytes/) { RLCD::Policy.from_io(io) }
  end

  it "reports a missing chunk" do
    expect_raises(RLCD::FormatError, /missing FEAT chunk/) { RLCD::Model.from_io(IO::Memory.new(binary(spec_policy(8)))) }
  end

  it "rejects unrecognized input" do
    expect_raises(RLCD::FormatError, /unrecognized model format/) { RLCD::Format.detect("model".to_slice) }
  end
end

describe RLCD::Model do
  it "validates that its parts fit together" do
    featurizer = RLCD::Featurizer.new(dim: 6)
    expect_raises(ArgumentError, /labels/) { RLCD::Model.new(featurizer, spec_policy(8), ["a", "b"]) }
    expect_raises(ArgumentError, /input_dim/) { RLCD::Model.new(RLCD::Featurizer.new(dim: 7), spec_policy(8), ["a", "b", "c"]) }
    expect_raises(ArgumentError, /temperature/) { RLCD::Model.new(featurizer, spec_policy(8), ["a", "b", "c"], temperature: 0.0) }
    expect_raises(ArgumentError, /different policy/) do
      RLCD::Model.new(featurizer, spec_policy(8), ["a", "b", "c"], optimizer: RLCD::Adam.new(spec_policy(8)))
    end
  end

  it "predicts labels sorted by temperature-scaled probability" do
    model     = spec_model
    predicted = model.predict("def add(a, b)\n  a + b\nend")
    predicted.map(&.[1]).should eq(predicted.map(&.[1]).sort.reverse)
    predicted.sum(&.[1]).should be_close(1.0, 1e-9)
    predicted.first[0].should eq("crystal")
  end

  it "round-trips JSON and every binary precision" do
    model = spec_model(optimizer: true)
    text  = "fn main() -> i32 { 0 }"
    json  = RLCD::Model.from_json(model.to_json)
    json.labels.should eq(model.labels)
    json.featurizer.seed.should eq(model.featurizer.seed)
    json.featurizer.code?.should be_true
    json.probs(text).should eq(model.probs(text))
    json.optimizer.not_nil!.t.should eq(model.optimizer.not_nil!.t)

    RLCD::Precision.each do |precision|
      loaded = RLCD::Model.from_io(IO::Memory.new(binary(model, precision)))
      loaded.temperature.should eq(1.5)
      loaded.optimizer.not_nil!.t.should eq(model.optimizer.not_nil!.t)
      model.probs(text).zip(loaded.probs(text)) { |a, b| b.should be_close(a, precision.f64? ? 0.0 : 0.02) }
    end
  end

  it "loads a model file as a policy" do
    model = spec_model
    max_param_delta(model.policy, RLCD::Policy.from_io(IO::Memory.new(binary(model)))).should eq(0.0)
  end

  it "resumes training identically from saved optimizer state" do
    model    = spec_model(optimizer: true)
    examples = [{"puts 1", 0}, {"let x = 1;", 1}, {"[1, 2]", 2}].map do |(text, label)|
      RLCD::Example.new(model.featurizer.features(text, "", ""), label)
    end
    {RLCD::Format::Json, RLCD::Format::Binary}.each do |format|
      original = spec_model(optimizer: true)
      path     = File.tempname("rlcd_spec", format.json? ? ".json" : ".rlcd")
      original.save(path)
      loaded = RLCD::Model.load(path)
      File.delete(path)

      RLCD::Trainer.new(policy: original.policy, optimizer: original.optimizer, rng: Random.new(90)).train(examples, 3)
      RLCD::Trainer.new(policy: loaded.policy, optimizer: loaded.optimizer, rng: Random.new(90)).train(examples, 3)
      max_param_delta(original.policy, loaded.policy).should eq(0.0)
    end

    fresh = spec_model(optimizer: true)
    reset = RLCD::Model.from_io(IO::Memory.new(binary(fresh)))
    RLCD::Trainer.new(policy: fresh.policy, optimizer: fresh.optimizer, rng: Random.new(90)).train(examples, 3)
    RLCD::Trainer.new(policy: reset.policy, rng: Random.new(90)).train(examples, 3)
    max_param_delta(fresh.policy, reset.policy).should be > 0.0
  end
end

describe "RLCD::Trainer optimizer" do
  it "rejects an optimizer bound to another policy" do
    expect_raises(ArgumentError, /different policy/) do
      RLCD::Trainer.new(policy: spec_policy(8), optimizer: RLCD::Adam.new(spec_policy(8)))
    end
  end

  it "keeps the optimizer learning rate unless one is given" do
    policy = spec_policy(8)
    adam   = RLCD::Adam.new(policy, lr: 0.02)
    RLCD::Trainer.new(policy: policy, optimizer: adam).optimizer.lr.should eq(0.02)
    RLCD::Trainer.new(policy: policy, optimizer: adam, lr: 0.01).optimizer.lr.should eq(0.01)
  end
end

private def checkpoint_examples : Array(RLCD::Example)
  RLCD::SyntheticTask.new(input_dim: 6, class_count: 3, noise: 0.8, rng: Random.new(60)).generate(120)
end

private def checkpoint_trainer(policy : RLCD::Policy, optimizer : RLCD::Adam? = nil, seed : UInt64? = 61_u64, rng : Random = Random.new) : RLCD::Trainer
  RLCD::Trainer.new(policy: policy, lr: 0.02, lr_decay: 0.5, rng: rng, optimizer: optimizer, seed: seed)
end

describe RLCD::Checkpoint do
  it "round-trips inside a model, JSON and binary" do
    policy = spec_policy(8)
    model  = RLCD::Model.new(RLCD::Featurizer.new(dim: 6), policy, ["a", "b", "c"], checkpoint: RLCD::Checkpoint.new(3, 10, 99_u64))
    RLCD::Model.from_json(model.to_json).checkpoint.should eq(model.checkpoint)
    RLCD::Model.from_io(IO::Memory.new(binary(model))).checkpoint.should eq(model.checkpoint)
    unseeded = RLCD::Model.new(RLCD::Featurizer.new(dim: 6), policy, ["a", "b", "c"], checkpoint: RLCD::Checkpoint.new(10, 10, nil))
    RLCD::Model.from_io(IO::Memory.new(binary(unseeded))).checkpoint.not_nil!.finished?.should be_true
    RLCD::Model.from_json(RLCD::Model.new(RLCD::Featurizer.new(dim: 6), policy, ["a", "b", "c"]).to_json).checkpoint.should be_nil
  end

  it "rejects an epoch outside the run" do
    expect_raises(RLCD::FormatError, /outside 0..4/) { RLCD::Checkpoint.from_json(%({"epoch": 5, "epochs": 4, "seed": null})) }
  end
end

describe "RLCD::Trainer checkpoints" do
  it "shuffles identically for the same seed, whatever the trainer rng" do
    a = spec_policy(4, 62)
    b = spec_policy(4, 62)
    checkpoint_trainer(a, rng: Random.new(1)).train(checkpoint_examples, 3)
    checkpoint_trainer(b, rng: Random.new(2)).train(checkpoint_examples, 3)
    max_param_delta(a, b).should eq(0.0)
  end

  it "resumes an interrupted run to the same weights, learning-rate decay included" do
    straight = spec_policy(4, 63)
    checkpoint_trainer(straight).train(checkpoint_examples, 6)

    policy  = spec_policy(4, 63)
    trainer = checkpoint_trainer(policy)
    trainer.train(checkpoint_examples, 6) { |stats| break if stats.epoch == 3 }
    trainer.checkpoint.should eq(RLCD::Checkpoint.new(3, 6, 61_u64))
    path = File.tempname("rlcd_spec_checkpoint", ".rlcd")
    RLCD::Model.new(RLCD::Featurizer.new(dim: 6), policy, ["a", "b", "c"], optimizer: trainer.optimizer, checkpoint: trainer.checkpoint).save(path)
    saved = RLCD::Model.load(path)
    File.delete(path)

    resumed = checkpoint_trainer(saved.policy, saved.optimizer)
    epochs  = [] of Int32
    resumed.train(checkpoint_examples, saved.checkpoint.not_nil!) { |stats| epochs << stats.epoch }
    epochs.should eq([4, 5, 6])
    resumed.checkpoint.finished?.should be_true
    max_param_delta(straight, saved.policy).should eq(0.0)
  end

  it "rejects a start outside the run and a checkpoint of another seed" do
    trainer = checkpoint_trainer(spec_policy(4))
    expect_raises(ArgumentError, /outside 0..3/) { trainer.train(checkpoint_examples, 3, start: 4) }
    expect_raises(ArgumentError, /seed/) { trainer.train(checkpoint_examples, RLCD::Checkpoint.new(1, 3, 7_u64)) }
  end
end

describe "RLCD::Persistent#save" do
  it "replaces the file atomically and leaves no temporary file" do
    directory = File.tempname("rlcd_spec_save")
    Dir.mkdir(directory)
    path = File.join(directory, "model.rlcd")
    File.write(path, "old")
    policy = spec_policy(8)
    policy.save(path)
    Dir.children(directory).should eq(["model.rlcd"])
    max_param_delta(policy, RLCD::Policy.load(path)).should eq(0.0)
    File.delete(path)
    Dir.delete(directory)
  end
end
