# spec/rlcd_spec.cr
require "spec"
require "../src/rlcd"

private def softmax_of(logits : Array(Float64)) : Array(Float64)
  max  = logits.max
  exps = logits.map { |v| Math.exp(v - max) }
  sum  = exps.sum
  exps.map { |v| v / sum }
end

describe Rlcd::Policy do
  it "produces a probability distribution" do
    policy = Rlcd::Policy.new(input_dim: 4, hidden_dim: 8, option_count: 3, rng: Random.new(1))
    probs  = policy.probs([0.5, -0.2, 0.1, 0.9])
    probs.sum.should be_close(1.0, 1e-9)
    probs.each { |p| p.should be > 0.0 }
  end

  it "samples valid option indices" do
    policy = Rlcd::Policy.new(input_dim: 4, hidden_dim: 8, option_count: 3, rng: Random.new(2))
    100.times do
      a = policy.sample([0.5, -0.2, 0.1, 0.9])
      (0...3).should contain(a)
    end
  end
end

describe Rlcd::ScoringRule do
  it "prefers the true distribution (propriety)" do
    truth = [0.1, 0.7, 0.15, 0.05]
    expected = ->(q : Array(Float64)) do
      truth.each_with_index.sum { |p_y, y| p_y * Rlcd.score(Rlcd::ScoringRule::Brier, q, y) }
    end
    honest        = expected.call(truth)
    overconfident = expected.call([0.01, 0.97, 0.01, 0.01])
    hedged        = expected.call([0.25, 0.25, 0.25, 0.25])
    honest.should be > overconfident
    honest.should be > hedged
  end

  it "brier logit gradient matches finite differences" do
    rng    = Random.new(7)
    logits = Array.new(4) { rng.next_float * 2.0 - 1.0 }
    label  = 2
    probs  = softmax_of(logits)
    grad   = Rlcd.score_grad_logits(Rlcd::ScoringRule::Brier, probs, label)
    eps    = 1e-6
    logits.size.times do |k|
      plus  = logits.dup
      minus = logits.dup
      plus[k] += eps
      minus[k] -= eps
      numerical = (Rlcd.score(Rlcd::ScoringRule::Brier, softmax_of(plus), label) -
                   Rlcd.score(Rlcd::ScoringRule::Brier, softmax_of(minus), label)) / (2.0 * eps)
      grad[k].should be_close(numerical, 1e-6)
    end
  end

  it "log-score logit gradient matches finite differences" do
    rng    = Random.new(8)
    logits = Array.new(5) { rng.next_float * 2.0 - 1.0 }
    label  = 0
    probs  = softmax_of(logits)
    grad   = Rlcd.score_grad_logits(Rlcd::ScoringRule::LogScore, probs, label)
    eps    = 1e-6
    logits.size.times do |k|
      plus  = logits.dup
      minus = logits.dup
      plus[k] += eps
      minus[k] -= eps
      numerical = (Rlcd.score(Rlcd::ScoringRule::LogScore, softmax_of(plus), label) -
                   Rlcd.score(Rlcd::ScoringRule::LogScore, softmax_of(minus), label)) / (2.0 * eps)
      grad[k].should be_close(numerical, 1e-6)
    end
  end
end

describe Rlcd do
  it "ece is zero for perfect confidence and accuracy" do
    predictions = Array(Rlcd::Prediction).new
    50.times { predictions << {[0.0, 1.0, 0.0], 1} }
    ece, _ = Rlcd.ece(predictions)
    ece.should be_close(0.0, 1e-12)
  end

  it "ece is zero when confidence matches the hit rate" do
    rng         = Random.new(3)
    predictions = Array(Rlcd::Prediction).new
    20000.times do
      hit   = rng.next_float < 0.25
      probs = [0.25, 0.25, 0.25, 0.25]
      predictions << {probs, hit ? 0 : 1}
    end
    ece, _ = Rlcd.ece(predictions)
    ece.should be < 0.01
  end

  it "training improves the proper score on a verifiable task" do
    rng       = Random.new(11)
    task      = Rlcd::SyntheticTask.new(input_dim: 8, class_count: 3, noise: 0.8, rng: rng)
    train_set = task.generate(200)
    policy    = Rlcd::Policy.new(input_dim: 8, hidden_dim: 16, option_count: 3, rng: rng)

    mean_score = -> do
      train_set.sum { |e| Rlcd.score(Rlcd::ScoringRule::Brier, policy.probs(e.features), e.label) } / train_set.size
    end

    before  = mean_score.call
    trainer = Rlcd::Trainer.new(policy: policy, lr: 0.01, group_size: 8, rng: rng)
    trainer.train(train_set, 30)
    mean_score.call.should be > before
  end
end

describe Rlcd::Featurizer do
  it "is deterministic and normalizes non-empty input" do
    f = Rlcd::Featurizer.new(dim: 64, seed: 99_u64)
    a = f.features("The payment failed", "Should I escalate?", "critical severity")
    b = f.features("The payment failed", "Should I escalate?", "critical severity")
    a.should eq(b)
    a.size.should eq(64)
    norm = Math.sqrt(a.sum { |v| v * v })
    norm.should be_close(1.0, 1e-9)
  end

  it "returns a zero vector for empty input" do
    f = Rlcd::Featurizer.new(dim: 32, seed: 99_u64)
    f.features("", "", "").all?(&.zero?).should be_true
  end
end

describe Rlcd::NoulTask do
  it "produces binary labeled examples with the featurizer dimension" do
    task = Rlcd::NoulTask.new(dim: 64, seed: 42_u64)
    exs  = task.examples
    exs.size.should be > 0
    exs.each do |e|
      (0..1).should contain(e.label)
      e.features.size.should eq(64)
    end
  end
end

describe Rlcd::IntentTask do
  it "produces 4-class labeled examples with the featurizer dimension" do
    task = Rlcd::IntentTask.new(dim: 64, seed: 42_u64)
    exs  = task.examples
    exs.size.should be > 0
    exs.each do |e|
      (0..3).should contain(e.label)
      e.features.size.should eq(64)
    end
  end
end

describe "Rlcd::Policy (linear head)" do
  it "produces a probability distribution" do
    policy = Rlcd::Policy.new(input_dim: 4, hidden_dim: 0, option_count: 3, rng: Random.new(5))
    probs  = policy.probs([0.5, -0.2, 0.1, 0.9])
    probs.sum.should be_close(1.0, 1e-9)
    probs.each { |p| p.should be > 0.0 }
  end

  it "improves the proper score on a verifiable task" do
    rng       = Random.new(12)
    task      = Rlcd::SyntheticTask.new(input_dim: 8, class_count: 3, noise: 0.8, rng: rng)
    train_set = task.generate(200)
    policy    = Rlcd::Policy.new(input_dim: 8, hidden_dim: 0, option_count: 3, rng: rng)

    mean_score = -> do
      train_set.sum { |e| Rlcd.score(Rlcd::ScoringRule::Brier, policy.probs(e.features), e.label) } / train_set.size
    end

    before  = mean_score.call
    trainer = Rlcd::Trainer.new(policy: policy, lr: 0.01, group_size: 8, batch_size: 32, rng: rng)
    trainer.train(train_set, 30)
    mean_score.call.should be > before
  end
end

describe "Rlcd temperature scaling" do
  it "returns a positive finite temperature" do
    rng    = Random.new(13)
    logits = Array.new(50) { Array.new(3) { rng.next_float * 4.0 - 2.0 } }
    labels = Array.new(50) { rng.rand(3) }
    t      = Rlcd.fit_temperature(logits, labels)
    t.should be > 0.0
    t.finite?.should be_true
  end

  it "softens overconfident logits" do
    logits = [] of Array(Float64)
    labels = [] of Int32
    50.times { logits << [10.0, 0.0, 0.0]; labels << 0 }
    50.times { logits << [10.0, 0.0, 0.0]; labels << 1 }
    Rlcd.fit_temperature(logits, labels).should be > 1.0
  end
end

describe "Rlcd::Policy persistence" do
  it "save/load round-trips identical probabilities" do
    policy = Rlcd::Policy.new(input_dim: 4, hidden_dim: 8, option_count: 3, rng: Random.new(9))
    path   = "/tmp/rlcd_spec_model.json"
    policy.save(path)
    loaded = Rlcd::Policy.load(path)
    x      = [0.5, -0.2, 0.1, 0.9]
    p1     = policy.probs(x)
    p2     = loaded.probs(x)
    p1.size.times { |i| p2[i].should be_close(p1[i], 1e-9) }
    File.delete(path) if File.exists?(path)
  end
end

describe "Rlcd mini-batch training" do
  it "improves the proper score with batching" do
    rng       = Random.new(14)
    task      = Rlcd::SyntheticTask.new(input_dim: 8, class_count: 3, noise: 0.8, rng: rng)
    train_set = task.generate(200)
    policy    = Rlcd::Policy.new(input_dim: 8, hidden_dim: 16, option_count: 3, rng: rng)

    mean_score = -> do
      train_set.sum { |e| Rlcd.score(Rlcd::ScoringRule::Brier, policy.probs(e.features), e.label) } / train_set.size
    end

    before  = mean_score.call
    trainer = Rlcd::Trainer.new(policy: policy, lr: 0.01, group_size: 8, batch_size: 64, rng: rng)
    trainer.train(train_set, 30)
    mean_score.call.should be > before
  end
end
