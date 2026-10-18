# demo.cr
require "./src/rlcd"

def section(title : String)
  puts
  puts "=== #{title} ==="
end

def evaluate(policy : RLCD::Policy, examples : Array(RLCD::Example), temp : Float64 = 1.0)
  predictions = examples.map do |e|
    fwd   = policy.forward(e.features)
    probs = temp == 1.0 ? fwd.probs : RLCD.temperature_probs(fwd.logits, temp)
    {probs, e.label}
  end
  accuracy = predictions.count { |(p, l)| p.index(p.max) == l }.to_f / predictions.size
  ece, bins = RLCD.ece(predictions)
  {accuracy, ece, bins}
end

def print_bins(bins : Array(RLCD::Bin))
  puts "    confidence  accuracy    n"
  bins.each do |bin|
    next if bin.count == 0
    puts "    #{sprintf("%.2f-%.2f", bin.lower, bin.upper)}    #{sprintf("%.3f", bin.avg_confidence)}   #{sprintf("%.3f", bin.accuracy)}  #{bin.count}"
  end
end

def train_synthetic(noise : Float64, hidden : Int32, scoring : RLCD::ScoringRule, cal_weight : Float64, pg_weight : Float64, epochs : Int32, seed : UInt64)
  rng       = Random.new(seed)
  task      = RLCD::SyntheticTask.new(input_dim: 16, class_count: 4, noise: noise, rng: rng)
  train_set = task.generate(4096)
  val_set   = task.generate(512)
  test_set  = task.generate(512)
  policy    = RLCD::Policy.new(input_dim: 16, hidden_dim: hidden, option_count: 4, rng: rng)
  trainer   = RLCD::Trainer.new(policy: policy, scoring: scoring, calibration_weight: cal_weight, pg_weight: pg_weight, rng: rng)
  trainer.train(train_set, epochs)
  {policy, val_set, test_set}
end

rows = [] of Tuple(String, Float64, Float64)

section "1. Synthetic task - linear head vs MLP (log score)"
p_lin, _lin_val, lin_test = train_synthetic(1.2, 0, RLCD::ScoringRule::LogScore, 1.0, 0.0, 100, 101_u64)
acc, ece, _ = evaluate(p_lin, lin_test)
puts "  linear head:   accuracy=#{sprintf("%.3f", acc)}  ECE=#{sprintf("%.4f", ece)}"
rows << {"synthetic linear (log)", acc, ece}

p_mlp, mlp_val, mlp_test = train_synthetic(1.2, 32, RLCD::ScoringRule::LogScore, 1.0, 0.0, 100, 102_u64)
acc, ece, bins = evaluate(p_mlp, mlp_test)
puts "  MLP (tanh 32): accuracy=#{sprintf("%.3f", acc)}  ECE=#{sprintf("%.4f", ece)}"
rows << {"synthetic MLP (log)", acc, ece}
print_bins(bins)

section "2. Scoring rules - log vs brier (MLP)"
p_brier, _brier_val, brier_test = train_synthetic(1.2, 32, RLCD::ScoringRule::Brier, 1.0, 0.0, 100, 103_u64)
acc, ece, _ = evaluate(p_brier, brier_test)
puts "  brier reward:   accuracy=#{sprintf("%.3f", acc)}  ECE=#{sprintf("%.4f", ece)}"
rows << {"synthetic MLP (brier)", acc, ece}

section "3. Mode-sharpening - RLCD vs policy-gradient (noise 2.5)"
task      = RLCD::SyntheticTask.new(input_dim: 16, class_count: 4, noise: 2.5, rng: Random.new(7_u64))
train_set = task.generate(8192)
test_set  = task.generate(1024)

p_rlcd = RLCD::Policy.new(input_dim: 16, hidden_dim: 32, option_count: 4, rng: Random.new(7_u64))
RLCD::Trainer.new(policy: p_rlcd, scoring: RLCD::ScoringRule::LogScore, calibration_weight: 1.0, pg_weight: 0.0, rng: Random.new(7_u64)).train(train_set, 200)
acc, ece, bins = evaluate(p_rlcd, test_set)
puts "  RLCD (log-score reward): accuracy=#{sprintf("%.3f", acc)}  ECE=#{sprintf("%.4f", ece)}"
rows << {"RLCD log-score", acc, ece}
print_bins(bins)

p_pg = RLCD::Policy.new(input_dim: 16, hidden_dim: 32, option_count: 4, rng: Random.new(7_u64))
RLCD::Trainer.new(policy: p_pg, scoring: RLCD::ScoringRule::LogScore, calibration_weight: 0.0, pg_weight: 1.0, rng: Random.new(7_u64)).train(train_set, 200)
acc, ece, bins = evaluate(p_pg, test_set)
puts "  PG (0/1 correctness):    accuracy=#{sprintf("%.3f", acc)}  ECE=#{sprintf("%.4f", ece)}"
rows << {"PG correctness", acc, ece}
print_bins(bins)

section "4. Text tasks - yesno and intent (default curated)"
rng   = Random.new(42_u64)
yn    = RLCD::NoulTask.new(dim: 128, seed: 42_u64)
syn   = RLCD.split_examples(yn.examples, 0.7, 0.15, rng)
p_yn  = RLCD::Policy.new(input_dim: 128, hidden_dim: 32, option_count: 2, rng: rng)
RLCD::Trainer.new(policy: p_yn, scoring: RLCD::ScoringRule::LogScore, rng: rng).train(syn[0], 120)
acc, ece, _ = evaluate(p_yn, syn[2])
puts "  yesno:   accuracy=#{sprintf("%.3f", acc)}  ECE=#{sprintf("%.4f", ece)}"
rows << {"yesno (default curated)", acc, ece}

itask = RLCD::IntentTask.new(dim: 128, seed: 42_u64)
sint  = RLCD.split_examples(itask.examples, 0.7, 0.15, rng)
p_int = RLCD::Policy.new(input_dim: 128, hidden_dim: 32, option_count: 4, rng: rng)
RLCD::Trainer.new(policy: p_int, scoring: RLCD::ScoringRule::LogScore, rng: rng).train(sint[0], 120)
acc, ece, _ = evaluate(p_int, sint[2])
puts "  intent:  accuracy=#{sprintf("%.3f", acc)}  ECE=#{sprintf("%.4f", ece)}"
rows << {"intent (default curated)", acc, ece}

section "5. Injectable curated examples"
custom = [{"our production DB is down for all customers", 1}, {"a single-user typo in a footer", 0}]
cyn    = RLCD::NoulTask.new(dim: 128, seed: 42_u64, curated: custom)
exs    = cyn.examples
puts "  injected #{custom.size} curated examples; #{exs.size} total (#{exs.size - custom.size} generated)"
puts "  injected labels: #{exs[0, custom.size].map(&.label).join(", ")}"
puts "  default NoulTask curated count: #{RLCD::NoulTask.new(seed: 42_u64).examples.size - 1600}"

section "6. Temperature scaling"
val_logits = mlp_val.map { |e| p_mlp.forward(e.features).logits }
val_labels = mlp_val.map { |e| e.label }
temp       = RLCD.fit_temperature(val_logits, val_labels)
acc_u, ece_u, _ = evaluate(p_mlp, mlp_test, 1.0)
acc_c, ece_c, _ = evaluate(p_mlp, mlp_test, temp)
puts "  fitted T=#{sprintf("%.3f", temp)} on #{mlp_val.size} validation examples"
puts "  ECE before=#{sprintf("%.4f", ece_u)}  after=#{sprintf("%.4f", ece_c)}  (accuracy #{sprintf("%.3f", acc_u)} -> #{sprintf("%.3f", acc_c)})"
rows << {"synthetic MLP temperature-scaled", acc_c, ece_c}

section "7. Save / load round-trip"
path   = "/tmp/rlcd_demo_model.json"
p_mlp.save(path)
loaded = RLCD::Policy.load(path)
x      = mlp_test[0].features
p1     = p_mlp.probs(x)
p2     = loaded.probs(x)
max_d  = (0...p1.size).max_of { |i| (p1[i] - p2[i]).abs }
puts "  saved #{path}, reloaded; max |delta prob| = #{sprintf("%.6f", max_d)}"
File.delete(path) if File.exists?(path)

section "8. Score primitive (ordered levels)"
score_task = RLCD::ScoreTask.new(dim: 128, seed: 42_u64)
sset       = RLCD.split_examples(score_task.examples, 0.7, 0.15, rng)
p_score    = RLCD::Policy.new(input_dim: 128, hidden_dim: 32, option_count: 4, rng: rng)
RLCD::Trainer.new(policy: p_score, scoring: RLCD::ScoringRule::LogScore, rng: rng).train(sset[0], 120)
sample_probs = p_score.probs(sset[2][0].features)
mae = sset[2].sum { |e| (RLCD.expected_score(p_score.probs(e.features)) - e.label).abs } / sset[2].size
acc, ece, _ = evaluate(p_score, sset[2])
puts "  levels: #{score_task.legend.join(", ")}"
puts "  sample probs=#{sample_probs.map { |p| sprintf("%.3f", p) }.join(" ")} -> score #{sprintf("%.3f", RLCD.expected_score(sample_probs))} (label #{sset[2][0].label})"
puts "  held-out: accuracy=#{sprintf("%.3f", acc)}  ECE=#{sprintf("%.4f", ece)}  score MAE=#{sprintf("%.3f", mae)}"
rows << {"score (ordered levels)", acc, ece}

section "9. Confidence & structured state"
puts "  confidence uniform=#{sprintf("%.3f", RLCD.confidence([0.25, 0.25, 0.25, 0.25]))}  one-hot=#{sprintf("%.3f", RLCD.confidence([1.0, 0.0, 0.0, 0.0]))}"
state = JSON.parse(%q({"ticket": {"subject": "Duplicate charge", "messages": [{"text": "I was charged twice"}]}}))
f9    = RLCD::Featurizer.new(dim: 128, seed: 42_u64)
ref   = f9.features(state, "does `ticket.messages[0].text` request a refund?", "refund")
lit   = f9.features("ticket: subject: Duplicate charge messages: text: I was charged twice", "does I was charged twice request a refund?", "refund")
puts "  field-ref features == literal features: #{ref == lit}"

section "10. System One (multiple questions, one state)"
sys = RLCD::SystemOne.new(RLCD::Featurizer.new(dim: 128, seed: 42_u64), {2 => p_yn, 4 => p_int})
questions = [
  RLCD::Question.new("refund_requested", RLCD::Primitive::Noul, "does the customer request a refund?", ["no", "yes"]),
  RLCD::Question.new("department", RLCD::Primitive::Choice, "which team should handle this?", ["billing", "technical", "sales", "other"]),
]
sys.answer("I was charged twice for my last invoice and want my money back", questions).each do |a|
  case a
  in RLCD::NoulAnswer
    puts "  noul[#{a.id}]=#{sprintf("%.3f", a.noul)}"
  in RLCD::ChoiceAnswer
    puts "  choice[#{a.id}]=#{a.choice}  confidence=#{sprintf("%.3f", a.confidence)}"
  in RLCD::ScoreAnswer
    puts "  score[#{a.id}]=#{sprintf("%.3f", a.score)}  confidence=#{sprintf("%.3f", a.confidence)}"
  end
end

section "Summary"
puts "  #{"scenario".ljust(34)}  accuracy  ECE"
rows.each do |(name, a, e)|
  puts "  #{name.ljust(34)}  #{sprintf("%.3f", a)}   #{sprintf("%.4f", e)}"
end
