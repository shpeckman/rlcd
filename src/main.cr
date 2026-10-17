# src/main.cr
require "./rlcd"
require "option_parser"

epochs             = 200
group_size         = 8
lr                 = 0.005
calibration_weight = 1.0
pg_weight          = 0.0
noise              = 1.5
hidden             = 32
seed               = 42_u64
scoring            = Rlcd::ScoringRule::LogScore
train_size         = 8192
test_size          = 512

OptionParser.parse do |parser|
  parser.banner = "Usage: rlcd [options]"
  parser.on("--epochs N", "Training epochs (default #{epochs})") { |v| epochs = v.to_i }
  parser.on("--group N", "GRPO group size for the policy-gradient term (default #{group_size})") { |v| group_size = v.to_i }
  parser.on("--lr F", "Learning rate (default #{lr})") { |v| lr = v.to_f }
  parser.on("--lambda F", "Proper-scoring-rule reward weight (default #{calibration_weight})") { |v| calibration_weight = v.to_f }
  parser.on("--pg-weight F", "Sampled-correctness REINFORCE weight; >0 reintroduces mode-sharpening (default #{pg_weight})") { |v| pg_weight = v.to_f }
  parser.on("--noise F", "Task noise, higher = more ambiguity (default #{noise})") { |v| noise = v.to_f }
  parser.on("--train-size N", "Training examples (default #{train_size})") { |v| train_size = v.to_i }
  parser.on("--test-size N", "Held-out examples (default #{test_size})") { |v| test_size = v.to_i }
  parser.on("--hidden N", "Hidden layer size (default #{hidden})") { |v| hidden = v.to_i }
  parser.on("--seed N", "RNG seed (default #{seed})") { |v| seed = v.to_u64 }
  parser.on("--scoring NAME", "Proper scoring rule: log | brier (default log)") do |v|
    scoring = v == "brier" ? Rlcd::ScoringRule::Brier : Rlcd::ScoringRule::LogScore
  end
  parser.on("-h", "--help", "Show this help") { puts parser; exit }
end

rng       = Random.new(seed)
task      = Rlcd::SyntheticTask.new(input_dim: 16, class_count: 4, noise: noise, rng: rng)
train_set = task.generate(train_size)
test_set  = task.generate(test_size)

policy = Rlcd::Policy.new(input_dim: 16, hidden_dim: hidden, option_count: 4, rng: rng)

evaluate = -> do
  predictions = test_set.map { |e| {policy.probs(e.features), e.label} }
  accuracy    = predictions.count { |(probs, label)| probs.index(probs.max) == label }.to_f / predictions.size
  ece, bins = Rlcd.ece(predictions)
  {accuracy, ece, bins}
end

print_bins = ->(bins : Array(Rlcd::Bin)) do
  puts "  confidence  accuracy   n"
  bins.each do |bin|
    next if bin.count == 0
    puts "  #{sprintf("%.2f-%.2f", bin.lower, bin.upper)}      #{sprintf("%.3f", bin.avg_confidence)}     #{sprintf("%.3f", bin.accuracy)}  #{bin.count}"
  end
end

puts "Task: 4-class Gaussian classification, noise=#{noise} (irreducible ambiguity by design)"
puts "RLCD reward: #{scoring} proper scoring rule on the reported distribution (verifiable outcome), lambda=#{calibration_weight}"
puts "Policy-gradient correctness term (mode-sharpening baseline): pg-weight=#{pg_weight} (group=#{group_size})"
puts

acc0, ece0, bins0 = evaluate.call
puts "Before training: accuracy=#{sprintf("%.3f", acc0)}  ECE=#{sprintf("%.4f", ece0)}"

trainer = Rlcd::Trainer.new(
  policy: policy,
  lr: lr,
  group_size: group_size,
  scoring: scoring,
  calibration_weight: calibration_weight,
  pg_weight: pg_weight,
  rng: rng,
)

trainer.train(train_set, epochs) do |stats|
  next unless stats.epoch % (epochs // 5).clamp(1, epochs) == 0 || stats.epoch == 1
  puts "  epoch #{sprintf("%4d", stats.epoch)}  sampled-accuracy=#{sprintf("%.3f", stats.accuracy)}  reward=#{sprintf("%.3f", stats.mean_reward)}  score=#{sprintf("%.4f", stats.mean_score)}"
end

puts
acc1, ece1, bins1 = evaluate.call
puts "After training:  accuracy=#{sprintf("%.3f", acc1)}  ECE=#{sprintf("%.4f", ece1)}"
puts
puts "Reliability table (held-out):"
print_bins.call(bins1)
