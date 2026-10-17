# src/main.cr
require "./rlcd"
require "option_parser"

task_name          = "synthetic"
epochs             = 200
group_size         = 8
batch_size         = 32
lr                 = 0.005
lr_decay           = 0.0
calibration_weight = 1.0
pg_weight          = 0.0
noise              = 1.5
hidden             = 32
seed               = 42_u64
scoring            = Rlcd::ScoringRule::LogScore
train_size         = 8192
val_size           = 1024
test_size          = 512
feat_dim           = 128
calibrate          = false
cal_csv            = nil.as(String?)
save_path          = nil.as(String?)
load_path          = nil.as(String?)

OptionParser.parse do |parser|
  parser.banner = "Usage: rlcd [options]"
  parser.on("--task NAME", "Task: synthetic | yesno | intent (default #{task_name})") { |v| task_name = v }
  parser.on("--epochs N", "Training epochs (default #{epochs})") { |v| epochs = v.to_i }
  parser.on("--group N", "GRPO group size (default #{group_size})") { |v| group_size = v.to_i }
  parser.on("--batch N", "Mini-batch size (default #{batch_size})") { |v| batch_size = v.to_i }
  parser.on("--lr F", "Learning rate (default #{lr})") { |v| lr = v.to_f }
  parser.on("--lr-decay F", "Linear LR decay fraction over epochs, 0 = off (default #{lr_decay})") { |v| lr_decay = v.to_f }
  parser.on("--lambda F", "Proper-scoring-rule reward weight (default #{calibration_weight})") { |v| calibration_weight = v.to_f }
  parser.on("--pg-weight F", "Sampled-correctness REINFORCE weight; >0 reintroduces mode-sharpening (default #{pg_weight})") { |v| pg_weight = v.to_f }
  parser.on("--noise F", "Task noise (synthetic) (default #{noise})") { |v| noise = v.to_f }
  parser.on("--train-size N", "Training examples (synthetic) (default #{train_size})") { |v| train_size = v.to_i }
  parser.on("--val-size N", "Validation examples (synthetic) (default #{val_size})") { |v| val_size = v.to_i }
  parser.on("--test-size N", "Held-out examples (synthetic) (default #{test_size})") { |v| test_size = v.to_i }
  parser.on("--feat-dim N", "Featurizer dimension (yesno/intent) (default #{feat_dim})") { |v| feat_dim = v.to_i }
  parser.on("--hidden N", "Hidden layer size, 0 = linear head (default #{hidden})") { |v| hidden = v.to_i }
  parser.on("--seed N", "RNG seed (default #{seed})") { |v| seed = v.to_u64 }
  parser.on("--scoring NAME", "Proper scoring rule: log | brier (default log)") do |v|
    scoring = v == "brier" ? Rlcd::ScoringRule::Brier : Rlcd::ScoringRule::LogScore
  end
  parser.on("--calibrate", "Fit temperature on validation split, report ECE before/after") { calibrate = true }
  parser.on("--cal-csv PATH", "Dump reliability bins CSV") { |v| cal_csv = v }
  parser.on("--save PATH", "Save trained weights to JSON") { |v| save_path = v }
  parser.on("--load PATH", "Load weights from JSON, skip training") { |v| load_path = v }
  parser.on("-h", "--help", "Show this help") { puts parser; exit }
end

rng = Random.new(seed)

case task_name
when "yesno"
  t            = Rlcd::NoulTask.new(dim: feat_dim, seed: seed)
  input_dim    = t.featurizer.dim
  option_count = t.option_count
  sets         = Rlcd.split_examples(t.examples, 0.7, 0.15, rng)
  train_set    = sets[0]
  val_set      = sets[1]
  test_set     = sets[2]
when "intent"
  t            = Rlcd::IntentTask.new(dim: feat_dim, seed: seed)
  input_dim    = t.featurizer.dim
  option_count = t.option_count
  sets         = Rlcd.split_examples(t.examples, 0.7, 0.15, rng)
  train_set    = sets[0]
  val_set      = sets[1]
  test_set     = sets[2]
else
  t            = Rlcd::SyntheticTask.new(input_dim: 16, class_count: 4, noise: noise, rng: rng)
  input_dim    = 16
  option_count = 4
  train_set    = t.generate(train_size)
  val_set      = t.generate(val_size)
  test_set     = t.generate(test_size)
end

policy =
  if lp = load_path
    Rlcd::Policy.load(lp, rng)
  else
    Rlcd::Policy.new(input_dim: input_dim, hidden_dim: hidden, option_count: option_count, rng: rng)
  end

probs_for = ->(features : Array(Float64), temp : Float64) do
  fwd = policy.forward(features)
  temp == 1.0 ? fwd.probs : Rlcd.temperature_probs(fwd.logits, temp)
end

evaluate = ->(examples : Array(Rlcd::Example), temp : Float64) do
  predictions = examples.map { |e| {probs_for.call(e.features, temp), e.label} }
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

puts "Task: #{task_name}"
puts "RLCD reward: #{scoring} proper scoring rule, lambda=#{calibration_weight}"
puts "Policy-gradient mode-sharpening term: pg-weight=#{pg_weight} (group=#{group_size})"
puts "Mini-batch=#{batch_size}, lr=#{lr}, lr-decay=#{lr_decay}, hidden=#{hidden}, epochs=#{epochs}"
puts

if lp = load_path
  puts "Loaded weights from #{lp}; skipping training"
else
  acc0, ece0, _ = evaluate.call(test_set, 1.0)
  puts "Before training: accuracy=#{sprintf("%.3f", acc0)}  ECE=#{sprintf("%.4f", ece0)}"

  trainer = Rlcd::Trainer.new(
    policy: policy,
    lr: lr,
    group_size: group_size,
    scoring: scoring,
    calibration_weight: calibration_weight,
    pg_weight: pg_weight,
    batch_size: batch_size,
    lr_decay: lr_decay,
    rng: rng,
  )

  trainer.train(train_set, epochs) do |stats|
    next unless stats.epoch % (epochs // 5).clamp(1, epochs) == 0 || stats.epoch == 1
    puts "  epoch #{sprintf("%4d", stats.epoch)}  sampled-accuracy=#{sprintf("%.3f", stats.accuracy)}  reward=#{sprintf("%.3f", stats.mean_reward)}  score=#{sprintf("%.4f", stats.mean_score)}"
  end

  if sp = save_path
    policy.save(sp)
    puts "Saved weights to #{sp}"
  end
end

puts
temperature = 1.0
if calibrate
  val_logits  = val_set.map { |e| policy.forward(e.features).logits }
  val_labels  = val_set.map { |e| e.label }
  temperature = Rlcd.fit_temperature(val_logits, val_labels)
  puts "Fitted temperature T=#{sprintf("%.3f", temperature)} on #{val_set.size} validation examples"
end

stage = load_path ? "After loading" : "After training"
acc1, ece1, bins1 = evaluate.call(test_set, temperature)
puts "#{stage}:  accuracy=#{sprintf("%.3f", acc1)}  ECE=#{sprintf("%.4f", ece1)}"

if calibrate
  acc_uncal, ece_uncal, _ = evaluate.call(test_set, 1.0)
  puts "Before temperature scaling:  accuracy=#{sprintf("%.3f", acc_uncal)}  ECE=#{sprintf("%.4f", ece_uncal)}"
end

puts
puts "Reliability table (held-out#{calibrate ? ", temperature-scaled" : ""}):"
print_bins.call(bins1)

if csv = cal_csv
  Rlcd.write_bins_csv(bins1, csv, ece1)
  puts "Wrote reliability CSV to #{csv}"
end
