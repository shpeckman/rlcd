# rlcd

RLCD-style reinforcement learning in pure Crystal: training a decision model to output
**calibrated probability distributions** over a fixed option set, optimized against
**verifiable outcomes** with a **proper scoring rule** as the reward — the pattern behind
"reinforcement learning for calibrated decisions".

No dependencies. No FFI. The model, backprop, optimizer, and evaluation are all in `src/`.

This is the same idea behind TypeSafe AI's **Jev** "System One" model and its training
method, **RLCD** (Reinforcement Learning for Calibrated Decisions): unstructured state in,
typed probabilistic decisions out, with confidence that tracks accuracy.

## The idea

An LLM-style policy that generates text and gets a 0/1 correctness reward (plain RLVR)
learns to be *right*, and sharpens toward one-hot outputs regardless of how uncertain the
input actually is. A decision model instead reports a full probability distribution, and
the reward evaluates the distribution itself:

```
reward = S(p, y)     # p = reported distribution, y = verifiable ground truth
```

`S` is a **proper scoring rule** (log score or Brier score): it is maximized in
expectation only by reporting your true belief. Optimizing it turns "I don't know" into
the rational answer on ambiguous inputs, which is what makes the probabilities usable by
downstream code (thresholding, escalation, routing).

## Layout

```
rlcd/
├─╴demo.cr              # complete demo suite: every feature, with a summary table
├─╴shard.yml
├─╴src/
│  ├─╴main.cr              # CLI demo
│  ├─╴rlcd.cr              # Shard entry point
│  ╰─╴rlcd/
│     ├─╴answer.cr         # NoulAnswer / ChoiceAnswer / ScoreAnswer + confidence & score helpers
│     ├─╴calibration.cr    # Expected calibration error (ECE), reliability bins, temperature scaling, CSV dump
│     ├─╴featurizer.cr     # `Decision` record + `Featurizer`: signed hashed bag-of-words; structured JSON state + field refs
│     ├─╴mat.cr            # Minimal dense matrix (`Mat`): vector products, outer-product accumulation
│     ├─╴optimizer.cr      # `Adam` over the policy parameters (mutable learning rate)
│     ├─╴policy.cr         # `Policy`: 1-hidden-layer MLP (or linear head with `hidden_dim = 0`), softmax over options, categorical sampling, JSON save/load
│     ├─╴scoring.cr        # `ScoringRule::LogScore` / `ScoringRule::Brier`, exact logit gradients
│     ├─╴system_one.cr     # `Primitive` / `Question` + `SystemOne`: many typed questions against one state
│     ├─╴task.cr           # `SyntheticTask`: class-conditional Gaussians with irreducible noise — verifiable labels, nontrivial Bayes error
│     ├─╴tasks.cr          # `NoulTask` (yes/no) + `IntentTask` (choice) + `ScoreTask` (ordered levels)
│     ╰─╴trainer.cr        # `Trainer`: reward computation, optional GRPO-style policy-gradient term, mini-batch gradient accumulation, LR decay, backprop
╰─╴spec/
   ╰─╴rlcd_spec.cr
```

## Usage

```bash
crystal build --release -o bin/rlcd src/main.cr   # mkdir -p bin first
./bin/rlcd
```

Options: `--task synthetic|yesno|intent|score`, `--epochs`, `--lr`, `--lr-decay` (linear
decay fraction), `--batch` (mini-batch size), `--noise`, `--scoring log|brier`, `--lambda`
(proper-scoring reward weight), `--pg-weight` (sampled-correctness policy-gradient weight,
for comparison), `--group` (GRPO group size), `--train-size`, `--val-size`, `--test-size`,
`--feat-dim` (featurizer dimension for the text tasks), `--seed`, `--hidden` (`0` = linear
head), `--calibrate` (fit temperature on validation split), `--cal-csv PATH` (dump reliability
bins), `--save PATH`, `--load PATH`.

The `yesno`, `intent`, and `score` tasks run the pipeline end-to-end over natural language:
a `Featurizer` turns each `(state, question, criteria)` triple into a fixed-size vector, and
the same `Policy`/`Trainer`/ECE machinery learns calibrated predictions against the known
labels.

The demo trains a task with irreducible ambiguity and prints held-out accuracy, ECE, and a
reliability table (confidence vs. empirical accuracy per bin). The linear head (`--hidden 0`)
converges to ECE ≈ 0.016 on the default synthetic task, and `--calibrate` (temperature
scaling on the validation split) reduces ECE further; the text tasks (`--task yesno` /
`--task intent`) reach near-zero ECE on their seeded, verifiable labels.

Mode-sharpening comparison (a tanh MLP at `--noise 2.5`):

Objective                                               | Accuracy | ECE
--------------------------------------------------------|----------|---------
RLCD: log-score reward (`--lambda 1`, default)          | 0.56     | **0.11**
PG: 0/1 correctness reward (`--lambda 0 --pg-weight 1`) | 0.57     | **0.42**

Same accuracy, but the PG baseline says "99% sure" while being right 58% of the time;
the proper-scoring-rule model's confidence tracks its actual hit rate.

## Library API

```crystal
require "rlcd"

policy  = RLCD::Policy.new(input_dim: 16, hidden_dim: 32, option_count: 4, rng: Random.new(42))
trainer = RLCD::Trainer.new(policy: policy, scoring: RLCD::ScoringRule::LogScore, batch_size: 32)
trainer.train(examples, epochs: 200) { |stats| puts stats }

probs = policy.probs(features)                       # => [0.7, 0.1, 0.15, 0.05]
ece, bins = RLCD.ece(predictions)                    # predictions: Array({Array(Float64), Int32})

featurizer = RLCD::Featurizer.new(dim: 128)
features   = featurizer.features(state, question, criteria)

yesno_task  = RLCD::NoulTask.new(seed: 42_u64)
intent_task = RLCD::IntentTask.new(seed: 42_u64)
score_task  = RLCD::ScoreTask.new(seed: 42_u64)      # ordered levels (ordinal)

yesno_task  = RLCD::NoulTask.new(seed: 42_u64, curated: [{"custom escalation", 1}, {"custom non-escalation", 0}])
intent_task = RLCD::IntentTask.new(seed: 42_u64, curated: [{"custom intent", 2}])

conf  = RLCD.confidence(probs)                       # (p_max - 1/n) / (1 - 1/n)
pos   = RLCD.expected_score(probs)                   # probability-weighted level
sconf = RLCD.score_confidence(probs)                 # ordinal peakedness

sys = RLCD::SystemOne.new(featurizer, {2 => yesno_policy, 4 => choice_policy})
answers = sys.answer(state, [
  RLCD::Question.new("refund", RLCD::Primitive::Noul, "does it request a refund?", ["no", "yes"]),
  RLCD::Question.new("team", RLCD::Primitive::Choice, "which team?", ["billing", "technical", "sales"]),
])

t = RLCD.fit_temperature(logits, labels)             # temperature scaling on a validation split
scaled = RLCD.temperature_probs(logits, t)

policy.save("model.json")
loaded = RLCD::Policy.load("model.json")
```

`RLCD::Example` is a `record Example, features : Array(Float64), label : Int32`. For the
synthetic task the features are raw Gaussian vectors; for the text tasks they come from a
`Featurizer` over `(state, question, criteria)` (`RLCD::Decision`). Everything downstream of
the features is shared. `NoulTask` (yes/no), `IntentTask` (choice), and `ScoreTask` (ordered
levels) mirror Jev's `noul`/`choice`/`score` primitives, and each seeds its generated
examples with a small `curated` list; pass your own via `curated:`.

`Featurizer` also accepts a structured JSON `state` and resolves backtick field references in
the question and criteria (`` `ticket.messages[0].text` ``). `SystemOne` bundles many typed
questions against one state into a single call, dispatching each to the policy whose option
count matches its criteria.

## Development

```bash
crystal spec          # gradient checks (finite differences), calibration sanity, training smoke test
crystal build --release -o bin/rlcd src/main.cr
crystal run demo.cr   # complete demo suite: every feature, with a side-by-side summary table
```
