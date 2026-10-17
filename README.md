# rlcd

RLCD-style reinforcement learning in pure Crystal: training a decision model to output
**calibrated probability distributions** over a fixed option set, optimized against
**verifiable outcomes** with a **proper scoring rule** as the reward — the pattern behind
"reinforcement learning for calibrated decisions".

No dependencies. No FFI. The model, backprop, optimizer, and evaluation are all in `src/`.

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

| File | Contents |
|---|---|
| `src/rlcd/mat.cr` | Minimal dense matrix (`Mat`): vector products, outer-product accumulation |
| `src/rlcd/policy.cr` | `Policy`: 1-hidden-layer MLP (or linear head with `hidden_dim = 0`), softmax over options, categorical sampling, JSON save/load |
| `src/rlcd/scoring.cr` | `ScoringRule::LogScore` / `ScoringRule::Brier`, exact logit gradients |
| `src/rlcd/trainer.cr` | `Trainer`: reward computation, optional GRPO-style policy-gradient term, mini-batch gradient accumulation, LR decay, backprop |
| `src/rlcd/optimizer.cr` | `Adam` over the policy parameters (mutable learning rate) |
| `src/rlcd/calibration.cr` | Expected calibration error (ECE), reliability bins, temperature scaling, CSV dump |
| `src/rlcd/featurizer.cr` | `Decision` record + `Featurizer`: signed hashed bag-of-words over (state, question, criteria) |
| `src/rlcd/task.cr` | `SyntheticTask`: class-conditional Gaussians with irreducible noise — verifiable labels, nontrivial Bayes error |
| `src/rlcd/tasks.cr` | `NoulTask` (binary yes/no) + `IntentTask` (multi-class intent): realistic, seeded, verifiable tasks |
| `src/main.cr` | CLI demo |

## Usage

```bash
crystal build --release -o bin/rlcd src/main.cr   # mkdir -p bin first
./bin/rlcd
```

Options: `--task synthetic|yesno|intent`, `--epochs`, `--lr`, `--lr-decay` (linear decay
fraction), `--batch` (mini-batch size), `--noise`, `--scoring log|brier`, `--lambda`
(proper-scoring reward weight), `--pg-weight` (sampled-correctness policy-gradient weight,
for comparison), `--group` (GRPO group size), `--train-size`, `--val-size`, `--test-size`,
`--feat-dim` (featurizer dimension for the text tasks), `--seed`, `--hidden` (`0` = linear
head), `--calibrate` (fit temperature on validation split), `--cal-csv PATH` (dump reliability
bins), `--save PATH`, `--load PATH`.

The `yesno` and `intent` tasks run the pipeline end-to-end over natural language: a
`Featurizer` turns each `(state, question, criteria)` triple into a fixed-size vector, and
the same `Policy`/`Trainer`/ECE machinery learns calibrated predictions against the known
labels.

The demo trains on a synthetic 4-class task with irreducible ambiguity and prints held-out
accuracy, ECE, and a reliability table (confidence vs. empirical accuracy per bin).

Typical result at `--noise 2.5`:

| Objective | Accuracy | ECE |
|---|---|---|
| RLCD: log-score reward (`--lambda 1`, default) | 0.56 | **0.11** |
| PG: 0/1 correctness reward (`--lambda 0 --pg-weight 1`) | 0.57 | **0.42** |

Same accuracy, but the PG baseline says "99% sure" while being right 58% of the time;
the proper-scoring-rule model's confidence tracks its actual hit rate.

## Library API

```crystal
require "rlcd"

policy  = Rlcd::Policy.new(input_dim: 16, hidden_dim: 32, option_count: 4, rng: Random.new(42))
trainer = Rlcd::Trainer.new(policy: policy, scoring: Rlcd::ScoringRule::LogScore, batch_size: 32)
trainer.train(examples, epochs: 200) { |stats| puts stats }

probs = policy.probs(features)                       # => [0.7, 0.1, 0.15, 0.05]
ece, bins = Rlcd.ece(predictions)                    # predictions: Array({Array(Float64), Int32})

featurizer = Rlcd::Featurizer.new(dim: 128)
features   = featurizer.features(state, question, criteria)

yesno_task = Rlcd::NoulTask.new(seed: 42_u64)
intent_task = Rlcd::IntentTask.new(seed: 42_u64)

t = Rlcd.fit_temperature(logits, labels)             # temperature scaling on a validation split
scaled = Rlcd.temperature_probs(logits, t)

policy.save("model.json")
loaded = Rlcd::Policy.load("model.json")
```

`Rlcd::Example` is a `record Example, features : Array(Float64), label : Int32`. For the
synthetic task the features are raw Gaussian vectors; for the text tasks they come from a
`Featurizer` over `(state, question, criteria)` (`Rlcd::Decision`). Everything downstream of
the features is shared.

## Development

```bash
crystal spec          # gradient checks (finite differences), calibration sanity, training smoke test
crystal build --release -o bin/rlcd src/main.cr
```
