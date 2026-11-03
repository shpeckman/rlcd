# AGENTS.md

Orientation for AI agents working on this repository.

## What this is

`rlcd` — RLCD-style RL in pure Crystal: a small MLP policy trained to output calibrated
probability distributions over a fixed option set, with a proper scoring rule (log score /
Brier) as the reward against verifiable ground-truth labels. Zero dependencies.

## Layout

- `src/rlcd.cr` — namespace, `VERSION`, requires everything under `src/rlcd/`
- `src/rlcd/answer.cr` — `NoulAnswer` / `ChoiceAnswer` / `ScoreAnswer` records and
  `RLCD.confidence` / `RLCD.expected_score` / `RLCD.score_confidence`
- `src/rlcd/mat.cr` — `Mat` dense matrix (Slice-backed): `mul_vec`, `t_mul_vec`, `add_outer!`
- `src/rlcd/policy.cr` — `Policy` MLP (tanh hidden, softmax head; `hidden_dim = 0` = linear
  head), `param_slices`, JSON `save`/`load`, `Forward` cache record
- `src/rlcd/scoring.cr` — `ScoringRule` enum, `RLCD.score`, `RLCD.score_grad_logits`
- `src/rlcd/trainer.cr` — `Trainer`: proper-score reward + optional GRPO policy-gradient
  term (`pg_weight`), mini-batch gradient accumulation, linear LR decay, backprop,
  `Example` / `Stats` records
- `src/rlcd/optimizer.cr` — `Adam` (operates on `policy.param_slices`, mutable `lr`)
- `src/rlcd/calibration.cr` — `RLCD.ece` + `Bin` reliability table, `RLCD.fit_temperature` /
  `RLCD.temperature_probs`, `RLCD.write_bins_csv`
- `src/rlcd/featurizer.cr` — `Decision` record + `Featurizer` signed hashed bag-of-words
  over (state, question, criteria); opt-in `code: true` symbols and character 2–5-grams;
  structured JSON state + backtick field references
- `src/rlcd/system_one.cr` — `Primitive` enum, `Question` record, `SystemOne` (many typed
  questions answered against one state)
- `src/rlcd/task.cr` — `SyntheticTask` Gaussian classification with irreducible noise
- `src/rlcd/tasks.cr` — `NoulTask` (yes/no) + `IntentTask` (choice) + `ScoreTask` (ordered
  levels), `RLCD.split_examples`
- `src/main.cr` — CLI demo binary (`rlcd` target)
- `demo.cr` — comprehensive demo suite
- `spec/rlcd_spec.cr` — specs, including finite-difference gradient checks

## Commands

```bash
crystal spec
mkdir -p bin && crystal build --release -o bin/rlcd src/main.cr
./bin/rlcd --noise 2.5
```

## Code standards

- Every file starts with a comment containing its own path; no other comments.
- Compiler-friendly, performance-focused, idiomatic Crystal; data-driven design.
- No functionality beyond what was asked; complete files, no stubs.
- Do not touch the `version` field in `shard.yml` (keep `VERSION` in sync only if asked).
- `out` is a reserved keyword — never use it as an identifier.
- `crystal build -o bin/NAME` fails with a bare `collect2` linker error if the output
  directory does not exist — `mkdir -p` first.
- Ivars assigned from other ivars in `initialize` need explicit type annotations at
  class level, or inference fails.
- Keep `README.md` and this file in sync with code changes.

## Design notes

- The RLCD reward is the proper scoring rule evaluated on the reported distribution;
  the gradient is exact (`score_grad_logits`), group sampling only enters via the
  optional `pg_weight` correctness term, which exists to *demonstrate* mode-sharpening.
- Propriety holds in expectation over outcomes — specs must evaluate expected score,
  never a single realized outcome.
