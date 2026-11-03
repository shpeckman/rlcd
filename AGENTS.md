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
  head), `param_slices`, `layout` (name, rows, cols per tensor, same order), `shape_error`,
  JSON/binary codec, `Forward` cache record
- `src/rlcd/scoring.cr` — `ScoringRule` enum, `RLCD.score`, `RLCD.score_grad_logits`
- `src/rlcd/trainer.cr` — `Trainer`: proper-score reward + optional GRPO policy-gradient
  term (`pg_weight`), mini-batch gradient accumulation, linear LR decay, backprop, optional
  `optimizer:` to resume, `seed:` for per-epoch shuffling, `train(…, start:)` / `train(…, checkpoint)`,
  `epoch` / `checkpoint`, `Example` / `Stats` records
- `src/rlcd/checkpoint.cr` — `Checkpoint` record (epoch, epochs, seed; `finished?`), JSON/binary codec
- `src/rlcd/optimizer.cr` — `Adam` (operates on `policy.param_slices`, mutable `lr`, bound
  `policy`, step count `t`, moments `m`/`v`), JSON/binary codec
- `src/rlcd/calibration.cr` — `RLCD.ece` + `Bin` reliability table, `RLCD.fit_temperature` /
  `RLCD.temperature_probs`, `RLCD.write_bins_csv`
- `src/rlcd/featurizer.cr` — `Decision` record + `Featurizer` signed hashed bag-of-words
  over (state, question, criteria); opt-in `code: true` symbols and character 2–5-grams;
  structured JSON state + backtick field references; JSON/binary codec
- `src/rlcd/model.cr` — `Model`: featurizer + policy + labels + temperature + optional `Adam` + optional `Checkpoint`;
  `probs` / `predict`; JSON/binary codec over the container chunks
- `src/rlcd/format.cr` — `FormatError`, `Layout` alias, `Format` (JSON / binary, `from_path`,
  `detect`), `Precision` (F64 / F32 / I8 tensor codec, `moment_precision`)
- `src/rlcd/container.cr` — `Container`: binary header, `TAG` table, `Writer`, `Chunks`
  (`decode` / `decode?` check that each payload is consumed exactly), CRC-32 trailer
- `src/rlcd/checksum.cr` — `Checksum`: pure-Crystal CRC-32 IO wrapper (no zlib, no FFI)
- `src/rlcd/persistent.cr` — `Persistent` mixin: `save(path, format, precision)` (atomic: temp file, fsync,
  rename) and `self.load(path, rng)` for `Policy` and `Model`
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
- Persistence follows Crystal's protocols: `to_json(JSON::Builder)` / `self.new(JSON::PullParser)`
  and `to_io(io, IO::ByteFormat)` / `self.from_io(io, IO::ByteFormat)`. Each type owns its
  codec (`encode` / `decode` for chunk payloads); `Persistent` adds `save` / `load` once.
- `Policy#layout` is the single source of tensor order and shape for JSON keys, binary
  tensors, `Precision` byte counts and `Adam` moments. Keep it in the same order as
  `param_slices`.
- The binary format is versioned (`Container::VERSION`) and chunked. New data gets a new
  `Container::TAG` entry, and readers skip unknown chunks. Change a chunk's payload layout
  only with a version bump.
- Decoders check declared shapes against the payload size before allocating. Every malformed
  input raises `FormatError`.
- The writer records byte order in the header flags, and readers follow the flags; the
  `IO::ByteFormat` passed to `from_io` is not used for decoding. `save` always writes
  little-endian.
- Seeded epochs: epoch *n* shuffles with `Random::PCG32.new(seed, n)`, never with state carried between epochs,
  so resuming from a checkpoint needs no RNG state. LR decay uses the absolute epoch. Keep both properties, or
  resumed runs stop matching uninterrupted ones (a spec checks this).
- Optimizer state is bound to one policy (`Adam#policy`); `Trainer` and `Model` reject a
  mismatch. Moments are stored at `Precision#moment_precision`, never I8.
