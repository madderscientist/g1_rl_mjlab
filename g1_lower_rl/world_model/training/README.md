# Position-WM training

For the official full-body **MLP + SuperMimic** policy task, native playback,
ONNX and complete policy/online-WM/critic/Adam continuation, use the
[mocap task guide](../../tasks/motion_tracking/README.md).
The WM-only commands below are not policy continuation and must not replace
the online WM in a full policy checkpoint.

The [v3 contract](../README.md) uses 145 neural inputs, 76 raw outputs and one
held-position-target transition every 20ms. The release contains FP32 weights,
all frozen normalization buffers and physical provenance, **not an optimizer**.

## Continue from the release

```bash
python scripts/train_wm.py \
  --initialize path/to/g1_wm.pt \
  --checkpoint path/to/audited_native22650/checkpoint.pt \
  --replica-source path/to/pinned/SuperMimic-Reproduced \
  --motion-dir motions/mocap_supermimic_v1 \
  --fixed-validation \
  --output-dir logs/world_model/finetune \
  --device cuda:0 --hours 1
```

The audited native22650 teacher/lineage reports, full mmap mocap data and
pinned simulator source must be supplied separately.
`--initialize` retains model/scales and starts fresh Adam and counters.
Inputs remain in physical units: do not normalize twice or refit the scales.
For custom training, use `load_world_model(...)` and create an optimizer on
`model.parameters()`; loading preserves parameter gradients.

The release checkpoint is update31750. Its collector uses frozen native22650
actions, MuJoCo labels and native position actuators without explicit GC.
To use that strict native collector, set `--teacher-kind supermimic22650`,
provide its audited frozen checkpoint/lineage reports and add
`--replica-source path/to/pinned/SuperMimic-Reproduced`. It requires the full
manifest-backed mmap corpus and does not substitute other teachers.
Shared native loading/reset helpers live in [`native.py`](../native.py).

## Data, normalization and losses

- A `TransitionBatch` stores initial state71, packed inputs145, next state71,
  final-substep world foot forces `[2,3]`, validity, segment and contract3.
  Complete state cannot be recovered from the gravity-based neural inputs.
- Replay windows never cross resets, clip swaps, invalid data or discontinuities.
  Nonfinite native physics and overflow fail explicitly.
- Families/identical sources cannot cross train/validation partitions.
  Sampling is uniform over clips then frames, with disjoint environment columns;
  mmap stays on CPU with a bounded GPU frame cache. Motion data uses the 29
  canonical robot joints; references use actual fixed29 FK.
- Fit normalization only on training columns, then freeze it: input mean/scale,
  update `delta_scale`, physical-error `state_scale`, and next-root force scale.
  `state_scale` and `delta_scale` are not interchangeable.
- Autoregression integrates predicted states with recorded actual targets/context;
  it does not ask the teacher for new actions on imagined states. Losses include
  normalized state/force errors; evaluation also reports physical-unit errors,
  contacts and a constant-velocity baseline.

`MotionCollector` is the common native recording base, not another teacher.
It records the actual clamped position target held for four 5ms substeps,
checks shared foot friction and rejects unconditioned external forces.
Data must match the recorded geometry, actuator parameters and physical contract.
Custom sources can supply batches via `run_training(..., collector_factory=...)`.

## Checkpoints

| Mode | Model/scales | Adam/counters |
|---|---|---|
| `--initialize release.pt` | Retained | New |
| `--resume full_snapshot.pt` | Retained | Continued |
| `--fork-from full_snapshot.pt` | Retained | Audited capacity branch |

Modes are mutually exclusive. Release artifacts cannot be resumed or forked.
Full snapshots include Adam, RNG, sources, corpus, counters and validation hashes.
Resume strictly checks sources, collector, world size, architecture and validation,
and rebuilds simulator trajectories and replay.

Capacity forks require a new directory, verified parent snapshot, unchanged
physics/model/loss semantics and exact parent `fixed_validation_rank_<rank>.pt`
files. They preserve validation/evaluation sizes and the cumulative time budget.

Single-process and `torchrun` DDP use the same trainer. `STOP` requests a graceful
checkpointed stop. Compilation/BF16/TF32 are explicit options; source snapshots
are immutable and never hot-patched.

## Export and tests

```bash
python scripts/export_wm.py \
  --source path/to/full_training_snapshot.pt \
  --destination export/g1_wm.pt

python -m pytest -q \
  tests/test_world_model.py tests/test_world_model_checkpoint.py \
  tests/test_wm_collection.py tests/test_world_model_adapters.py \
  tests/test_wm_training.py
```

Export preserves weights/scales and physical/data metadata with an optional JSON
summary, rejecting incompatible/nonfinite state and accidental overwrites.
Synthetic tests cover fitting, strict continuation and CPU DDP; synthetic data
is never a source of physical pretrained weights. Native derivative probes
measure sensitivity to held position targets. Open-loop metrics do not certify
native gradient accuracy, contact equivalence or closed-loop policy gains.
