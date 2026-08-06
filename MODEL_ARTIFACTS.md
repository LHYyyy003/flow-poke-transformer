# Model artifacts

Large checkpoints are intentionally excluded from Git and stored on the AutoDL data disk.

Artifact root:

```text
/root/autodl-tmp/flow-poke-transformer-repro/models
```

The safest multi-scene default is the original billiards model, or equivalently either
physics checkpoint with `model.transformer.use_physics_bias = False`. Disabling the
bias restored the original model exactly across all five deterministic validation scenes.

The best experimental physics-enabled checkpoint currently remains
`current/physics_bias_collision_smooth_adapt_500/checkpoint_0000500_strength025.pt`.
It uses collision-smooth kinematics with distance `0.04`, temperature `0.008`,
and has its layer/head scales pre-multiplied by physics strength `0.25`.

`current/physics_bias_collision_smooth_1500/checkpoint_0001500.pt` is a retained
continuation candidate, not the recommended checkpoint: on the same deterministic
head-on scene its mean EPE remained better than the original model, but it regressed
from `0.02720 px` to `0.04181 px` relative to the 500-step checkpoint and its P95 EPE
rose above the original model (`0.36216 px` versus `0.25087 px`). See the artifact
root's `SHA256SUMS` and `ARTIFACTS.txt` for exact paths and checksums. It should not
replace the no-bias baseline until collision-window and worst-scene metrics improve.
