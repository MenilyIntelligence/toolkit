# menily · toolkit

> Open-source data processing toolkit for turning human motion, VR demonstrations,
> and first-person video into task-level demonstration data for VLA models.
>
> *Part of [Menily Intelligence](https://www.menily.ai)'s embodied AI data infrastructure.*

## Why

Training a VLA (vision-language-action) model requires millions of **task-level** demonstrations — not raw video frames, not isolated motion capture — but trajectories annotated with the *task being executed*, the *visual context*, and the *action sequence* at a resolution that downstream policies can actually learn from.

Most teams either:

1. Hand-label video data frame by frame (slow, expensive, non-scalable)
2. Run simulation only (domain gap, unrealistic motion)
3. Rely on proprietary robot teleoperation labs (geographically concentrated, politically fragile)

`menily/toolkit` is the pre-processing layer that converts heterogeneous recordings — **smartphone POV video, VR hand-tracking logs, IMU motion capture, teleoperation traces** — into a unified task-level demonstration format that any VLA model can train on.

## Design

```
raw input                    adapter              task-level output
──────────                   ────────              ──────────────────
smartphone video   ─┐
VR demonstration   ─┼─► segmentation ──► alignment ──► task schema
motion capture     ─┤    ▲                  ▲                ▲
first-person POV   ─┘    │                  │                │
                    language prompts   action space    VLA training
```

We ship three adapters:

- `toolkit.pov`    — first-person video → joint trajectory + visual tokens
- `toolkit.vr`     — Quest / Vision Pro hand-tracking → end-effector trajectory
- `toolkit.mocap`  — optical MoCap BVH / FBX → full-body action sequence

Each adapter outputs the same [**schema**](https://github.com/MenilyIntelligence/schema) — one file, one task.

## Quick start

```bash
pip install menily-toolkit  # not yet on PyPI — see Status below

from menily.toolkit import pov, schema

tasks = pov.segment("./demo.mp4", language="pour water from the cup")
for task in tasks:
    task.save_as(schema.TaskLevelDemoV1, "./out/")
```

## Status

**Pre-MVP.** We are building in open but the public release is staged:

- [x] Schema draft (see [`menily/schema`](https://github.com/MenilyIntelligence/schema))
- [ ] `pov` adapter — internal alpha
- [ ] `vr` adapter — internal alpha
- [ ] `mocap` adapter — design
- [ ] PyPI release
- [ ] Reference dataset card on HuggingFace

If you are building a VLA / VLM / world-model training pipeline and want early access or a specific adapter prioritized: <Masashi@Menily.AI>

## License

Apache-2.0 (planned — pending first public release).

## Related

- [menily.ai](https://www.menily.ai)
- [menily/schema](https://github.com/MenilyIntelligence/schema) — task-level demo data specification
- [menily/research](https://github.com/MenilyIntelligence/research) — notes on data infrastructure for embodied AI

---

中文说明：**menily/toolkit** 是一个开源数据处理工具链，把人类动作视频、VR 演示、动捕文件、第一人称视频转化为 VLA / 世界模型可以直接训练的**任务级示教数据**。当前处于 Pre-MVP 阶段，接受定向早期接入申请：<Masashi@Menily.AI>。
