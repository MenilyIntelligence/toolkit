# menily/toolkit

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](#installation)
[![Status: Internal Alpha](https://img.shields.io/badge/status-internal%20alpha-orange.svg)](#status-and-roadmap)
[![Schema](https://img.shields.io/badge/schema-menily.task--demo%2F1-green.svg)](https://github.com/MenilyIntelligence/schema)

> The Python reference implementation for [`menily/schema`](https://github.com/MenilyIntelligence/schema).
> Three adapters: POV video, VR hand-tracking, motion capture.
> Converts heterogeneous raw data into unified task-level VLA demonstration data.

## Table of contents

- [What this is](#what-this-is)
- [Why it exists](#why-it-exists)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Adapters in detail](#adapters-in-detail)
- [Interoperability](#interoperability)
- [Status and roadmap](#status-and-roadmap)
- [Related projects](#related-projects)
- [License](#license)
- [Contributing](#contributing)
- [Citation](#citation)
- [中文简介](#中文简介)

---

## What this is

`menily/toolkit` is a Python library that ingests four classes of raw data:

- 📹 First-person video (smartphone / GoPro / Vision Pro)
- 🎮 VR hand-tracking logs (Meta Quest Pro / Apple Vision Pro / PICO 4U)
- 🎭 Motion capture (BVH / FBX / C3D)
- 🤖 Robot teleoperation traces (HDF5 / pickle / RLDS)

…and converts every one of them into the same task-level format defined by [`menily/schema v1`](https://github.com/MenilyIntelligence/schema).

## Why it exists

Training a VLA (vision–language–action) model requires millions of **task-level** demonstrations — not raw video frames, not isolated motion clips, but trajectories annotated with the *task being executed*, the *visual context*, the *action sequence*, and the *body morphology* at a resolution that downstream policies can actually learn from.

Most teams today end up in one of three dead ends:

1. Hand-label video frame by frame (slow, expensive, non-scalable)
2. Train only on simulation (domain gap, unrealistic motion)
3. Depend on a proprietary robot teleoperation lab (geographically concentrated, politically fragile)

`menily/toolkit` is the preprocessing layer that lets all four raw data sources feed into the same VLA training pipeline without per-source glue code.

## Architecture

```
raw input                    adapter              task-level output
──────────                   ────────              ──────────────────
smartphone video   ─┐                      
VR demonstration   ─┼─► segmentation ──► alignment ──► menily/schema v1
motion capture     ─┤    ▲                  ▲                ▲
teleoperation      ─┘    │                  │                │
                    language prompts   action space    VLA training
```

Each adapter produces the same `Task` object conforming to [`menily/schema v1`](https://github.com/MenilyIntelligence/schema) — **one file, one task, fully self-contained**.

## Installation

⚠️ `menily-toolkit` is in **internal alpha**. PyPI release is planned (see [Status](#status-and-roadmap)). Early access:

```bash
# Future (PyPI)
pip install menily-toolkit

# Today (from source, internal alpha)
git clone https://github.com/MenilyIntelligence/toolkit
cd toolkit
pip install -e .
```

Dependencies: Python 3.10+, NumPy, PyTorch, mediapipe (for POV video hand-keypoint detection), ffmpeg (for video I/O).

## Quick start

```python
from menily.toolkit import pov, schema

# POV video → task-level demonstration(s)
tasks = pov.segment(
    video_path="./demo_pour_water.mp4",
    language="Pour water from the blue cup into the kettle.",
    language_variants=[
        "把蓝色杯子里的水倒进水壶里",
        "Fill the kettle with water from the blue cup",
    ],
    fps=30,
    viewpoint="ego",
    body_morphology="bimanual_humanoid",
    collection_region="SEA",
)

# Validate + save each segmented task
for task in tasks:
    report = task.validate()
    assert report.passed
    task.save_as(schema="menily.task-demo/1", out_dir="./out/")
```

Output: one JSON file per task under `./out/`, each conforming to `menily/schema v1`.

## Adapters in detail

### `toolkit.pov` — First-person video

Input: MP4/MOV from smartphone, GoPro, Vision Pro, or any egocentric camera.

Pipeline:
1. Frame sampling (resample to target fps, default 30Hz)
2. Hand keypoint detection (MediaPipe or HaMeR)
3. Trajectory reconstruction (keypoints → end-effector 6-DoF)
4. Task segmentation (optical flow + action-energy + language-timestamp anchors)
5. Per-segment task object emission

```python
from menily.toolkit import pov

tasks = pov.segment(
    video_path="./raw/demo.mp4",
    language="...",
    fps=30,
    viewpoint="ego",
    body_morphology="bimanual_humanoid",
    collection_region="SEA",
)
```

### `toolkit.vr` — VR hand-tracking

Input: JSON or binary logs from Meta Quest Pro, Apple Vision Pro, or PICO 4U VR devices.

```python
from menily.toolkit import vr

tasks = vr.from_quest_log(
    log_path="./raw/quest_session.json",
    language="Assemble the blue widget onto the base plate.",
    fps=60,
    viewpoint="ego",
    body_morphology="bimanual",
    calibration={
        "origin": "room_center",
        "scale_to_robot": 0.9,
    },
)
```

Strength: native 60-90Hz, sub-centimeter trajectory precision.
Weakness: visual context is virtual — downstream teams usually pair with a separate RGB render.

### `toolkit.mocap` — Motion capture

Input: BVH / FBX / C3D from OptiTrack, Vicon, Xsens.

```python
from menily.toolkit import mocap

tasks = mocap.from_bvh(
    bvh_path="./raw/optitrack.bvh",
    segmentation_file="./raw/task_segments.json",
    body_morphology="humanoid",
    retarget_to="unitree_g1",
    retarget_backend="adamorph",   # or "omniretarget" / "spark" / "kdmr" / "custom"
    physics_filter=True,
)
```

Retargeting backends ([AdaMorph](https://arxiv.org/abs/2601.07284), [OmniRetarget](https://omniretarget.github.io), [SPARK](https://arxiv.org/abs/2603.11480), [KDMR](https://arxiv.org/abs/2603.09956)) are pluggable — the toolkit doesn't reimplement retargeting, it composes existing research.

## Interoperability

| Direction | Format | Method |
|---|---|---|
| Export downstream | RLDS (Open X-Embodiment) | `Task.to_rlds()` |
| Export downstream | HuggingFace `datasets.Dataset` | `Task.to_hf_dataset()` |
| Import upstream | Existing RLDS / Open X-Embodiment | `from_rlds(path)` |
| Import upstream | BONES-SEED (motion data) | `mocap.from_bones_seed(path)` *(planned)* |

## Status and roadmap

| Component | Status | PyPI release |
|---|---|---|
| `toolkit.core` — `Task` object, validation, I/O | Stable | 2–3 weeks |
| `toolkit.pov` — first-person video adapter | Internal alpha | 4–6 weeks |
| `toolkit.vr` — VR hand-tracking adapter | Internal alpha | 4–6 weeks |
| `toolkit.mocap` — motion capture adapter | Design finalized | 8–10 weeks |
| Reference dataset card on HuggingFace | Pending | After PyPI |

We build in open but stage releases — the schema is stable first, then core, then each adapter. If you are building a VLA / VLM / world-model pipeline and want early access or a specific adapter prioritized, email: <Masashi@Menily.AI>.

## Related projects

| Repo | Description |
|---|---|
| [menily/schema](https://github.com/MenilyIntelligence/schema) | The specification this toolkit implements |
| [menily/research](https://github.com/MenilyIntelligence/research) | Research notes + design rationale |
| [menily.ai](https://www.menily.ai) | Organization site — team, publications, contact |

### Recommended reading

- 📄 [Task-Level Demonstration Data for VLA Models: A Survey (PDF)](https://www.menily.ai/research/01-task-level-vla-data-survey.pdf) — 12-page preprint, April 2026
- 📝 [异构机器人数据源统一接口的设计思路：menily/toolkit 的架构笔记](https://juejin.cn/) — Masashi, 掘金, April 2026

## License

Apache License 2.0 — see [`LICENSE`](./LICENSE) (added with first tagged release).

## Contributing

- 🐛 **Bug reports** → [open an Issue](https://github.com/MenilyIntelligence/toolkit/issues/new)
- 💡 **API design discussion** → PRs welcome; discuss in an Issue first for significant changes
- 📧 **Early-access requests / specific-adapter prioritization** → <Masashi@Menily.AI>
- 🌐 **Organization** → [github.com/MenilyIntelligence](https://github.com/MenilyIntelligence)

## Citation

```bibtex
@misc{menily2026toolkit,
  author       = {Masashi},
  title        = {menily/toolkit: Python Reference Implementation for
                  Task-Level VLA Demonstration Data},
  year         = {2026},
  howpublished = {Menily Intelligence, Apache-2.0 open source},
  url          = {https://github.com/MenilyIntelligence/toolkit}
}
```

---

## 中文简介

**menily/toolkit** 是 [`menily/schema`](https://github.com/MenilyIntelligence/schema) 的 Python 参考实现。

三个 Adapter：

- `toolkit.pov` — 第一人称视频（手机、GoPro、Vision Pro）→ 任务级示教数据
- `toolkit.vr` — VR 手部追踪（Quest / Vision Pro / PICO）→ 末端执行器轨迹
- `toolkit.mocap` — 动作捕捉（BVH / FBX / C3D）→ 全身动作序列（含 retargeting）

统一输出符合 `menily/schema v1` 的任务单元，可直接喂给 VLA 训练管道，或互转到 Open X-Embodiment / RLDS 和 HuggingFace Datasets。

当前处于 Pre-MVP 阶段，PyPI 发布分批推进。需要定向早期接入：<Masashi@Menily.AI>。
