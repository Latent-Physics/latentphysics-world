<div align="center">

<img src="docs/media/hero.png" width="820" alt="Latent Physics World — real runs on one consumer GPU"/>

# Latent Physics World

### A high-precision, high-speed physics simulator for physical AI.
### 面向物理智能的高精度 · 高速物理模拟器。

Thousands of contact-accurate worlds on a single GPU — where robots learn
to touch, grasp, and move through the human world before they ever step into ours.

*单张 GPU 上成千上万个接触精确的并行物理世界——让机器人在踏入现实之前,先学会触碰、抓取与穿行。*

</div>

---

## The bottleneck to physical intelligence isn't the brain — it's the world.

Foundation models can already reason, plan, and speak. What they cannot do is
*act* — because acting in the physical world takes billions of contact-rich
interactions to learn, and the real world does not scale. It cannot be reset,
it cannot be parallelized, and every failure has a cost. A robot cannot learn
to load a dishwasher by breaking ten thousand of them.

Simulation is the only path to physical intelligence at scale — **but only if
it is contact-accurate, massively parallel, and transferable to reality.** And
the environment that matters most is also the hardest: the cluttered, contact-
dense **human indoor world**, where robots must both manipulate and navigate.

> **物理智能的瓶颈不是大脑,是世界。** 大模型已经会推理、规划、对话,唯独不会*行动*——
> 因为在物理世界中行动,需要以亿计的接触交互去学习,而真实世界无法扩展、不能重置、
> 每次失败都有代价。通往规模化物理智能的唯一路径是仿真——**前提是它接触精确、能大规模
> 并行、且能迁移回真机。** 而最有价值也最难的环境,正是杂乱、接触密集的**人类室内世界**:
> 机器人要在其中同时完成操作与导航。

## What we are building

**Latent Physics World (LPW)** is a GPU-native physics simulator built around
two numbers: **how closely contact matches reality, and how many worlds run
per second.** Thousands of contact-accurate worlds in parallel on a single
accelerator; what happens inside transfers to real hardware.

Not a viewer of the world. An engine that *runs* it.

> **Latent Physics World(LPW)** 是一个 GPU 原生的物理模拟器,只围绕两个数字构建:
> **接触与现实的差距有多小,每秒能运行多少个世界。** 单张加速卡上成千上万个接触精确的
> 并行物理世界,仿真中发生的一切可迁移到真实硬件——不是世界的观察者,而是*运行*世界的引擎。

<div align="center">
<img src="docs/media/architecture.svg" width="820" alt="LPW architecture: simulation interface / worlds &amp; perception / physics core — your applications above, your compute below"/>
</div>

LPW sits between what you build and what you compute on: above the box, your
environments, pipelines, and data engines; below it, whatever compute you
have — one consumer GPU today, a fleet tomorrow.

> LPW 位于"你的应用"与"你的算力"之间:上面是你的环境、管线与数据引擎,
> 下面是你手上的算力——今天一张消费级显卡,明天一支 GPU 舰队。

## What makes it different

| | Pillar | 支柱 |
|---|---|---|
| **⚡** | **Contact-accurate physics at scale** — thousands of parallel worlds on one GPU, with production-grade contact and friction. | 数千并行世界的接触精确物理 |
| **🏠** | **Indoor worlds on demand** — a pipeline that turns raw 3D assets into simulatable, collision-ready worlds. | 按需生成的室内世界(资产管线) |
| **👁** | **Multi-modal perception** — LiDAR, depth, and segmentation, GPU-batched across every world. | 多模感知(LiDAR·深度·分割) |
| **🎯** | **Sim-to-real** — domain randomization and calibration built to close the reality gap. | sim-to-real(域随机化 + 标定) |
| **🧠** | **PyTorch-native interface** — zero-copy tensors, fully batched; simulation state flows straight into your stack. | PyTorch 原生接口(零拷贝·批量) |

## Gallery — real runs, real numbers

LPW is early and moving fast — and it already runs. Every clip below is an
actual simulation from this repo on a single consumer GPU (RTX 5070 Ti),
backed by committed tests. Nothing staged, nothing rendered offline.

> LPW 尚处早期、推进很快——但**已经能跑**。以下每段动图都是本仓库在单张消费级
> GPU 上的真实运行,数字均有已提交的测试背书,无摆拍。

| | | |
|---|---|---|
| **Trained policy — 100% success** PPO on 2048 parallel worlds; deterministic eval 100%. *PPO 策略,确定性评估 100% 成功* ([train](examples/train_franka_reach.py)) | **Procedural indoor worlds** seeded rooms: walls, furniture, clutter, cameras. *程序化室内场景(可复现)* ([code](latentphysics/assets/scene_gen.py)) | **Asset pipeline** concave mesh → CoACD convex parts → simulation. *凹网格→凸分解→仿真* ([code](latentphysics/assets/__init__.py)) |
| <img src="docs/media/policy_reach.webp" width="240"/> | <img src="docs/media/procedural_room.webp" width="240"/> | <img src="docs/media/convex_decomposition.webp" width="240"/> |
| **GPU depth + segmentation** native batch renderer, meters-true depth. *批量深度+分割(米制)* ([code](latentphysics/perception/camera.py)) | **Batched LiDAR** 5,760 beams × N worlds in one launch → point clouds. *批量激光雷达点云* ([code](latentphysics/perception/lidar.py)) | **~5M physics steps/s** 8192 contact-accurate worlds on one GPU; contact forces match the reference engine to 0.00%. *单卡 8192 世界,约每秒五百万物理步* ([test](tests/test_envs_gpu.py)) |
| <img src="docs/media/depth_segmentation.webp" width="240"/> | <img src="docs/media/lidar_pointcloud.webp" width="240"/> | <img src="docs/media/hero.png" width="240"/> |

And the whole thing speaks PyTorch:

```python
import latentphysics as lpw

scene = lpw.load_scene("scenes/kitchen.xml", lpw.Config(n_worlds=4096))
for _ in range(1000):
    scene.step()          # thousands of worlds, one GPU, contact-accurate
obs = scene.qpos()        # zero-copy PyTorch tensor, ready to train on
```

*Getting started and platform requirements (Linux / NVIDIA CUDA) live in [`docs/`](docs/).*

## Roadmap — accuracy first, then speed, then reality

A simulator earns trust on two axes: how close it is to reality, and how fast
it runs. Every stage below pushes one of the two.

> **路线图——先精度,再速度,最终对齐现实。** 模拟器的价值只在两条轴上:
> 离现实有多近,跑得有多快。每一阶段都在推进其中之一。

| Stage | | Milestone · 里程碑 |
|---|---|---|
| **R0 · Engine core** | ✅ | Contact-accurate GPU physics + asset pipeline, verified on real hardware. 接触精确的 GPU 物理与资产管线,真实硬件验证。 |
| **R1 · Precision manipulation** | ✅ | Vectorized simulation interface, reference-engine fidelity gate (0.00% contact-force gap), physics sentinels; usability proven end-to-end. 向量化仿真接口、对参考引擎的保真门(接触力 0.00% 差)、物理哨兵,端到端可用性验证。 |
| **R2 · Worlds & perception** | ✅ | Procedural indoor scenes; batched LiDAR, depth, segmentation; a 12-task auto-verified benchmark suite. 程序化室内场景、批量激光雷达/深度/分割、12 任务自动验证基准。 |
| **R3 · Speed at scale** | | BVH broadphase, sleep-aware CUDA graphs, multi-GPU — cluttered 100+ geom scenes at millions of steps per second. BVH 宽相、sleep 感知的 CUDA 图、多卡:杂乱大场景也要每秒百万步。 |
| **R4 · Richer worlds** | | Indoor scenes from 3D datasets (USD/GLB), articulated furniture, higher-fidelity sensing; soft bodies on the horizon. 3D 数据集室内场景导入、可动家具、更高保真传感;软体物理在望。 |
| **R5 · Calibrated to reality** | | Real-robot calibration and learned residual dynamics (latent-space physics) — a simulator that converges to reality over time. 真机标定 + 残差学习动力学(潜空间物理):随时间向现实收敛的模拟器。 |

## Acknowledgements — we stand on open foundations

LPW's physics core is built on the shoulders of open research and open source.
We gratefully build on and depend upon [MuJoCo](https://github.com/google-deepmind/mujoco)
and [mujoco_warp](https://github.com/google-deepmind/mujoco_warp) (Apache-2.0),
[NVIDIA Warp](https://github.com/NVIDIA/warp), and [PyTorch](https://pytorch.org).
Full attribution is in [`NOTICE`](NOTICE) and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

> LPW 的物理内核站在开放研究与开源社区的肩膀上,谨致谢并依赖上述项目;完整署名见 `NOTICE`。

---

<div align="center">

**Building the world where physical intelligence is born.**

</div>
