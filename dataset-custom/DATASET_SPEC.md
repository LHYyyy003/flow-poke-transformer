# Adaptive Scene Billiards 训练数据集生成规范

> 用途：将本文档和 `data.py` 交给第三方，使其能生成与本实验相同语义、相同字段和相同张量形状的训练数据。

## 1. 实现和依赖

数据生成器的唯一实现是：

```text
data.py
```

Python 相关命令必须在仓库的 Conda 环境 `myriad` 内运行：

```bash
conda run --no-capture-output -n myriad \
  python -c "import sys; print(sys.executable)"
```

`data.py` 的直接运行时依赖只有 PyTorch、NumPy、OpenCV 和 `billiards`。PyTorch 请根据对方的 CPU/CUDA 平台安装。

```bash
conda run --no-capture-output -n myriad \
  python -m pip install \
  numpy \
  opencv-python-headless \
  billiards
```

## 2. 数据集是如何存在的

本实验没有预先保存的图片目录或轨迹文件。训练使用的
`AdaptiveSceneEpisodeDataset` 是无限 PyTorch `IterableDataset`：

```python
batch = next(iter(loader))
```

每当 DataLoader 请求一个样本时，对应 worker 才在 CPU 上即时生成一个 scene episode，并直接返回
`dict[str, torch.Tensor]`；不预生成、不缓存、不写入图片或轨迹文件。训练步数决定消费多少数据，
数据集本身没有长度，也不会自然结束。这与 MYRIAD 官方台球数据的 `while True: yield sample` 策略一致。

多 worker 和多卡会按全局 stream index 自动分片，不会让不同 worker/rank 生成相同样本。为了验证和调试，
仍可调用 `train_dataset.sample_at(index)` 复现指定样本；有限验证集使用独立的
`DeterministicAdaptiveSceneEpisodeDataset`。

一个 episode 包含：

```text
一个隐含场景（四边形边界 + 可选圆障碍 + 恢复系数）
  ├─ query trajectory：训练时要预测的轨迹
  ├─ 0～K 条 context trajectories：来自同一场景的已观测经验
  └─ noisy occupancy mask：模型实际获得的不准确场景先验
```

query 和 context 共享场景几何与恢复系数，但使用不同的球初始位置和速度。context 不包含 query 轨迹，因此不会泄漏 query 的未来。

## 3. 正式训练集配置

正式训练使用无限在线流和训练种子 `seed=51`。数据集的 `base_seed` 按训练代码约定设置为
`seed * 1_000 = 51_000`，由训练的 `max_steps`（而不是 `num_samples` 或 epoch）决定停止时机。

### 3.1 物理场景配置

| 参数 | 值 | 含义 |
|---|---:|---|
| `image_size` | 96 | occupancy mask 的高和宽 |
| `num_balls` | 6 | 球数 |
| `radius` | 0.035 | 球半径，使用归一化场景坐标 |
| `dt` | 0.04 | 两帧之间的仿真时间 |
| `trajectory_steps` | 64 | flow 段数；position 共 65 帧 |
| `moving_probability` | 0.55 | 每个球初始时运动的概率 |
| `min_speed` | 0.18 | 初速度下限 |
| `max_speed` | 0.45 | 初速度上限 |
| `corner_jitter` | 0.045 | 四边形顶点的随机扰动幅度 |
| `max_disk_obstacles` | 1 | 最多圆形障碍数 |
| `disk_probability` | 0.35 | 每个候选圆障碍被创建的概率 |
| `min_disk_radius` | 0.055 | 圆障碍半径下限 |
| `max_disk_radius` | 0.10 | 圆障碍半径上限 |
| `min_restitution` | 0.72 | 场景恢复系数下限 |
| `max_restitution` | 1.0 | 场景恢复系数上限 |
| `mask_supersample` | 2 | clean mask 超采样倍数 |

### 3.2 Episode 和 mask 配置

| 参数 | 值 | 含义 |
|---|---:|---|
| `max_context_trajectories` | 3 | 最多 context 轨迹数 |
| `fixed_context_trajectories` | `None` | 训练时不固定，在 0、1、2、3 中均匀选择 |
| `max_context_tokens` | 128 | 输出的固定 context token 数 |
| `mask_max_shift_px` | 3.0 | mask x/y 方向最大平移像素 |
| `mask_morphology_px` | 2 | 形态学操作最大半径 |
| `mask_hole_probability` | 0.35 | 产生缺口的概率 |
| `mask_false_positive_probability` | 0.25 | 产生假障碍物的概率 |
| `mask_noise_std` | 0.04 | 像素高斯噪声标准差 |

### 3.3 按原配置创建数据集

为防止将来默认值改动，交付时建议显式写出所有参数：

```python
from data import (
    AdaptiveEpisodeConfig,
    AdaptiveSceneEpisodeDataset,
    SceneSimulationConfig,
)

simulation = SceneSimulationConfig(
    image_size=96,
    num_balls=6,
    radius=0.035,
    dt=0.04,
    trajectory_steps=64,
    moving_probability=0.55,
    min_speed=0.18,
    max_speed=0.45,
    corner_jitter=0.045,
    max_disk_obstacles=1,
    disk_probability=0.35,
    min_disk_radius=0.055,
    max_disk_radius=0.10,
    min_restitution=0.72,
    max_restitution=1.0,
    mask_supersample=2,
)

episode = AdaptiveEpisodeConfig(
    max_context_trajectories=3,
    fixed_context_trajectories=None,
    max_context_tokens=128,
    mask_max_shift_px=3.0,
    mask_morphology_px=2,
    mask_hole_probability=0.35,
    mask_false_positive_probability=0.25,
    mask_noise_std=0.04,
)

train_dataset = AdaptiveSceneEpisodeDataset(
    simulation=simulation,
    episode=episode,
    base_seed=51_000,
)

sample = train_dataset.sample_at(0)  # 仅用于复现/调试；训练应通过 DataLoader 迭代
```

## 4. 单个样本的格式合同

记：

- `T = trajectory_steps = 64`
- `N = num_balls = 6`
- `C = max_context_tokens = 128`
- `H = W = image_size = 96`
- `D = max_disk_obstacles = 1`

`dataset.sample_at(index)` 或流式迭代得到的单个样本必须返回下表中的全部字段。

| 字段 | dtype | 单样本 shape | 含义 |
|---|---|---:|---|
| `positions` | `torch.float32` | `[T+1, N, 2]` = `[65,6,2]` | query 中的球心 `(x,y)` |
| `flows` | `torch.float32` | `[T, N, 2]` = `[64,6,2]` | `positions[t+1] - positions[t]` |
| `collision_objects` | `torch.bool` | `[T, N]` = `[64,6]` | 该时间段参与任意碰撞的球 |
| `ball_collision_objects` | `torch.bool` | `[T, N]` | 参与球球碰撞的球 |
| `obstacle_collision_objects` | `torch.bool` | `[T, N]` | 与墙或圆障碍碰撞的球 |
| `object_colors` | `torch.float32` | `[N, 3]` = `[6,3]` | 固定球 ID 对应的 RGB，范围 `[0,1]` |
| `exists` | `torch.bool` | `[T+1, N]` = `[65,6]` | 球是否存在；当前实现全为 `True` |
| `clean_mask` | `torch.float32` | `[1,H,W]` = `[1,96,96]` | 真实 occupancy，0 为自由区，1 为障碍，边界可为 soft value |
| `noisy_mask` | `torch.float32` | `[1,H,W]` | 模型获得的持久损坏 occupancy |
| `mask_confidence` | `torch.float32` | `[1,H,W]` | noisy mask 像素置信度，范围 `[0.05,1]` |
| `scene_vertices` | `torch.float32` | `[4,2]` | 真实凸四边形顶点 |
| `scene_disks` | `torch.float32` | `[D,3]` = `[1,3]` | padding 后的圆障碍 `(x,y,radius)` |
| `scene_disk_exists` | `torch.bool` | `[D]` = `[1]` | `scene_disks` 中哪些行有效 |
| `scene_restitution` | `torch.float32` | `[]` | 当前场景的标量恢复系数 |
| `context_count` | `torch.int64` | `[]` | 本 episode 实际生成的 context 轨迹数 |
| `context_positions` | `torch.float32` | `[C,2]` = `[128,2]` | context transition 的当前位置 |
| `context_flows` | `torch.float32` | `[C,2]` | 进入当前位置的 incoming flow |
| `context_next_flows` | `torch.float32` | `[C,2]` | 观测到的下一段 flow |
| `context_confidence` | `torch.float32` | `[C]` | 有效 token 为 1，padding 为 0 |
| `context_valid` | `torch.bool` | `[C]` | token 有效性 mask |

`DataLoader(batch_size=B)` 会在所有张量前增加 batch 维。例如 `positions` 变为 `[B,65,6,2]`，`context_count` 变为 `[B]`。

### 4.1 坐标和时间对齐

- 位置使用 `(x,y)` 顺序，场景坐标归一化到约 `[0,1]`。
- `flow` 是一个 `dt` 内的位移，不是除以 `dt` 后的速度。
- `flows[t] = positions[t+1] - positions[t]`。
- `collision_objects[t]` 对应产生 `flows[t]` 的仿真时间段内的碰撞事件。
- 训练一步预测的当前状态是 `positions[t+1]`，incoming flow 是 `flows[t]`，目标是 `flows[t+1]`。

### 4.2 模型输入与隐藏真值

场景编码器只接收：

```text
noisy_mask
mask_confidence
context_positions
context_flows
context_next_flows
context_confidence
context_valid
```

`clean_mask` 只用于训练辅助监督和评价。`scene_vertices`、`scene_disks`、`scene_disk_exists` 和 `scene_restitution` 用于评价、调试和可视化，不能直接喂给模型。

## 5. 生成算法

### 5.1 场景

1. 以 `[(0.07,0.07), (0.93,0.07), (0.93,0.93), (0.07,0.93)]` 为基础四边形。
2. 每个顶点的 x/y 分别加上 `[-corner_jitter,+corner_jitter]` 均匀扰动，再 clip 到 `[0.015,0.985]`。
3. 只接受每个相邻边叉积都大于 `0.02` 的凸多边形。
4. 对每个候选圆障碍，以 `disk_probability` 决定是否生成；半径均匀采样，中心在 `[0.30,0.70]^2` 内采样，并保证与墙、其他圆和球有足够间距。
5. 整个场景共享一个从 `[min_restitution,max_restitution]` 均匀采样的墙/圆障碍恢复系数。

### 5.2 单条轨迹

1. 按球 ID 顺序采样不重叠初始位置。球心从 `[0.04,0.96]^2` 采样，必须在场景内且不与墙、圆障碍或其他球重叠。
2. 随机选一个球保证它一定运动；其他球分别以 `moving_probability` 决定是否运动。
3. 运动球的方向在 `[0,2π)` 均匀采样，速度大小在 `[min_speed,max_speed]` 均匀采样；静止球速度为 0。
4. 使用 `billiards.Billiard` 进行连续碰撞仿真，依次 evolve 到绝对时间 `(step+1)*dt`。
5. 保存初始位置和每次 evolve 后的位置，再作差得到 flow。

球的质量均为 1。球球碰撞使用 `billiards` 的等质量默认碰撞规则；墙和圆障碍使用本实现中的 scene-specific normal restitution。

### 5.3 Context token

每条 context 轨迹产生 `(T-1)*N` 个候选 token。对时间 `t` 和球 `n`：

```text
context_position  = positions[t+1, n]
context_flow      = flows[t, n]
context_next_flow = flows[t+1, n]
acceleration      = norm(context_next_flow - context_flow)
```

默认设置下，一条 context 轨迹有 `63*6=378` 个候选 token。如果候选数超过 128：

1. 取约一半 acceleration 最大的 token，优先保留碰撞附近的转移。
2. 其余 token 从剩余候选集随机采样。
3. 将两部分合并并随机打乱。

如果候选数不足 128，末尾用 0 padding，并令 `context_valid=False`。如果 `context_count=0`，所有 context 数值均为 0，所有 `context_valid` 均为 `False`。

### 5.4 Occupancy mask

1. 在 `image_size * mask_supersample` 分辨率上，使用像素中心坐标测试每个点是否位于四边形内、圆障碍外。
2. 自由区记为 0，多边形外部和圆障碍记为 1。
3. 对超采样 mask 分块取平均，生成 `clean_mask`。
4. 按下列固定顺序生成 `noisy_mask`：
   - x/y 方向独立均匀平移，OpenCV bilinear interpolation，边界使用 replicate。
   - 当 `mask_morphology_px>0` 时以 0.7 概率做一次膨胀或腐蚀，两者各 0.5 概率。
   - 以 `mask_hole_probability` 概率将一个随机矩形区域乘以 `[0,0.25]` 的均匀随机数。
   - 以 `mask_false_positive_probability` 概率绘制一个强度为 `[0.65,1.0]` 的随机圆形假阳性。
   - 加入标准差为 `mask_noise_std` 的像素独立高斯噪声。
   - 做 `3x3, sigmaX=0.6` 的 Gaussian blur，然后 clip 到 `[0,1]`。
5. 置信度按下式生成：

```text
mask_confidence = clip(2 * abs(noisy_mask - 0.5), 0.05, 1.0)
```

## 6. 随机种子合同

设：

```text
scene_seed = base_seed + sample_index
```

各模块使用独立的 NumPy `default_rng` 种子：

| 用途 | 种子 |
|---|---:|
| 场景几何与恢复系数 | `scene_seed` |
| query 轨迹 | `71_000_000 + scene_seed` |
| context 轨迹数 | `83_000_000 + scene_seed` |
| 第 `j` 条 context 轨迹 | `97_000_000 + scene_seed * 17 + j` |
| context token 采样 | `101_000_000 + scene_seed` |
| noisy mask | `109_000_000 + scene_seed` |

这个分离很重要。例如将 `fixed_context_trajectories` 从 0 改成 3，同一
`base_seed/sample_index` 的场景、query 和 noisy mask 仍保持不变，只改变 context。

流式训练中，worker/rank 分片规则为：

```text
shard_id     = rank * num_workers + worker_id
shard_count  = world_size * num_workers
sample_index = stream_start_index + shard_id + local_step * shard_count
```

因此单卡/多卡、多 worker 下不会重复消费同一个 `sample_index`。`IterableDataset` 不能设置
`shuffle=True`；连续 seed 本身分别初始化独立的 NumPy RNG，已提供足够的程序化随机性。

## 7. DataLoader 用法

```python
from data import make_online_dataloader

loader = make_online_dataloader(
    train_dataset,
    batch_size=64,
    num_workers=8,
    prefetch_factor=3,
)

batch = next(iter(loader))
print(batch["positions"].shape)          # torch.Size([64, 65, 6, 2])
print(batch["context_positions"].shape) # torch.Size([64, 128, 2])
```

训练循环直接按 step 消费无限迭代器：

```python
train_iterator = iter(loader)
for step in range(max_steps):
    batch = next(train_iterator)
    # forward / backward / optimizer.step()
```

不要对该训练 dataset 调用 `len()`，不要传 `shuffle=True`，也不要用 epoch 长度控制训练。
`num_workers` 和 `prefetch_factor` 只影响 CPU 生成吞吐；`batch_size` 决定每步消费的 episode 数。
推荐 `persistent_workers=True`，避免反复创建 worker，并让 worker 的流游标持续前进。

## 8. 训练/验证划分约定

正式训练流：

```text
base_seed   = 51_000
length      = infinite
stop        = max_steps
```

训练代码为验证建立两个场景完全对齐的有限 map-style 数据集：

```text
val_base_seed = 9_000_000 + seed = 9_000_051
val_zero: fixed_context_trajectories = 0
val_full: fixed_context_trajectories = 3
```

```python
from dataclasses import replace
from data import DeterministicAdaptiveSceneEpisodeDataset

val_zero = DeterministicAdaptiveSceneEpisodeDataset(
    simulation=simulation,
    episode=replace(episode, fixed_context_trajectories=0),
    num_samples=val_size,
    base_seed=val_base_seed,
)
val_full = DeterministicAdaptiveSceneEpisodeDataset(
    simulation=simulation,
    episode=replace(episode, fixed_context_trajectories=3),
    num_samples=val_size,
    base_seed=val_base_seed,
)
```

在正式命令 `batch_size=768, val_batches=4` 下：

```text
val_size = batch_size * val_batches = 3_072
```

`val_zero[i]` 和 `val_full[i]` 具有相同的场景、query 和 mask，用来公平评估增加 context 后的改善。

注意：碰撞过采样和 rollout window 采样发生在训练循环的 `sample_query_windows` 中，不是 dataset 生成格式的一部分。dataset 始终返回完整 query 轨迹。

## 9. 如果必须将数据落盘

原训练管线不需要落盘。如果需要向不共享代码环境的人交付少量样本，建议每个 `.pt` 文件保存一个原始 sample dict，并另存 JSON metadata。不要对张量转置、压缩、改 dtype 或改键名。

```python
from pathlib import Path
import torch

output_dir = Path("adaptive_scene_dataset")
output_dir.mkdir(parents=True, exist_ok=True)

for index in range(100):
    torch.save(train_dataset.sample_at(index), output_dir / f"sample_{index:08d}.pt")
```

建议 metadata 至少包含：

```json
{
  "format": "adaptive_scene_episode_v1",
  "generator": "data.py",
  "base_seed": 51000,
  "first_index": 0,
  "num_samples": 100,
  "simulation_config": "使用本文第 3.1 节的完整字段",
  "episode_config": "使用本文第 3.2 节的完整字段"
}
```

大量 episode 落盘体积很大，因此正式训练应交付生成代码、完整配置和种子，并使用无限在线流，
而不是预先导出全部 `.pt` 文件。

## 10. 一致性验证

使用下面的检查确认字段形状、dtype 和核心语义正确：

```python
import torch

sample = train_dataset.sample_at(1)

assert sample["positions"].shape == (65, 6, 2)
assert sample["flows"].shape == (64, 6, 2)
assert sample["noisy_mask"].shape == (1, 96, 96)
assert sample["context_positions"].shape == (128, 2)
assert sample["positions"].dtype == torch.float32
assert sample["collision_objects"].dtype == torch.bool
assert sample["context_count"].dtype == torch.int64

torch.testing.assert_close(
    sample["flows"],
    sample["positions"][1:] - sample["positions"][:-1],
)

assert torch.equal(
    sample["collision_objects"],
    sample["ball_collision_objects"] | sample["obstacle_collision_objects"],
)

assert ((0 <= sample["clean_mask"]) & (sample["clean_mask"] <= 1)).all()
assert ((0 <= sample["noisy_mask"]) & (sample["noisy_mask"] <= 1)).all()
assert sample["context_valid"].sum() <= 128
```

## 11. 交付检查清单

第三方在声称数据一致前，应确认：

- [ ] 使用本文对应的 `data.py` 生成数据。
- [ ] 物理配置、episode 配置、`base_seed` 和 `stream_start_index` 全部一致。
- [ ] 训练使用无限 `AdaptiveSceneEpisodeDataset`，没有 `shuffle=True` 或 epoch 长度假设。
- [ ] 验证/导出使用 `DeterministicAdaptiveSceneEpisodeDataset` 或 `sample_at(index)`。
- [ ] 坐标顺序为 `(x,y)`，flow 是位移而非速度。
- [ ] 返回的 20 个键名、dtype 和 shape 与第 4 节一致。
- [ ] context 与 query 共享场景，但轨迹初始状态和随机流相互独立。
- [ ] padding 使用 0，并用 `context_valid=False` 和 `scene_disk_exists=False` 标识。
- [ ] 真实场景参数没有被当作模型输入。
- [ ] 通过形状、dtype 和语义检查。
