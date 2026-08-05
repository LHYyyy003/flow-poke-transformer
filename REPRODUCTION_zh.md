# Flow Poke Transformer 复现说明

本目录基于官方 `CompVis/flow-poke-transformer` 仓库，固定到提交
`495998c800d22589b48e8809b8821c43bac93f00`。默认复现目标是官方 FPT
交互式 Demo；训练完整模型还需要按 `README.md` 准备 WebVid 等训练数据。

## 已配置环境

- Python 3.11（隔离环境：`.venv`；模型源码使用了 Python 3.11 的星号下标语法）
- PyTorch 2.8 + torchvision 0.23
- Demo 所需的 Gradio、einops、jaxtyping、matplotlib 等依赖
- NVIDIA GPU：默认使用 CUDA

激活环境：

```bash
cd /home/lhy4191/lhy/flow-poke-transformer-repro
source .venv/bin/activate
```

启动 Demo：

```bash
./run_demo.sh
```

浏览器访问 `http://localhost:55555`。首次启动会自动下载官方 FPT 权重和
DINOv2 权重到 PyTorch 缓存目录，因此需要联网并等待下载完成。

如需启用编译优化，可直接运行：

```bash
.venv/bin/python -m scripts.demo.app --compile True --warmup_compiled_paths True
```

首次验证建议保留 `run_demo.sh` 的无编译设置，以减少启动时间。端口可通过
`FPT_PORT=7860 ./run_demo.sh` 修改。

## 重建环境

```bash
rm -rf .venv
uv venv --python 3.11
uv pip install --python .venv/bin/python -r requirements-demo.txt
```

## 训练入口

官方单卡训练命令如下（需要预处理后的数据分片）：

```bash
python train.py fpt --tar_base /path/to/preprocessed/shards \
  --out_dir output/fpt --compile True
```

数据预处理步骤见 `scripts/data/README.md`，训练参数见 `python train.py --help`。
