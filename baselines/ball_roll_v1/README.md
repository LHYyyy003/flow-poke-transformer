# ball_roll_v1 baseline

固定场景使用仓库自带的 `scripts/myriad_eval/qual_examples/ball_roll.jpg`。在网球中心
`(0.276, 0.721)` 施加向右 `(0.120, 0.000)` 的 poke，并查询球面上的 5 个位置。

位置指标分为 poke 点动作重建误差，以及将球近似为刚体平移时 5 个球面点的代理 EPE。
这里没有真实下一帧标注，因此该指标只能用于修改前后的回归对比，不能解释为真实数据集精度。

重新运行：

```bash
cd /home/lhy4191/lhy/flow-poke-transformer-repro
.venv/bin/python baselines/ball_roll_v1/run_baseline.py
```

生成物保存在 `artifacts/`：输入图、位置叠加图、64×64 稠密光流可视化、JSON 指标和
包含 GMM 分量及原始预测的 NPZ 文件。脚本固定随机种子、场景坐标和推理精度设置。

## 冻结结果（RTX 3090）

环境为 PyTorch 2.8.0+cu128、bfloat16、未启用 `torch.compile`：

| 指标 | 结果 |
| --- | ---: |
| poke 点终点误差（448×448） | 0.134 px |
| 5 个球面点代理平均 EPE | 1.087 px |
| 5 个球面点最大 EPE | 2.491 px |
| 图像编码平均耗时 | 10.314 ms |
| 5 点并行推理中位耗时 | 79.850 ms |
| 32×32 稠密推理中位耗时 | 79.917 ms |
| 64×64 稠密推理中位耗时 | 81.870 ms |
| 峰值已分配显存 | 1.055 GiB |
| 缓存命中后的模型加载时间 | 8.906 s |

32×32 测试出现一次 349.687 ms 的长尾，因此其平均值为 93.783 ms；回归比较时建议同时
观察中位数和 P95，不要只看均值。完整精度、均值、P95、权重哈希及逐点预测见
`artifacts/metrics.json`。
