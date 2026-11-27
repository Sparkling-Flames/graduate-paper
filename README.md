# 轻量一维卷积网络的人体活动识别与知识蒸馏

本仓库保存毕业论文的研究代码、实验记录和写作材料。11 月 26 日的实验已经按原目录展开，训练入口和已完成的结果均可直接打开。

## 从哪里开始

| 用途 | 入口 |
| --- | --- |
| 查看排期 | [毕业论文十二周进度计划](docs/毕业论文十二周进度计划.pdf) |
| 比较 UCI HAR 的补充训练数据 | [MotionSense 与 PAMAP2 英文会议简报](docs/MotionSense_and_PAMAP2_comparison.pdf) · [本地数据与下载来源](data/README.md) |
| 了解已有结果 | [11 月 26 日研究进展](studies/har_cnn_20261126/output/har_cnn_20261126/HAR_1D_CNN_研究进展_20261126.pdf) · [汇总数据](studies/har_cnn_20261126/output/har_cnn_20261126/analysis_summary.json) |
| 阅读 UCI HAR 训练代码 | [run_cnn.py](studies/har_cnn_20261126/research/round5/cnn_experiments/run_cnn.py) · [实验协议](studies/har_cnn_20261126/research/round5/cnn_experiments/protocol.md) |
| 阅读 MotionSense 训练代码 | [run_motionsense.py](studies/har_cnn_20261126/research/round5/motionsense_experiments/run_motionsense.py) |
| 查看依赖、数据准备与复现命令 | [原始复现说明](studies/har_cnn_20261126/README.md) |
| 开始新实验 | [experiments/](experiments/) |

## 目录约定

- `studies/har_cnn_20261126/` 是既有研究的完整快照。代码、协议、权重、预测和图表保留原有相对路径。已有指标来自开发验证集。
- `experiments/` 存放后续实验。每项实验的代码直接入库，输出写入该实验自己的目录，以便和历史结果比较。
- `docs/` 存放进度计划、前期报告与后续论文稿件。`examples/` 存放已有的独立练习脚本。
- `data/` 存放本机公开数据及其原始下载包。云端数据快照使用 GitHub Releases 附件单独存放；原始 ZIP 由 Git 忽略。下载入口、许可与放置路径见[数据说明](data/README.md)。
- `archives/` 存放原始交付 ZIP，作为本机备份。Git 中已包含展开的研究快照。

## 查看命令行入口

从复现目录运行，在已安装原始复现说明所列依赖的 Python 环境中执行：

```powershell
Set-Location .\studies\har_cnn_20261126
python research/round5/cnn_experiments/run_cnn.py --help
python research/round5/motionsense_experiments/run_motionsense.py --help
```

如需重训，在独立工作副本中按原始复现说明操作，以保留历史权重和结果。
