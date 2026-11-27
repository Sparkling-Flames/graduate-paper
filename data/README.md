# 本地数据

## 下载和存放

克隆仓库后，按实验需要下载 ZIP，保存至下表路径。Git 只跟踪本目录的说明；抽样清单、指标和图表随对应实验代码提交。

| 数据集 | 本地路径 | 官方来源及署名 | 许可 |
| --- | --- | --- | --- |
| UCI HAR | `data/UCI-HAR.zip` | [Reyes-Ortiz 等，UCI](https://doi.org/10.24432/C54S4K) | UCI 数据页标注 CC BY 4.0 |
| MotionSense | `data/MotionSense_A_DeviceMotion.zip` | [Malekzadeh 等，设备运动数据 ZIP](https://raw.githubusercontent.com/mmalekzadeh/motion-sense/master/data/A_DeviceMotion_data.zip) | [作者仓库的 MIT 许可](https://github.com/mmalekzadeh/motion-sense/blob/master/LICENSE) |
| PAMAP2 | `data/PAMAP2_Dataset.zip` | [Reiss 与 Stricker，原始 ZIP](https://archive.ics.uci.edu/ml/machine-learning-databases/00231/PAMAP2_Dataset.zip) | [UCI 数据页，CC BY 4.0](https://doi.org/10.24432/C5NW2H) |
| OPPORTUNITY | `data/OPPORTUNITY.zip` | [Roggen 等，UCI](https://archive.ics.uci.edu/dataset/226/opportunity+activity+recognition) | UCI 数据页标注 CC BY 4.0 |

### GitHub 数据快照约定

首次发布时使用标签 `datasets-v1`，每个原始 ZIP 作为一个独立附件，沿用上表文件名。发布说明列明原作者、官方来源、许可及是否修改，并保留随附的许可文本。数据包如有更新，使用新标签，以便旧实验继续引用原版本。当前下载可使用上表官方来源；数据快照集中在[仓库 Releases](https://github.com/Sparkling-Flames/graduate-paper/releases)。

## UCI HAR 解压说明

[UCI HAR 官方下载页](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) 提供原始归档。外层 ZIP 内含 `UCI HAR Dataset.zip`。本机已将训练集和元数据展开到 `data/UCI HAR Dataset/`，供 [first_window.py](../examples/first_window.py) 读取。重新准备时，从内层 ZIP 提取 `train/` 及数据集顶层的说明文件到同一位置。

11 月 26 日 UCI CNN 实验使用快照中的训练缓存。MotionSense 使用独立的数据准备流程，见[原始复现说明](../studies/har_cnn_20261126/README.md)。

## UCI HAR 的两套候选补充数据

| 本地文件 | 数据内容 | 官方来源 |
| --- | --- | --- |
| `MotionSense_A_DeviceMotion.zip` | 24 人，15 个试次，按试次和人员组织的 360 个活动 CSV。原实验输入文件名为 `MotionSense_A_DeviceMotion.zip`。 | [作者仓库](https://github.com/mmalekzadeh/motion-sense) · [作者下载包](https://raw.githubusercontent.com/mmalekzadeh/motion-sense/master/data/A_DeviceMotion_data.zip) |
| `PAMAP2_Dataset.zip` | 原始可穿戴传感器记录。标准活动位于 `PAMAP2_Dataset/Protocol/`，可选活动另存于 `Optional/`。 | [UCI 数据说明](https://doi.org/10.24432/C5NW2H) · [官方下载包](https://archive.ics.uci.edu/ml/machine-learning-databases/00231/PAMAP2_Dataset.zip) |

[会议简报](../docs/MotionSense_and_PAMAP2_comparison.pdf)中的五类窗口数沿用既有开发划分。UCI 统计官方训练文件中除验证人员 3、28、29、30 及活动 6（躺卧）外的行；MotionSense 在[试次清单](../studies/har_cnn_20261126/research/round5/motionsense_experiments/data/trial_manifest.csv)中过滤 `split=train` 并去除 `jogging` 后汇总 `windows`。这两项均按 128 个采样点切窗，相邻窗口重叠 50%。

PAMAP2 的 9 份标准协议记录中，101–108 号均有坐、站、行走、上楼、下楼五类标签（活动编号 2、3、4、12、13）；109 号仅有过渡和跳绳记录。因此按这五类活动扩充训练时，标准协议可用人员为 8 人。
