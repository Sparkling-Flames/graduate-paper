# 本地数据

[UCI HAR 官方下载页](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) 提供原始归档。外层 ZIP 内含 `UCI HAR Dataset.zip`。本机已将训练集和元数据展开到 `data/UCI HAR Dataset/`，供 [first_window.py](../examples/first_window.py) 读取。重新准备时，从内层 ZIP 提取 `train/` 及数据集顶层的说明文件到同一位置。

11 月 26 日 UCI CNN 实验使用快照中的训练缓存。MotionSense 使用独立的数据准备流程，见[原始复现说明](../studies/har_cnn_20261126/README.md)。
