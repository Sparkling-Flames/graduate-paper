# 新实验

[2026-11-27 多方向先导实验](exploration_20261127/README.md)涵盖训练人员覆盖、跨数据集混合、概率校准、教师软目标、时间感受野和传感器位置。

[单腕部署的窗口筛选与蒸馏实验](wrist_eligibility/README.md)对照三处信号均完整与目标部位可用两种评估范围。

[新手机环境下的动作采集预算](activity_budget/README.md)在固定示范人数和标注窗数下双向比较不同的动作组合。

[陀螺仪触发](gyro_trigger/README.md)和[重力方向投影](gravity_transfer/README.md)分别检验选择性读取传感器和跨库信号表示。所有方向的选题判断见[研究总览](../docs/HAR_研究方向总览.md)。

每个研究问题建一个子目录。把可运行的脚本、相对基线改动的变量、人员划分和运行命令放在一起。结果表和图按实验归档。

既有实现可从 [UCI HAR 基线](../studies/har_cnn_20261126/research/round5/cnn_experiments/) 和 [MotionSense 基线](../studies/har_cnn_20261126/research/round5/motionsense_experiments/) 开始阅读。
