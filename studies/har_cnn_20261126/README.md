# 1D CNN HAR 研究复现包

版本：2026-11-26。研究对象是惯性时间序列的轻量 1D CNN 与知识蒸馏。

## 先阅读什么

1. `output/har_cnn_20261126/HAR_1D_CNN_研究进展_20261126.pdf`：本轮完整结论、协议、文献、候选方向。
2. `research/round5/cnn_experiments/results_review.md`：UCI 全部结果及教师与错误分析。
3. `output/har_cnn_20261126/analysis_summary.json`：两数据集的均值、样本标准差与配对差值。
4. `research/round5/cnn_architecture_review.md` 与 `cnn_transfer_review.md`：文献和代码审阅工作记录，含正式书目及获取边界。
5. `research/round5/ls_kd_literature_addendum.md`：LS 教师、蒸馏温度与已有研究的补遗。附带 `research/round4/publication_review.md` 是上轮历史工作记录，本轮只订正其中一项论文链接；本轮 CNN 结论应以前四项和此补遗为准。

这是一套开发研究记录。UCI 官方测试集和 MotionSense 的预留测试人员没有被评分。41 次新训练包括 UCI 的 30 个学生与 1 个 LS 教师，以及 MotionSense 的 9 个学生与 1 个 CE 教师。UCI 另复用一个旧 CE 教师。不能把开发集数字写成最终测试准确率。

## 依赖

实际训练环境为 Python 3.12.14，PyTorch 2.7.1+cpu，NumPy 2.3.5，pandas 2.2.3，SciPy 1.17.0，scikit-learn 1.8.0。所有实验使用 CPU、2 个计算线程；具体环境信息与脚本 SHA256 保存在各实验目录。

在独立 Python 环境中安装：

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy==2.3.5 pandas==2.2.3 scipy==1.17.0 scikit-learn==1.8.0
```

绘图另需 matplotlib。PDF 生成另需 reportlab 和 Noto Sans SC 的 Regular/Bold TrueType 字体；已生成的 PDF 不要求这些依赖。原环境之外的硬件、底层数学库或版本变化可能影响数值复现，应记录新环境，不能修改旧协议使复现门槛静默失效。

## UCI 检查结果与重新训练

从本包最外层目录运行以下命令。包内包含原 CNN 的官方训练缓存、冻结教师、旧学习曲线与索引核对所需的原特征缓存。新目录重复的 CNN 数据缓存省略，可由 prepare 从原缓存复制恢复；原始 UCI 数据压缩包不随包重复分发。

```bash
python research/round5/cnn_experiments/run_cnn.py prepare
python research/round5/cnn_experiments/run_cnn.py summarize
python research/round5/cnn_experiments/audit_outputs.py
```

`train` 会验证并复用同配置下已完成的结果，不会强制重训。若要重新训练，请先完整复制本包形成新的实验工作副本，在副本中归档现有 `research/round5/cnn_experiments/runs` 及汇总输出，保留协议、原始来源与教师身份，再运行：

```bash
python research/round5/cnn_experiments/run_cnn.py train
```

当前实验脚本拒绝覆盖存在冲突或未完成身份的 run。不要删除或改写旧研究记录来伪装新的运行。完整 fresh rerun 会重训 LS 教师与固定的 30 个学生，CE 教师仍按协议复用。

## MotionSense 检查结果与重新训练

为减少重复，包内不含 73.7 MB 官方源 ZIP 与 37 MB 派生输入缓存。包含源 SHA256、全部切窗记录、归一化统计、teacher logits、初值、训练顺序、曲线、权重、预测和结果。以下命令只从作者公开仓库下载原 ZIP，先按预定人员排除测试 CSV 内容，再构建训练和验证缓存。

```bash
python research/round5/fetch_resources.py motionsense
python research/round5/motionsense_experiments/run_motionsense.py prepare
python research/round5/motionsense_experiments/run_motionsense.py summarize
python research/round5/motionsense_experiments/audit_outputs.py
```

已有 result.json 会直接复用。**重新训练请先复制并归档旧 runs。** MotionSense 脚本会覆盖未完成同名目录，不支持从中断优化器状态续训。它的 80/160 轮比较是在一次连续运行中完成的，不是断点恢复实验。

```bash
python research/round5/motionsense_experiments/run_motionsense.py train
```

## 两套结果重新汇总

```bash
python research/round5/report/analyze_results.py
python research/round5/report/compile_report.py
```

汇总脚本只读完成结果，不访问测试集，不训练新模型。报告编译脚本针对本次冻结实验；少量文字中的既有结果和结论不是通用自动科研解释器。若改变协议或重跑后数字不同，应重新核查文字，不可直接沿用结论。

## 预训练教师核验

HARNet 权重来自作者公开地址，未包含在本包。作者代码及许可在 `research/round5/sources`；遵循其中的 academic non-commercial 条款。以下核验使用合成输入，不代表已经训练下游 HAR 教师：

```bash
python research/round5/fetch_resources.py harnet
python research/round5/check_harnet_weights.py
```

分类头是随机初始化。模型要求三轴加速度、30 Hz、真实 5/10 秒，不能直接使用 UCI 2.56 秒窗口或现有九通道 z-score 缓存。

## 指标与完整性

每个学生都有配对初值与全程批次顺序、best80/last80/best160/last160 权重和预测。UCI 还提供末 10 轮平均；最佳规则均是验证 macro-F1 最大，其后按硬标签 NLL 与较早轮次处理并列。

UCI 的 `audit_complete.json` 独立核对 124 组检查点；MotionSense 的 `independent_audit.json` 独立核对 38 组检查点。三种子 SD 仅刻画初始化与优化变动，同一验证人员被重复使用，不能把窗口或种子当成新的独立人群。附带 `MANIFEST_SHA256.json` 可核对包内文件内容。

论文参考文献的完整正文不随包分发。PDF 与两份文献工作记录提供出版社、作者或官方数据链接；未取得全文之处已明示。工作记录可能保留内部检索定位符，它们不是额外文献，应以实际 URL 和 DOI 为准。
