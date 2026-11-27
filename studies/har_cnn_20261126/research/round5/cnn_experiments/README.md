# CNN固定教师容量与预算实验

完整协议见`protocol.md`，旧先导核查见`existing_pilot_audit.md`。新输出与round3完全隔离；源数据仅使用已审查的官方train缓存。官方test不打开。

从项目根目录运行：

```bash
research/round3/experiments/venv/bin/python research/round5/cnn_experiments/run_cnn.py prepare
research/round3/experiments/venv/bin/python -u research/round5/cnn_experiments/run_cnn.py train
research/round3/experiments/venv/bin/python research/round5/cnn_experiments/run_cnn.py summarize
research/round3/experiments/venv/bin/python research/round5/cnn_experiments/audit_outputs.py
```

`prepare`只复制/验证缓存与训练顺序、检查损失梯度并复算旧冻结教师logits，不训练模型。`train`执行唯一预定矩阵：27个主学生、3个LS教师诊断学生、1个新LS教师；旧CE教师直接复制。没有超参数CLI网格入口。全部CPU，2个计算线程，不测设备延迟。

`audit_outputs.py`仅在完整矩阵结束后执行，使用sklearn与float64 log-sum-exp独立重算124组checkpoint预测、逐人指标与混淆矩阵，并核对权重哈希、选择规则、配对关系与旧80轮复现；不训练、不选择新模型。

学生保存`best80_state.pt`、`last80_state.pt`、`best160_state.pt`、`last160_state.pt`及同名预算的验证预测NPZ。`best_state.pt`/`last_state.pt`为完整预算的兼容副本。初值、批次顺序、冻结教师身份与脚本SHA256保存在run配置和结果中。

汇总文件：

- `summary_checkpoints.csv`：每run/预算的best、last和末10轮平均指标。
- `aggregate_metrics.csv`：每容量/方法/预算的三seed均值与样本SD。
- `paired_differences.csv`与`paired_aggregate.csv`：同初始化、同batch顺序的配对差值。
- `per_subject_metrics.csv`与`class_confusions.csv`：各真实checkpoint的人员与类别诊断。
- `runs/student_w8_{ce,ls,kd_ce}_seed20261121/round3_prefix_reproduction.json`：原80轮逐轮指标的复现核验。
- `data/teacher_{ce,ls}_logits_manifest.json`：冻结教师目标、行号哈希、预算/epoch/状态哈希与概率统计。
- `progress.json`与`run.log`：当前训练进度；`completion.json`只在完整矩阵结束后产生。

注意：best160包含前80轮候选，选择机会更多。必须同时查看last和末10轮，不把验证峰值差当作独立测试改善。所有结果属于同一四人验证划分；种子标准差不是跨人员总体不确定性。中断run不会自动覆盖，应先明确归档；完整run仅在配置和脚本身份一致时复用。
