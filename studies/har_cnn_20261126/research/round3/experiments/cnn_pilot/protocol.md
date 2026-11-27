# CNN学习先导

完整事先决定见../decision_log.md的D3。

LS教师诊断完成后、CNN扩展运行之前

- 诊断结果：LS teacher F1=0.96190271（原CE teacher=0.95259277）；w32三seed新KD F1为0.96840374/0.97208402/0.97017304，均值0.97022027；相对原KD平均+0.01804678，相对直接LS平均+0.00348570，方向均3/3正。前一个对照超过预定0.005参考，后一个未超过。该结果支持继续研究teacher训练目标/输出来源，但不改写主默认KD未超LS的结果，不进行统计显著性声称。
- MLP训练停止：主矩阵1 teacher+27 students，诊断1 teacher+3 students，总32模型；已足够形成毕业论文的可执行主线与一个有信息量的诊断。不开新网格、不继续换teacher。
- 条件CNN扩展的目的：学习原始时序输入、Conv1d感受野/池化、参数量与同损失配对比较；观察在另一输入和结构下，CE/LS/KD的相对排序是否同向。MLP使用561手工特征，CNN使用9×128信号，故不把二者差值归因于架构本身。
- 固定配置：仍用同17训练人员/4验证人员，训练文件中9路原始窗口；按训练窗口和时间轴拟合逐通道均值/标准差。teacher通道32/64，student通道8/16；每层kernel=5、padding=2，Conv-ReLU后第一层MaxPool1d(2)，第二层后AdaptiveAvgPool1d(1)，最后Linear到6类，无BN/Dropout。含bias参数teacher=12166，student=1126。
- teacher seed20261120；学生只用seed20261121。Adam=.001、batch128、80epochs、默认CE/LS/KD参数与主实验相同，固定初始化和batch顺序。CNN单独目录，不混入3seed主矩阵。
- 成本与停止规则：先在teacher既定80epoch流程的第5epoch记录实际时间和80epoch线性估计；若预计teacher>240秒则停止该扩展并记录成本，保留已有输出。否则继续teacher；若完整teacher的best train accuracy<.95或val macro-F1<.85，则停止KD扩展，不通过调参追赶门槛。通过时完成1个student容量的CE/LS/KD，共4模型，之后停止CNN训练。预计完整扩展3–8分钟，单seed结果只用于学习/可行性。
- 官方test特征标签仍完全不读。全部训练完成后再单独测CPU延迟，测前通知主任务暂停重度渲染。

