# Round 4 内部研究笔记：从毕业论文先导结果到可投稿的问题

日期：2026-11-25。用途：供根代理综合，不是用户最终文档。本轮未训练模型、未改动 Round 3 结果、未读取官方最终测试特征或标签。

## 1. 结论与优先级

**优先 A：固定标注人员与窗口预算下，研究教师目标中的哪些信息能够迁移到未见人员，尤其是非目标类别概率结构是否提供超出标量平滑的价值。** 这是最贴近现有结果、最容易从现有代码继续、也最值得形成一篇完整实证论文的问题。贡献不能写成“LS 教师有用”“调温有用”“KD 不等于 LS”“首次在 HAR 使用 DKD”或“首次跨人员 KD”。这些都有直接先例。

**次选 B：多位置传感器教师到单位置学生时，区分教师正确性与学生输入的可观测性。** 方向应用价值更清楚，但已有 Virtual Fusion、跨位置对比学习及 DIWD 2026 等强近邻；它需要新数据、位置对照和严格一致的标签预算，工作量高于 A。把“全传感器→少传感器”本身作为创新已经不成立。

subject OOF、单参数温度校准、非目标概率排列、DKD 和 logit standardization 应作为识别机制与建立强基线的工具；不能仅把这些工具换成 HAR 名词就当成方法贡献。主张一个小机制成立，比先发明大框架更有希望；但一个局部正结果也不能保证发表。

## 2. 现有结果到底支持什么

以下数字来自我们已有实验及独立审查，属于开发集证据，不是外部论文结论。详见 `research/round3/review/teacher_intervention_audit.json`、`research/round3/experiments/` 和 `research/round4/overfit_audit.md`。

- 同一宽 32 学生、三个配对种子、同四名验证人员：CE 教师 KD 的 macro-F1 为 95.2173%，直接 LS 为 96.6735%；换 LS 教师后 KD 为 97.0220%。对原 KD 增加 1.8047 个百分点，对直接 LS 仅增加 0.3486 个百分点。
- 两个最佳教师均已达到 100% 训练分类准确率；训练窗上 T=2 的软目标平均熵分别为 0.0615585 与 1.3178585 nats，平均最大概率分别为 0.9869123 与 0.5985651。这里的 T=2 并没有让 CE 教师变得足够柔软。
- CE 教师从最佳 epoch 25 到 epoch 80，train NLL 从约 0.001234 降到 0.0000226，val NLL 从约 0.24358 升到 0.40816，而 val F1 只下降约 0.2585 个百分点。比“教师准确率严重崩溃”更贴切的描述是概率继续极端化、跨人员概率质量恶化。LS 教师的验证 NLL 更稳定。
- CNN 单种子 KD 93.7166%，CE 93.0945%，LS 93.0710%；输入为 UCI 提供的九通道预处理惯性时序窗。单种子和预算边界上的最佳点，不能支持稳定架构规律。

**已成立的是：教师训练目标改变了学生可获得的监督，并改变了开发集结果。尚未成立的是：变化由准确率、校准、目标熵、非目标类别关系中的哪一项独立造成。** 同一教师干预同时改变这些因素。更高熵不等于更好校准，训练集熵也不等于未见人员可靠性。当前四人验证集同时参与 checkpoint 选择，不能继续在该四人上选新机制后称其独立验证。

### 一个有解释力但不是新理论的精确联系

令 q 为教师目标分布，p 为学生分布，y 为真类。DKD [R6] 的分解为：

\[
\mathrm{KL}(q\|p)=\mathrm{KL}([q_y,1-q_y]\|[p_y,1-p_y])+(1-q_y)\mathrm{KL}\!\left(\frac{q_{\neg y}}{1-q_y}\middle\|\frac{p_{\neg y}}{1-p_y}\right).
\]

我们的两个教师对全部训练窗均分类正确，因此训练集平均 q_y 等于平均最大概率。原标准 KD 中，非目标条件分布项的平均乘数从 **0.0130877** 变为 **0.4014349**，约 **30.67 倍**。这是已有分解与已有缓存的推导；它只说明乘数变化，**不表示非目标 KL、梯度或因果贡献增加 30.67 倍**。这正是必须纳入 DKD 基线、不能直接把所有改善称为“额外平滑”的原因。DHMMD 已将 target/non-target 解耦用于 wearable HAR [R15]，因此 DKD 套 HAR 本身也不是新意。

## 3. 决定研究边界的文献证据

### 3.1 LS 与 KD 的争论已经发展到什么位置

**Müller et al., NeurIPS 2019 [R1]。** §4、Fig. 5（正文 pp.6–7）展示了 LS 教师在其图像设置中可能损害蒸馏，并以类别关系受抑制解释。不能读成普遍不相容定理。MNIST 对比还改变了教师正则化方式；CIFAR 实验的教师监督权重及温度设置也与我们 CE+KD 混合目标不同。

**Shen et al., ICLR 2021 [R2]。** 已阅读主体、主要实验及 Appendix A–E，并查看 Table 2 与 Fig. 5 的实际页面。主要图像实验使用 T=1、仅教师软监督；其 LS 定义为真类 1−α、其他类 α/(K−1)，不同于我们 PyTorch uniform-inclusive 的 ε 定义。Table 2（PDF p.7）ResNet-50→ResNet-18 的 ImageNet Top-1 从 71.425±0.038 变成 71.816±0.017；教师从 76.056 到 76.128。它直接阻止“发现 LS 教师能改善 KD”的创新声明。Appendix E 的混合硬软目标也给出支持例；长尾结果又表明并非任何条件都改善。Figure 5 是训练轨迹与若干运行的展示，不能拿其中单次曲线标注替代 Table 2 三次均值。其稳定性指标正文与 Appendix C 代码的统计对象/方差、标准差用语有差异，若复用必须自行明确定义，不能照抄名称。

**Chandrasegaran et al., ICML 2022 [R3]。** 已读 §2–7，查看 Table 2/3 页面；附录仅定向核查。作者用教师训练 LS 与蒸馏温度的交互协调上述争议，并分析学生倒数第二层表征的变化。Table 2（PDF p.8）LS 教师→ResNet-18 的 ImageNet Top-1：T=1 为 71.616，T=2 为 68.428，T=3 为 66.570；该结果不能直接外推我们六类 HAR 的混合硬软目标。它说明“增加温度总会帮助软目标”不成立；也说明研究 LS 教师与温度的交互早已有体系化工作。

**Sultan, EMNLP 2023 [R4]。** 本轮复核 §2–4、Table 2 和 Limitations。四个文本任务中标准 KD 的预测熵均低于 ERM，而 LS 更高；研究的主温度为 T=1、教师以 CE 训练。这是对“KD 必然靠降低置信度起作用”的反例，不能推出所有任务中的正则化解释都错误。作者明确只有一个随机种子。我们的 HAR 熵变化不能用它直接解释，须实际区分教师信息与平滑作用。

### 3.2 简单温度、置信度与关系消融已经有哪些先例

| 先例 | 已有内容与本轮阅读范围 | 对我们投稿定位的约束 |
|---|---|---|
| Born Again Networks, ICML 2018 [R5] | 已读 §3.2：CWTM 用教师最大置信度给样本加权；DKPP 排列非 argmax 概率，检验类别身份是否关键 | 置信度加权、打乱非目标概率不是新方法。可借作严格控制，创新必须来自新的可迁移规律 |
| DKD, CVPR 2022 [R6] | 已读 §3、Eq. 5/6 与 Table 1；将目标/非目标项解耦 | 原教师接近 one-hot 时非目标项受压制已有理论解释；需要强基线 |
| CTKD, AAAI 2023 [R7] | 已核原始 arXiv/AAAI 元数据及摘要；动态可学习温度、课程机制。未将出版社返回的访问挑战 PDF 当成论文 | 学一个温度、按训练阶段变温度不能直接叫创新 |
| NormKD, 2023 preprint [R8] | 已读 §3.3、Algorithm 1：按每例 logits 标准差调整教师/学生温度，并使用教师尺度平方作为每例损失因子 | 固定 T 对不同 logits 尺度产生不同软度早已有研究；它同时改变目标和权重，机制比较必须拆开 |
| Logit Standardization, CVPR 2024 [R9] | 已读方法与实验：分别标准化教师/学生 logits，再使用基温度与 KD | z-score logits、解除 teacher/student 尺度匹配不是新意；与 NormKD 的每例权重并不完全相同 |
| Fan et al., ICML 2024 [R10] | 本轮复核官方摘要、书目；将校准与 capacity mismatch 联系，建议排序损失。未声称本轮阅读全文 | “教师校准好所以 KD 好”和 ranking-based 替代 KL 已有直接先例 |
| Dao et al., ICLR 2021 [R11] | 已读 §2–4 及实验主体；已有 cross-fitting 处理教师过拟合，fold 外教师给 fold 内学生样本产软标签 | OOF KD 不是创新；可检验按人员分组的 OOF 相对样本 OOF 是否有额外价值，但不能套用其独立同分布理论担保 |
| Li et al., Array 2026 [R12] | 仅核出版社元数据/摘要，称按教师熵自适应温度与 LS；全文未取得 | “熵条件化温度+LS”已有极近邻，不能未读全文便声称首创，也不能捏造具体公式或实验 |

冻结教师、固定目标时，KL(q||p) 与软目标交叉熵对学生参数的梯度相同，差一个与学生无关的 H(q)。讨论 loss 差异时必须写清目标是否随学生或可学习温度变化，不能把某论文中的一般措辞不加条件移植。

### 3.3 HAR 中最直接的阻挡先例

**DIWD 2026 [R13] 是最需要认真对照的一篇。** 本轮已读 12 页全文，核对 Table 1/2、§4.2、§4.5、§6。它在 WEAR 的四位置同步加速度信号上训练多位置教师，推理保留右臂传感器，按人员分组五折；利用另外三名 meta 人员的标签给每例 logit/feature 蒸馏项生成影响权重。它已同时涉及跨人员、特权传感器、样本选择和元监督。

其 pooled OOF macro-F1 为监督 0.561820、固定 KD 0.571623、DIWD 0.638451；单个 seed。**§4.2 中 meta 标签仅 DIWD 可以使用，监督和固定 KD 只在 D_fit 上训练；§6 作者自己承认需补等 meta 标签访问对照。** 因此已报告增益不能全归因于选择机制。它还已有平均权重匹配、静态门控、去 lookahead、batch-shared 消融；这些也不是我们新提出的贡献。等“平均系数”不等于等“加权损失/梯度”，作者亦有说明。

**TASKED 2023 [R14]。** 本轮核对 §3、Algorithm 1、§4.1–4.3。它已将 transformer、人员对抗对齐、MMD 与自蒸馏组合。但必须区别其不同目标访问条件：§3.3 同时允许仅源人员或源+目标人员；Algorithm 1 的步骤 13–16 明确采样目标 Xt、St 进入 domain/MMD 更新。故只能据此说“跨人员 KD/对抗学习已有”，不能不加区分地将全部结果称作训练完全不可见目标人员的 domain generalization。

**DHMMD 2026 [R15]。** 原机构记录及此前检索明确 target/non-target 解耦、异构多教师、双向特征学习用于 wearable HAR。已核书目/摘要，未取得全文，因此不报告其具体公式或数值。

**传感器减少也有明确先例。** Rey et al. 2022 [R16] 从训练期额外位置向目标位置转移表示；Nguyen et al. 2024 [R17] 使用训练期同步多传感器与对比学习，推理单传感器/子集；Mardanpour et al. 2023 [R18] 已有多惯性传感器的特征 KD。上述文献均足以阻止把“减少轴/位置/传感器”本身作为新意。

## 4. 优先 A：具体研究问题与最小机制矩阵

### A 的准确问题

> 在固定可用标注人员、训练窗口和学生容量时，教师软目标中保留真实类别身份的非目标概率关系，是否比相同置信度与熵的无语义目标更能泛化到完全未见人员？这种收益由哪些源人员可估计的可靠性指标预测？

这里“完全未见”指外层测试人员的 X、y 均不进入预处理拟合、调温、校准、门控、训练或选参。贡献候选是**一个经过反事实控制的可迁移性结论，以及能在新人员/新数据上执行的选择规则**。目前还不是已发现规律。

### 先做小而可区分的矩阵

教师 CE 与 LS 各固定一套训练预算；先在一个预先指定轻量容量上做下面八臂。所有臂同一人员/窗口预算、优化器、初始化和 batch 顺序，内层验证选择规则与搜索预算相同。下列干预均是计划，未执行。

| 编号 | 学生目标或干预 | 识别什么；不能识别什么 |
|---|---|---|
| A0 | CE | 无软目标基准 |
| A1 | 直接 uniform LS | 教师是否提供超出便宜正则化的收益；ε 必须拥有合理、相同规模的内层选择机会 |
| A2 | CE 教师、标准 KD | 复核原始观察 |
| A3 | LS 教师、标准 KD | 复核教师训练目标干预；单独比较不能解释机制 |
| A4 | CE 教师 logits 的 teacher-only 熵匹配后 KD | 用每例一维温度求解，使目标熵匹配 A3 对应窗；保持学生温度/损失系数不变，不引入额外每例尺度平方。保持 CE 教师排序，但不保证 q_y 与 A3 相同，因此不能称全部概率因素已控制 |
| A5 | A3 的非真类概率固定排列 | 每窗固定一个排列，在所有 epoch 保持不变；精确保留 q_y、完整概率多重集、熵、最大概率和教师正确/错误状态，破坏非目标类别身份。避免每 epoch 重抽带来额外噪声正则化 |
| A6 | CE 教师 + Logit Standardization | 检查已知尺度修正是否足以关闭缺口 |
| A7 | CE 教师 + DKD | 检查已知解耦机制是否足以关闭缺口；不能把复现其改善当新方法 |

A5 是对 BAN DKPP 思路的有目的控制。使用真类 y 固定位置而非仅固定 argmax，是为了 OOF 教师可能答错时仍保持 q_y 与教师正确性状态；应明确报告此区别。需保存排列与对应分布，确保它是确定的配对消融。

不要把“q_y 保持，剩余概率均匀”叫作保熵：除非原非目标分布均匀，这会增加熵。它可以是额外的 confidence-only 控制，但不能替代 A5。同样，一个全局温度只能匹配总体平均熵，通常不能逐窗匹配；熵匹配也不是 NLL 校准。

这八臂先排除最容易混淆的解释。若 A6/A7 已完全解决问题，先接受已知方法解释，不急于再造门控。若机制信号仍存在，扩到两种学生容量、第二数据集/时序模型、预定外层人员折，再以相同搜索预算补齐最强教师×最强已知方法组合。投稿阶段不能只让拟议方法获得优化机会，而把所有比较方法固定为默认参数。

### subject OOF 与少参数校准怎么进入 A

将它作为第二阶段的估计方法：比较 in-sample、普通样本 OOF 与按人员分组 OOF 的教师 NLL/Brier、q_y、条件非目标熵及关系稳定性。OOF 已有 Dao 2021，潜在增量是**按人员阻断后所估计的可靠性是否更能预测跨人员蒸馏收益**，而不是 OOF 这个操作。

随机窗口 OOF 可能仍共享人员特征、重叠时间窗；要有连续时间块/会话阻断对照，并匹配教师训练窗口数和训练比例。否则“人员 OOF 差/好”可能只是训练量或近重复差异。所有源人员标注均计入同一个预算，不能让门控方法额外看到三名带标签人员却不计入与基线的比较。

只有信号成立后才考虑源人员 OOF NLL 拟合的单参数 T，或极少参数可靠性条件化。目标是实现可检验选择规则，不是因为参数少就宣称创新。Fan 2024、CTKD、NormKD、Logit Standardization 和 Array 2026 均构成先例边界。

### A 的继续/停止规则

- **继续机制论文路线：** 在预先定好的外层人员折、至少两个数据集/输入设置中，真实类别关系优于 A5 的等熵、等 q_y 控制；已知尺度/解耦基线不能完全解释；源人员上的关系稳定性或可靠性指标能预测新人员结果，并改善学生选择或最差人员表现。
- **继续方法路线的更高门槛：** 一个少参数规则在相同标签、搜索与训练预算下重复优于强基线，且其增益与前述机制一致。可预设例如 0.5–1 个 F1 百分点的实用门槛，但这是团队决策值，不是通用发表标准。
- **停止“新 KD 算法”路线：** A5 保留全部收益，或直接 LS/全局温度/DKD/logit standardization 已关闭缺口，或效果只存在原四人开发集。此时可保留严谨的条件性实证结论，毕业论文仍完整，但不将局部调参包装为投稿贡献。
- 统计单位必须承认人员聚类；报告配对人员差异及不确定性、窗口 pooled 指标与人员均值的区别。三个 seed 不是三个独立人员样本。最差人员指标需事先定义，不能事后挑人讲故事。

## 5. 次选 B：输入可观测性约束下的特权传感器蒸馏

### B 的准确问题

> 多位置教师对某段动作很有把握时，单位置学生是否有足够输入证据学到该区别？哪些教师目标在减少传感器后仍可迁移，哪些只表达学生看不到的信息？

这比“全传感器教师帮助单传感器学生”更具体。教师高置信不保证该动作区别对学生可观察；但仅凭单传感器模型预测失败，也不能证明信息理论意义上的不可观察，可能只是容量或优化不足。需要以配对输入和多个学生能力对照建立经验可恢复性证据。

### 最小有判别力的设计

1. 选有同步多位置数据的数据集，固定单位置学生与其容量。先比较两种保留位置，避免只在右臂上得一条结论。UCI HAR 的单一手机位置不支持“跨身体位置”声明；也不能把 561 特征列随意删除等同于关掉真实传感器。
2. 统一总标注人员/窗口预算。若使用三名 meta 人员估计门控，则 CE/LS、固定 KD、已知尺度/解耦方法应有同等标签访问；可以设计所有方法同一总人员池、不同合法训练角色的对照，并清楚列出角色。还要报告教师及元训练成本。
3. 基线至少包括单位置 CE/LS、固定 KD、相同标签预算下的 DIWD/简化版本，以及全位置模型作为信息上限参考。与 [R16–R18] 的表示迁移方法对照应选可复现者，而非只比弱 CE。
4. 用源人员 OOF 的单位置预测与多位置教师关系构造经验可恢复性代理，比较仅教师熵、仅学生困难度、二者交互。若使用一两个参数的门控，先证明它判断的是迁移价值而不只是降低总体蒸馏权重。需有平均权重/静态门控控制；DIWD 已经做过这些，不能宣称我们首次想到。
5. 错误分析必须落到动作对与传感器位置：同一类别关系在某位置可恢复、在另一位置不可恢复，比仅平均 F1 更接近问题。必须以未见人员和第二数据集核验，不能靠一组漂亮案例建立规律。

### B 的继续/停止规则

**继续：** 总标签访问一致后，输入条件化选择仍改善多个人员/位置，并且改进能由预先定义的经验可恢复性指标解释；简单规则能接近昂贵元梯度方法，或能系统避免其失败情形。**停止新方法主张：** DIWD/既有跨位置表示方法已解决，优势消失于等标签预算，或传感器受限输入缺失的信息在多个合理学生中都无法恢复。B 的价值不能只来自又加一个 gating 网络。

## 6. 图表证据的关键读法

- **Shen Table 2 / Fig. 5（PDF p.7）：** Table 2 是三次运行均值与误差，Fig. 5 是训练轨迹；它们回答“在特定协议下 LS 教师能不能改善学生”，不是“为什么改善”的独立因果证据。实际页面已查看。
- **Chandrasegaran Table 2/3（PDF p.8）：** Table 2 分开教师 LS 与 T 的组合，Table 3 对所选类别集合量化表征距离变化。前者直接显示交互，后者支持作者提出的表征解释；距离指标及选类策略不等于证明所有任务上的唯一机制。实际页面已查看。
- **DKD Eq. 5/6（PDF p.3）：** 精确代数恒等式与新目标设计必须区分。我们的 30.67 倍仅来自旧 KD 中的系数，不来自新训练。方法页正文已读。
- **DIWD Table 1/2 与 §6：** 结果表确有大幅提升；但训练标签访问表述改变其因果解释。不能因为它是预印本便忽略，也不能因数值高而忽略作者主动承认的对照缺口。已读全文与表格数值。

## 7. 完整书目与可核查入口

阅读状态：F=本轮阅读全文主要主体及必要附录；S=定向阅读方法/结果；M=元数据/摘要或先前全文笔记复核。下列 web ref 是根代理复核线索，不直接当作用户展示文本。不能取得开放 PDF 的条目明确标出，不补造地址。

### R1

Rafael Müller, Simon Kornblith, Geoffrey E. Hinton. 2019. **When Does Label Smoothing Help?** Advances in Neural Information Processing Systems 32 (NeurIPS 2019).

- 官方：https://proceedings.neurips.cc/paper/2019/hash/f1748d6b0fd9d439f71450117eba2725-Abstract.html
- 官方 PDF：https://proceedings.neurips.cc/paper/2019/file/f1748d6b0fd9d439f71450117eba2725-Paper.pdf
- 状态 S：先前全文笔记、§4/Fig.5 定位复核。Ref `turn787318722view1`。

### R2

Zhiqiang Shen, Zechun Liu, Dejia Xu, Zitian Chen, Kwang-Ting Cheng, Marios Savvides. 2021. **Is Label Smoothing Truly Incompatible with Knowledge Distillation: An Empirical Study.** International Conference on Learning Representations (ICLR 2021).

- 官方：https://openreview.net/forum?id=PObuuGVrGaZ
- 官方 PDF：https://openreview.net/pdf?id=PObuuGVrGaZ
- 作者预印本 PDF：https://arxiv.org/pdf/2104.00676
- 项目页：http://zhiqiangshen.com/projects/LS_and_KD/index.html （本轮未审计代码）
- 状态 F。Refs `turn933060977search6`, `turn787318722view2`。

### R3

Keshigeyan Chandrasegaran, Ngoc-Trung Tran, Yunqing Zhao, Ngai-Man Cheung. 2022. **Revisiting Label Smoothing and Knowledge Distillation Compatibility: What was Missing?** Proceedings of the 39th International Conference on Machine Learning, PMLR 162:2890–2916.

- 官方：https://proceedings.mlr.press/v162/chandrasegaran22a.html
- 官方 PDF：https://proceedings.mlr.press/v162/chandrasegaran22a/chandrasegaran22a.pdf
- 作者项目：https://keshik6.github.io/revisiting-ls-kd-compatibility/
- 作者代码：https://github.com/sutd-visual-computing-group/LS-KD-compatibility （链接已定位，未运行）
- 状态 S：主体 §2–7、Table2/3、定向附录。Refs `turn751038921view1`, `turn944014113view2`, `turn933060977search2`。

### R4

Md Arafat Sultan. 2023. **Knowledge Distillation ≈ Label Smoothing: Fact or Fallacy?** Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, 4469–4477. DOI: 10.18653/v1/2023.emnlp-main.271.

- 官方：https://aclanthology.org/2023.emnlp-main.271/
- 官方 PDF：https://aclanthology.org/2023.emnlp-main.271.pdf
- 状态 S：先前全文笔记，本轮复核 §2–4、Table2、Limitations。作者完整姓名按 PDF，官网元数据缩写为 Md Sultan。Ref `turn787318722view0`。

### R5

Tommaso Furlanello, Zachary C. Lipton, Michael Tschannen, Laurent Itti, Anima Anandkumar. 2018. **Born Again Neural Networks.** Proceedings of the 35th International Conference on Machine Learning, PMLR 80:1607–1616.

- 官方：https://proceedings.mlr.press/v80/furlanello18a.html
- 官方 PDF：https://proceedings.mlr.press/v80/furlanello18a/furlanello18a.pdf
- 状态 S：§3.2、方法与实验相关段落。PDF 姓名含 C.，PMLR 元数据省略中间首字母。Refs `turn504504090view0`, `turn928833856search9`。

### R6

Borui Zhao, Quan Cui, Renjie Song, Yiyu Qiu, Jiajun Liang. 2022. **Decoupled Knowledge Distillation.** Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 11953–11962. DOI: 10.1109/CVPR52688.2022.01165.

- 官方：https://openaccess.thecvf.com/content/CVPR2022/html/Zhao_Decoupled_Knowledge_Distillation_CVPR_2022_paper.html
- 官方 PDF：https://openaccess.thecvf.com/content/CVPR2022/papers/Zhao_Decoupled_Knowledge_Distillation_CVPR_2022_paper.pdf
- 作者代码：https://github.com/megvii-research/mdistiller （PDF 指定链接，未运行）
- 状态 S：§3、Eq5/6、Table1。Refs `turn787318722search0`, `turn787318722search46`, `turn787318722search2`。

### R7

Zheng Li, Xiang Li, Lingfeng Yang, Borui Zhao, Renjie Song, Lei Luo, Jun Li, Jian Yang. 2023. **Curriculum Temperature for Knowledge Distillation.** Proceedings of the AAAI Conference on Artificial Intelligence 37(2):1504–1512. DOI: 10.1609/aaai.v37i2.25236.

- 官方：https://ojs.aaai.org/index.php/AAAI/article/view/25236
- 官方 PDF：https://ojs.aaai.org/index.php/AAAI/article/download/25236/25008 （本次取得的是访问挑战页，未当作论文）
- 作者预印本：https://arxiv.org/abs/2211.16231
- 作者 PDF：https://arxiv.org/pdf/2211.16231
- 作者代码：https://github.com/zhengli97/CTKD
- 状态 M。Refs `turn504504090view2`, `turn389562611search7`。

### R8

Zhihao Chi, Tu Zheng, Hengjia Li, Zheng Yang, Boxi Wu, Binbin Lin, Deng Cai. 2023. **NormKD: Normalized Logits for Knowledge Distillation.** arXiv:2308.00520. 本轮未确认正式会议/期刊归属，不擅自补写。

- 作者原文：https://arxiv.org/abs/2308.00520
- PDF：https://arxiv.org/pdf/2308.00520
- 状态 S：§3.3、Algorithm1、实验。Ref `turn751038921view4`。

### R9

Shangquan Sun, Wenqi Ren, Jingzhi Li, Rui Wang, Xiaochun Cao. 2024. **Logit Standardization in Knowledge Distillation.** Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 15731–15740.

- 官方：https://openaccess.thecvf.com/content/CVPR2024/html/Sun_Logit_Standardization_in_Knowledge_Distillation_CVPR_2024_paper.html
- 官方 PDF：https://openaccess.thecvf.com/content/CVPR2024/papers/Sun_Logit_Standardization_in_Knowledge_Distillation_CVPR_2024_paper.pdf
- 状态 S：方法、Algorithm1/2、实验主体。Ref `turn493221796search0`。

### R10

Wen-Shu Fan, Su Lu, Xin-Chun Li, De-Chuan Zhan, Le Gan. 2024. **Revisit the Essence of Distilling Knowledge through Calibration.** Proceedings of the 41st International Conference on Machine Learning, PMLR 235:12882–12894.

- 官方：https://proceedings.mlr.press/v235/fan24d.html
- 官方 PDF：https://raw.githubusercontent.com/mlresearch/v235/main/assets/fan24d/fan24d.pdf
- 状态 M：官方摘要与完整元数据；本轮未声称全文精读。Ref `turn828284142view0`。

### R11

Tri Dao, Govinda M. Kamath, Vasilis Syrgkanis, Lester Mackey. 2021. **Knowledge Distillation as Semiparametric Inference.** International Conference on Learning Representations (ICLR 2021).

- 作者机构：https://www.microsoft.com/en-us/research/publication/knowledge-distillation-as-semiparametric-inference-paper/
- 作者预印本：https://arxiv.org/abs/2104.09732
- 作者 PDF：https://arxiv.org/pdf/2104.09732
- 作者代码：https://github.com/microsoft/semiparametric-distillation
- 状态 S：§2–4、实验主体；未逐行核验所有证明附录。Refs `turn751038921view2`, `turn944014113view3`。

### R12

Ruoyu Li, Huihua Wang, Lijun Yun, Zaiqing Chen, Yuelong Xia, Han Liang. 2026. **Beyond fixed hyper-parameters: Adaptive temperature and label smoothing for knowledge distillation.** Array 31:101121. DOI: 10.1016/j.array.2026.101121.

- 出版社：https://www.sciencedirect.com/science/article/pii/S2590005626004443
- 未取得可核验开放 PDF，不编造地址。
- 状态 M：出版社元数据、摘要/highlights；不能引述其未读公式和数值。Ref `turn744364827search3`。

### R13

Bingxuan Xie. 2026. **Dynamic Influence-Weighted Distillation for Single-IMU Activity Recognition.** arXiv:2608.24904. 预印本，本轮未确认经同行评审发表。

- 作者原文：https://arxiv.org/abs/2608.24904
- 作者 PDF：https://arxiv.org/pdf/2608.24904
- 论文列出代码：https://github.com/1304126986/wear-right-arm-privileged-distillation （本轮未运行或审计完整仓库）
- 状态 F：12 页全文、§4.2/4.5/6、Table1/2。Ref `turn744364827academia8`；PDF 已在本地，根代理已独立核查关键段。

### R14

Sungho Suh, Vitor Fortes Rey, Paul Lukowicz. 2023. **TASKED: Transformer-based Adversarial learning for human activity recognition using wearable sensors via Self-KnowledgE Distillation.** Knowledge-Based Systems 260:110143. DOI: 10.1016/j.knosys.2022.110143.

- 出版社：https://www.sciencedirect.com/science/article/pii/S0950705122012394
- 作者预印本：https://arxiv.org/abs/2209.09092
- 作者 PDF：https://arxiv.org/pdf/2209.09092
- 状态 S：§3、Algorithm1、§4.1–4.3、图1–3；不将全部目标访问设置混成 DG。Ref `turn928833856search3`。

### R15

Zhiwen Xiao, Huanlai Xing, Rong Qu, Hui Li, Ruibin Bai, Lexi Xu, Jincheng Peng, Li Feng, Qian Wan. 2026. **Decoupled Heterogeneous Multi-Teacher Mutual Distillation for Wearable Human Activity Recognition.** ACM Transactions on Embedded Computing Systems. DOI: 10.1145/3849875.

- DOI：https://doi.org/10.1145/3849875
- 机构：https://nottingham-repository.worktribe.com/index.php/output/72337013/decoupled-heterogeneous-multi-teacher-mutual-distillation-for-wearable-human-activity-recognition
- 未取得开放全文 PDF；正式卷期页码本轮不补造。
- 状态 M：此前官方机构元数据/摘要核验，2026-10-30 在线。Ref `turn637222875search0`；本轮机构 URL 返回 403，不能把错误页当新证据。

### R16

Vitor Fortes Rey, Sungho Suh, Paul Lukowicz. 2022. **Learning from the Best: Contrastive Representations Learning Across Sensor Locations for Wearable Activity Recognition.** Proceedings of the 2022 ACM International Symposium on Wearable Computers (ISWC), 28–32. DOI: 10.1145/3544794.3558464.

- DOI：https://doi.org/10.1145/3544794.3558464
- 作者预印本：https://arxiv.org/abs/2210.01459
- 作者 PDF：https://arxiv.org/pdf/2210.01459
- 状态 M：作者摘要/元数据，本轮不引用未核全文的性能表。Ref `turn828284142view3`。

### R17

Duc-Anh Nguyen, Cuong Pham, Nhien-An Le-Khac. 2024. **Virtual Fusion With Contrastive Learning for Single-Sensor-Based Activity Recognition.** IEEE Sensors Journal 24(15):25041–25048. DOI: 10.1109/JSEN.2024.3412397.

- DOI：https://doi.org/10.1109/JSEN.2024.3412397
- 作者预印本：https://arxiv.org/abs/2312.02185
- 作者 PDF：https://arxiv.org/pdf/2312.02185 （2023 作者稿，标题大小写及连字符与期刊略异）
- IEEE 作者摘要：https://ieee-sensorsalert.org/articles/virtual-fusion-with-contrastive-learning-for-single-sensor-based-activity-recognition/
- 状态 M：作者摘要/元数据、正式 DOI 对应。Refs `turn828284142view2`, `turn389562611search5`。

### R18

Malihe Mardanpour, Majid Sepahvand, Fardin Abdali-Mohammadi, Mahya Nikouei, Homeyra Sarabi. 2023. **Human activity recognition based on multiple inertial sensors through feature-based knowledge distillation paradigm.** Information Sciences, article 119073. DOI: 10.1016/j.ins.2023.119073.

- DOI：https://doi.org/10.1016/j.ins.2023.119073
- 出版社：https://www.sciencedirect.com/science/article/pii/S0020025523006588
- 本轮未获得可核验开放 PDF；卷号不补造。
- 状态 M：沿用已核官方书目，见 `research/har_kd_prior.md`，只作直接先例定位。

## 8. 对根报告的写作建议

敢于推荐 A，但把推荐理由写成它能把已有异常转化为可反驳的问题，而非“尚无人做过”。明确已有结果与未来验证两层。可以表述为：**当前最值得押注的是跨人员条件下软目标中真正可迁移的信息，而不是再设计一个更复杂的蒸馏损失。** B 是下一阶段具应用意义的备选，需要对传感器位置、标签访问和可恢复性给出实证证据。

不把 0.3486 个百分点开发集差值写成稳定优势；不把同一四人上的三个种子写成跨人泛化证明；不把 DIWD 的等平均权重消融误写成等标签对照；不把 DKD 已知分解或 OOF 已知过程包装为原创理论。
