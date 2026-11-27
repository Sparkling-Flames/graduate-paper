# Round 5：以 1D 时序 CNN 为中心的轻量 HAR 架构与蒸馏评审

日期：2026-11-25。内部分析；只检索文献、阅读代码和计算结构量，没有训练模型或下载大型数据。范围为手机/可穿戴惯性时序，不把视频或骨架 HAR 当作直接方法先例。

## 1. 先给裁决

**主线应明确写成“轻量 1D 时序 CNN 的结构预算与知识蒸馏”，不能只把 CNN 当成其他模型实验之后的附加验证。** 对毕业论文，先把一个清楚的 Conv1d 主干、学生容量、预算匹配、跨人测试和独立数据复现做完整；成熟方法仍然值得复现。对后续投稿，值得追问的是学生能表达什么时间结构，以及何种教师目标可被其学到，不能把常见模块组合当作创新。[C1–C7]

三个直接结论：

1. **“长感受野教师 → 短感受野学生”已被直接研究。** Yao et al. 的 Long Kernel Distillation 不仅提出这个范式，还包括多长核教师、输出和中间层蒸馏、自适应教师权重、核尺寸消融和嵌入式评价。任何只做这些组合的方案都不宜宣称新方法。[C1]
2. **“depthwise-separable CNN + KD”也已被直接研究。** LHAR 明确使用 depthwise 学生、跨人 HAR、教师集成和数据增强。TinyHAR/TinierHAR 则构成不依赖 KD 的轻量结构强对照。[C2–C4]
3. **本题还有非常清楚的实验价值。** 现有纯 CNN 中，宽度压缩与时间上下文尚未分开；标准窄卷积与 depthwise 结构尚未在近似相同参数/MAC 下比较；逐时间点的特征蒸馏是否要求学生表达超出自身局部感受野的信息，也值得检验。这些是本项目未完成的问题，不自动等同于全球未解决的文献空白。

推荐顺序：**毕业论文优先做匹配预算的纯 CNN + logit KD；时间感受野作为一个有解释力的结构扩展；时序中间特征蒸馏留作进一步研究。** 三个问题不是要求一次全部开展。

## 2. 现有 CNN 的一个关键诊断：宽度不等于时间上下文

实际静态核对：`research/round3/experiments/run_cnn_pilot.py` 中 `ConvHAR` 为：

`Conv1d(k=5,p=2) → ReLU → MaxPool1d(k=2,s=2) → Conv1d(k=5,p=2) → ReLU → GlobalAveragePool → Linear(6)`。

输入是 `[B,9,128]`。教师通道 32/64，学生 8/16。二者卷积核、池化和时间依赖结构相同，当前只改变通道宽度。

对局部卷积/池化路径，设初始感受野 r=1、采样间距 j=1：

\[
r_l=r_{l-1}+(k_l-1)d_lj_{l-1},\qquad j_l=j_{l-1}s_l.
\]

于是第一层卷积 r=5，池化后 r=6、j=2，第二层卷积后 **r=14**。50 Hz 下对应约 0.28 秒的局部覆盖。这个数是根据代码和标准卷积依赖公式推导的，**不是本轮测得的有效感受野**。[C10–C11]

关键限定：最后的全局平均池化会聚合整个 128 点窗口，故**不能说整个分类器“只看 14 点”**。准确表述是：池化前每个局部非线性特征至多联合处理 14 点，再把各位置特征汇总成全窗决策。全窗汇总并不等同于已经形成任意长程顺序关系的非线性表示。这里正是纯 CNN 可解释、可控的研究入口。

还须区分：

- **理论感受野**：由计算图允许的输入依赖范围决定；KD 不改变推理计算图，不能凭训练损失把它扩大。
- **有效感受野**：训练后不同位置的实际影响强弱，可用梯度等方法研究；它不是单凭 k、d 能得到的数。
- **最终分类器依赖范围**：GAP、全局注意力、SE 或 RNN 都可能引入全窗依赖；不能只算卷积支路便称为整个网络的感受野。
- **采样点与时间**：同样 128 点在 50 Hz 与 100 Hz 下的时间长度不同。不同数据集保持时间定义，比机械保持点数更重要。

### 2.1 最小感受野对照，直接接当前结构

保持第一层 k=5 与池化不变，仅调整第二层：

| 第二层 `(kernel,dilation)` | 第二层输出的理论局部跨度 | 与基准参数量的关系 | 可检验什么 |
|---|---:|---|---|
| (5,1) | 14 点 | 基准 | 当前结构 |
| (5,2) | 22 点 | 相同 | 扩大采样间距的效果 |
| (5,4) | 38 点 | 相同 | 更宽但稀疏的时间采样 |
| (17,1) | 38 点 | 更多 | 与 (5,4) 同跨度、不同采样密度和参数量 |
| (5,8) | 70 点 | 相同 | 进一步扩大跨度；不预设一定更好 |
| (33,1) | 70 点 | 更多 | 更大核的同跨度参照，非当前必做 |

注意 `(5,4)` 和 `(17,1)` 只匹配**名义跨度**，并不匹配采样位置、参数量或表示能力。扩大 dilation 也可能产生稀疏覆盖和边界问题；不能把它描述为没有代价的长核替代。初步实验取基准、(5,4)、(17,1) 三个足够，没必要遍历所有 kernel×dilation。

## 3. 直接文献先例：哪些创意已经被覆盖

| 文献 | 本轮可核实的技术内容 | 对选题的约束 | 阅读层次 |
|---|---|---|---|
| **Long Kernel Distillation，2025** [C1] | APR、ELK、SLK 长核教师指导短核学生；单/多教师；输出与中间层；标签引导的权重；USC-HAD 三种核尺寸消融 | 长核教师、短核学生、中间层、多教师、普通核尺寸扫描均不是新提案 | 出版社摘要、完整引言、贡献和章节片段；**未获全文** |
| **LHAR，正式卷期 2024，在线 2023** [C2] | 跨人 HAR、depthwise 学生、集成教师与多通道增强 | depthwise+KD 已是直接 HAR 先例 | IEEE 官方摘要与正式元数据；**未获全文/作者代码** |
| **TinyHAR，2022** [C3] | 模态/通道交互、时间提取与聚合的轻量结构 | 不应以大而旧的 CNN 当作唯一效率对照 | 官方机构摘要、作者模型源码；本轮未取得可读 PDF |
| **TinierHAR，2025** [C4] | residual depthwise-separable temporal conv、BiGRU、attention aggregation；14 数据集；结构消融和宽度缩放 | 单纯替换可分离卷积或缩放通道已有充分先例 | 前轮全文；本轮重读架构及作者模型源码 |
| **TCN-Inception，2024** [C5] | 并行多尺度卷积 + dilated TCN + residual/BN，四个传感器数据集 | “多尺度+膨胀+残差”不是空白组合 | 出版社摘要/引言和元数据；SSRN 下载实际返回 HTML，**未当全文 PDF** |
| **DMSCNet，2026** [C6] | 多尺度膨胀分支 + SE + 全局 temporal attention | 结构堆叠早有先例；完整模型不是极简纯 CNN | 官方 PDF 方法、评估、局限定点读；实际看 Fig.1、Tables 3–4 |
| **Mardanpour et al.，2023** [C7] | inertial HAR，response+feature KD、Tucker 分解，三轴教师→单轴学生 | 特征蒸馏和受限输入学生已有直接先例 | 官方摘要/highlights；**未获全文** |
| **DFKD，2026** [C8] | temporal sensor feature 去噪、autoencoder 压缩与 KD | “让学生中间特征更接近教师”不是未开发方向 | 官方摘要/引言片段与元数据；**未获全文** |
| **KD-HAR，2024** [C9] | 在 UCI HAR/WISDM 做 CNN 等教师到小学生压缩 | vanilla KD 是复现基线，不是论文新方法 | 前轮全文，本轮复用 |

2026 的补充边界：Hattab et al. 的 *Achieving more with less…* 出版社页面已经显示 dual-branch、dilated depthwise-separable 1D CNN；但卷期标注 **December 2026**，晚于本轮日期，本轮只有摘要与 Crossref，既不标“已全文阅读”，也不以其未核实的实验细节决定方案。[C16]

### 3.1 Long Kernel Distillation：必须严谨引用的结论

该论文明确对应我们原来可能想到的“长核教师教短核学生”。其三数据集是 **PAMAP2、WISDM、USC-HAD**，不是自动假定为 UCI HAR；摘要称最高改进 4.15%，但没有全文不能自行判定所有数值的基线、统计口径或 split。[C1]

可读片段还明确有 USC-HAD 的三种核尺寸消融（Fig.7）和嵌入式评估。因此不能写“此前没人研究核大小对蒸馏的影响”或“此前没有硬件评价”。所用 APR/ELK/SLK 分别借鉴连续点表示、大核重参数化和稀疏大核方向；不宜把它们误说成我们需要逐一新实现的三个通用标准类。

**尚不能从现有可读内容确认**：是否使用了严格人员独立验证；教师精度与感受野如何拆分；是否比较相同跨度的大核与膨胀核；是否系统交叉学生宽度；是否与同预算窄标准卷积充分对照。这些应写成需取得正文后继续核验的项目，不能把“当前看不到”改写成“作者没做”。

由此得到的研究定位是：我们可以开展**感受野实现方式、学生容量与可学蒸馏目标的受控实证**；仅复现其长→短思路对毕业论文有价值，但不足以构成投稿方法创新。

### 3.2 TinyHAR / TinierHAR 的实际代码含义

TinyHAR 的作者实现用 `[B,1,T,C_sensor]`、`Conv2d(kernel=(k,1))` 先沿时间提取特征，再显式进行跨传感器通道交互和时间聚合。它不是在人体三维空间上做 3D CNN。其模型源码可选 GRU/LSTM/attention 等时间模块；论文书目和源码默认选项要分开描述。[C3]

TinierHAR 本轮核实源码：`DepthwiseSeparableConv` 也是 `Conv2d((k,1))`，在 feature-map 轴上 depthwise，再 pointwise；前两块 max pooling，后面默认再添加配置指定的块，最后将 feature 与 sensor 维展平进 BiGRU，再注意力汇总。**它既不是纯 Conv1d 基线，也不能把源码第一维的 depthwise 直接解释成“每个真实传感器独立滤波”。** 其卷积阶段的 sensor 轴仍单独保留，GRU 输入阶段才展平。[C4]

这不妨碍用它作外部强对照，但当前主模型应维持清楚的 `[B,C,T]` 纯 Conv1d。没有必要为了“像最新论文”立即加入 GRU 和注意力；那会使感受野与结构效率的归因复杂化。

### 3.3 一篇新论文也需要审稿式核验：DMSCNet

实际查看 Fig.1（p.7）可见四个多尺度块及全局注意力。Table 4（p.15）中禁用 dilation 后，UCI F1 从 97.65 降到 97.25；这是该完整架构内的条件性消融，不能推导任意小 CNN 只加 dilation 都会获益。[C6]

三项不宜照抄的说法：

- 其 Table 3（p.14）称 TCN-Inception 为 strongest baseline，但同表 RealWorld 的 HART F1=89.65，高于 TCN-Inception=88.20；不能转述为已对所有最强对手作了同一显著性检验。
- p.7 描述 InceptionTime 分支直接作用 raw input、缺共享 bottleneck；然而原作者 `inception.py` 明确有共享 1×1 bottleneck（第35–51行）。不应拿该描述作为新的创新动机。[C12]
- 它把卷积分支跨度、SE/全局 CTA 与整体感受野混在一起叙述。按图示四个 stride=1、最大有效核9的串联分支，纯卷积路径最大跨度应为33；正文“约57”缺少可复核的路径说明，且全局 SE/CTA 本身已引入全窗依赖。我们应根据自己的计算图重新计算，不套用论文数字。

这些问题不等于否定整篇论文；它仍证明多尺度膨胀结构早有研究。其价值在于提醒本题选择透明、可审计的结构，而不是继续堆叠模块后用“更大 RF”解释一切。

## 4. 三个明确的 CNN 研究问题

### RQ1：相同或接近预算下，宽度压缩与可分离卷积压缩，哪一种更适合接受 KD？

**毕业论文优先级：最高。投稿增量：单独很低；作为后续方法的强基线非常必要。**

问题不应写成“depthwise 能不能减少参数”，因为答案从结构定义就能计算。应该问：在相近参数、MAC、相同时间感受野与跨人协议下，**标准窄 CNN 和 depthwise CNN 的准确率—延迟折中是否不同，KD 是否改变二者排序？**

可直接从现有学生做这个对照，不引入新框架：保持第一层标准卷积，在第二层用 `depthwise(k=5)+pointwise(k=1)`。下表为按现有含 bias、无 BN 架构代数计算的数值，不是本轮实测。MAC 仅计卷积/线性乘加，不含激活/池化/归一化等。

| 模型 | 隐藏通道 | 参数 | 单个 9×128 窗口的卷积+线性 MAC | 局部 RF |
|---|---|---:|---:|---:|
| 当前较宽 teacher | 32/64，标准卷积 | 12,166 | 840,064 | 14 |
| 当前 student | 8/16，标准卷积 | 1,126 | 87,136 | 14 |
| 第二层可分离 student | 8/16 | **662** | **56,928** | 14 |
| 接近预算的标准卷积 student | 7/8 | **664** | **58,288** | 14 |

这个 662 vs 664 参数的配对，比只将 depthwise 学生与 1,126 参数学生比较更有说服力。双方约相同预算且都用相同 RF，差异主要来自卷积因子化和通道分配，仍不是纯粹单变量的“算子效果”，需如实命名为结构预算比较。

**必要强 baseline**：各结构从头 CE；既定 LS 对照；相同冻结教师的 logit KD；相同 seed/批次顺序和选模规则；窄标准卷积预算对照。加入一个 TinierHAR/TinyHAR 作为已发表结构参照可选，不要求把作者整个工程换成主框架。

**指标与判断**：accuracy/macro-F1、每人/每类、参数、MAC、指定 CPU 上 batch=1 延迟中位数和 p95；KD 对同结构 CE 的变化，而不是只对 teacher 压缩比。若 depthwise 参数减少却 CPU 更慢，论文仍有结论：该后端上稀疏/分组算子开销抵消理论节省。不能用 MAC 代替这一实测。

**已有覆盖与剩余价值**：LHAR 已有 depthwise+KD，TinierHAR 已有结构效率与缩放。剩余价值主要是本题同预算、同协议下的选择证据，不宣称新方法。它非常适合毕业论文与学习卷积结构。

### RQ2：KD 的收益是否取决于教师与学生的时间上下文，而不仅是通道宽度？

**毕业论文优先级：高，建议作为唯一一个结构扩展。投稿潜力：有条件的中等，需要比普通核尺寸消融更强的控制。**

以第2节的基准、dilation=4、kernel=17 为候选。从固定一个中等学生开始，比较短上下文教师、同参数的扩张上下文教师以及同跨度的密集长核教师。之后才决定是否在三个学生宽度上重复，避免一开始把所有维度全交叉。

**必要强 baseline**：学生自身的较宽 RF 版本直接 CE 训练；短 RF 教师 KD；长 RF 教师 KD；同跨度的大核/膨胀核；教师自身验证质量、参数和训练成本。若只比“弱短核教师”与“更准且更多参数的大核教师”，不能归因于 RF。

**可解释的观察量**：同一学生上不同教师的配对 KD 增益；时间上下文更长时哪些活动混淆改变；不同测试人员是否同向；教师差异是否仅转化成不同置信度。梯度型有效感受野可选作诊断，但不将漂亮热图等同于成功知识迁移。

**严格理论边界**：在不改变学生推理结构的情况下，KD 无法扩大其理论局部 RF。可研究的是允许范围内特征与全窗汇总的学习质量，而不是宣传学生“继承了教师的物理感受野”。

**已有覆盖与剩余价值**：Long Kernel Distillation 已经直接覆盖核心范式和核尺寸扫描。可能的增量在于对同预算/同跨度实现、学生宽度和可学目标的系统受控证据；在取得其正文前，不应把这些具体控制说成已确认空白。仅将同一工作换成 UCI 或 MotionSense 不足以形成强投稿贡献。

### RQ3：时序中间特征蒸馏何时比 logit 蒸馏更有用，何时是在要求学生拟合其看不到的上下文？

**毕业论文优先级：可选；投稿潜力：三者中方法空间最大，但需要明确机制与反证。**

先限制为一个固定教师、一个学生、一个中间层，不上多教师、扩散模型或每层蒸馏。比较 logit KD、池化后的 feature hint，以及时间对齐的序列 feature hint。通道不同用训练期 1×1 projector；时间长度不同用明确的对齐/汇总规则；推理时移除 projector。

基本形式可为既有 CE+logit KD 加一个归一化 feature MSE。它是 FitNets 类基线的时序适配，**不是新损失名称**。[C13] 与其发明新缩写，优先比较通道投影 hint、池化 hint、Attention Transfer 的一维改写等成熟参照。[C14]

理论动机可以讲清楚：若学生某位置特征只能依赖局部集合 I_s，教师相应位置依赖更大的 I_t，那么两个输入在 I_s 内完全相同、在 I_t\I_s 内不同的时候，学生局部输出相同，教师输出却可能不同。强迫逐位置精确匹配一般存在不可实现的部分；GAP 后目标、局部 teacher hint 或更大的 student RF 是不同替代。这是一个可检验的表示约束，不是 KD 一定失败的定理。

**必要强 baseline**：无 hint 的相同 student、logit-only、单层 projector-MSE、池化 hint、同时间分辨率/同局部 RF 的 teacher-student 配对；若提新对齐机制，需与简单平均池化和已有 attention transfer 比较。不能只与 CE 比而忽略 logit KD。

**真正的成功标准**：跨人 macro-F1 和独立数据集上出现可重复的条件性改善；不是 feature cosine 更高。时序 hint 若只让训练特征更接近、测试识别下降，正好支持“目标可学性/过度约束”的反证。

**已有覆盖与剩余价值**：Long Kernel Distillation 已有中间层+输出；Mardanpour 有 feature+response；DFKD 专门改造学生特征；FitNets/Attention Transfer 是更早的一般先例。因此“加 feature KD”不新。潜在贡献必须落在**可学的时间尺度与对齐规则**，并说明何时减少提示、汇总提示或更换学生上下文更有效。

## 5. 必须保留的强 baseline，不把试验铺满

| 作用 | 最低配置 | 为什么不能省 |
|---|---|---|
| 纯 CNN 基础 | 同一个 Conv1d+GAP 家族，至少三个宽度 | 明确容量—性能关系；避免不同模块一起变 |
| 无蒸馏学生 | CE；沿用既定 LS 对照 | KD 增益必须相对相同结构训练法计算，不重复泛论 LS |
| 蒸馏基础 | 一个冻结 teacher 的 vanilla logit KD | 所有复杂 feature/多尺度蒸馏的直接对照 |
| 结构预算 | 参数/MAC 接近的窄标准卷积 vs depthwise | 避免“新结构只是更宽/更多参数”或仅对大模型比较 |
| 时间上下文 | 短核、膨胀核、同跨度长核，按阶段选少量 | 区分跨度、采样密度和容量 |
| 既有轻量结构 | TinyHAR 或 TinierHAR 选一个 | 防止只赢过一个过弱的自定义基线；也避免整套架构竞赛 |
| 既有多尺度结构 | 小型 Inception/TCN 仅在声称多尺度优越时加入 | InceptionTime/TCN-Inception 已覆盖相关组合 |

不要求所有行都成为同一毕业论文的全因子试验。当前建议核心为纯 CNN 三宽度×三训练法，增加 RQ1 的一个预算配对；RQ2/RQ3 只有在核心稳定后选其一。若导师更重研究解释，可把 RQ2 替代 RQ1 的扩展，而不是追加。

数据仍优先 UCI HAR + MotionSense，依已确定人员独立划分分别训练。它们的同类标签、采样管线和输入定义仍要清楚，不做一次未经校准的跨数据集直接迁移。已有官方 UCI test 应按当前封存规则使用，不能因为结构调试而提前看测试结果。

## 6. 阅读证据与源码的可复核性

- **Long Kernel Distillation**：出版社页面/搜索索引已读摘要、引言、贡献、kernel ablation/硬件 section snippets。Crossref 已核作者、卷316、113397、2025。Elsevier XML 403；OpenAlex 无可读 OA 位置；没有虚构 PDF 或代码。结论只以可读部分为依据。
- **LHAR**：IEEE 官方摘要/元数据、PubMed 交叉核对，正式卷期 2024/11；没有可用全文/代码，故不猜层数、核大小或数据集分割。
- **TinierHAR**：前轮全文在 `research/tinierhar.pdf`/`.txt`；本轮作者 `models/TinierHAR.py` 全段核查。没有运行模型。
- **TinyHAR**：机构记录与作者 `models/TinyHAR.py` 定点读；KIT PDF 本轮 503。机构作者稿与最终 Crossref 作者次序有差异，最终书目按 ACM/Crossref。
- **DMSCNet**：官方 PDF 27 页已下载；方法、协议、消融、结论定点读；Fig.1 p.7、Table3 p.14、Table4 p.15 用 Poppler 渲染后实际看图，不标全部27页精读。
- **TCN-Inception**：出版社摘要与引言、Crossref 正式元数据；SSRN 请求返回 HTML，已改 `.html`，没有伪装成成功 PDF。
- **其他**：Mardanpour/DFKD 为摘要层次；KD-HAR/TinierHAR 为前轮全文复用；InceptionTime 本轮核作者代码；FitNets/AT 本轮核原始论文元数据与方法摘要。

本轮保存的原始元数据、源码、可读 PDF 位于 `research/round5/cnn_sources/`。临时图表 PNG 仅用于内部核验，交付前删除。

## 7. 完整参考文献、官方/作者 PDF 与代码

### C1. Long Kernel Distillation：直接先例

Yao, Minghui; Cheng, Dongzhou; Zhang, Lei; Wang, Li; Wu, Hao; Song, Aiguo. (2025). **Long kernel distillation in human activity recognition.** *Knowledge-Based Systems*, **316**, 113397. DOI: **10.1016/j.knosys.2025.113397**.

- 官方页面：https://www.sciencedirect.com/science/article/pii/S0950705125004447
- 摘要页：https://www.sciencedirect.com/science/article/abs/pii/S0950705125004447
- 元数据：https://api.crossref.org/works/10.1016/j.knosys.2025.113397
- **官方/作者可读 PDF：本轮未取得。代码：未核实公开作者仓库。** 不用猜测出的 GitHub/PDF 地址替代。
- 工具 refs：`turn641417467search0`；**`turn650690773search0`**（含三个数据集、Fig.7 核尺寸消融和硬件评价片段）；`turn650690773search1`。

### C2. LHAR：depthwise 学生 + KD 的直接先例

Deng, Shizhuo; Chen, Jiaqi; Teng, Da; Yang, Chuangui; Chen, Dongyue; Jia, Tong; Wang, Hao. (2024; online 2023). **LHAR: Lightweight Human Activity Recognition on Knowledge Distillation.** *IEEE Journal of Biomedical and Health Informatics*, **28**(11), 6318–6328. DOI: **10.1109/JBHI.2023.3298932**.

- 官方论文：https://ieeexplore.ieee.org/document/10195168/
- PubMed 元数据：https://pubmed.ncbi.nlm.nih.gov/37494155/
- **可读官方/作者 PDF 与代码本轮未取得。**
- 工具 refs：**`turn707487533search1`**（IEEE，刊载2024、在线26 July2023）、`turn707487533search0`。

### C3. TinyHAR

Zhou, Yexu; Zhao, Haibin; Huang, Yiran; Riedel, Till; Hefenbrock, Michael; Beigl, Michael. (2022). **TinyHAR: A Lightweight Deep Learning Model Designed for Human Activity Recognition.** *Proceedings of the 2022 ACM International Symposium on Wearable Computers*, 89–93. DOI: **10.1145/3544794.3558467**.

- 官方机构页：https://publikationen.bibliothek.kit.edu/en/1000150216
- 官方机构 PDF：https://publikationen.bibliothek.kit.edu/1000150216/149254683 （本轮请求503，不能称成功下载）
- 出版商 PDF 入口：https://dl.acm.org/doi/pdf/10.1145/3544794.3558467
- 作者代码：https://github.com/teco-kit/ISWC22-HAR
- 核查源码：https://github.com/teco-kit/ISWC22-HAR/blob/main/models/TinyHAR.py
- refs：`turn347432275search0`；正式作者次序已由 Crossref 核：Riedel 在 Hefenbrock 之前，KIT 作者稿记录相反，不混写版本。

### C4. TinierHAR

Bian, Sizhen; Liu, Mengxi; Fortes Rey, Vitor; Geißler, Daniel; Lukowicz, Paul. (2025). **TinierHAR: Towards Ultra-Lightweight Deep Learning Models for Efficient Human Activity Recognition on Edge Devices.** *Proceedings of the 2025 ACM International Symposium on Wearable Computers*, 163–169. DOI: **10.1145/3715071.3750410**.

- 作者 PDF：https://arxiv.org/pdf/2507.07949
- 元数据：https://arxiv.org/abs/2507.07949
- 代码：https://github.com/zhaxidele/TinierHAR
- 本轮核查模型：https://github.com/zhaxidele/TinierHAR/blob/main/models/TinierHAR.py
- 配置：https://github.com/zhaxidele/TinierHAR/blob/main/configs/model.yaml
- 数据协议：https://github.com/zhaxidele/TinierHAR/blob/main/configs/data.yaml
- refs：`turn609137983view1`（前轮）；前轮全文 refs `turn620778909view0`、`turn351772627view0`。本轮源码直接下载至 `cnn_sources/tinierhar_model.py`。

### C5. TCN-Inception

Al-qaness, Mohammed A. A.; Dahou, Abdelghani; Trouba, Nafissa Toureche; Abd Elaziz, Mohamed; Helmi, Ahmed M. (2024). **TCN-Inception: Temporal Convolutional Network and Inception modules for sensor-based Human Activity Recognition.** *Future Generation Computer Systems*, **160**, 375–388. DOI: **10.1016/j.future.2024.06.016**.

- 官方页面：https://www.sciencedirect.com/science/article/pii/S0167739X24003182
- 摘要：https://www.sciencedirect.com/science/article/abs/pii/S0167739X24003182
- 作者预印本记录：https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4705410
- SSRN 检索给出的 PDF 请求本轮实际返回 HTML，**未取得可读 PDF**。未核实官方代码。
- refs：`turn707487533search4`、`turn631979002search22`；作者、卷页已由 Crossref 核查。

### C6. DMSCNet

Wu, Qingshan; Chu, Shengguang; Li, Kewen; Wang, Liechong. (2026). **DMSCNet: A Dilated Multi-Scale Contrastive Attention Network for Sensor-Based Human Activity Recognition.** *Applied Sciences*, **16**(12), 6037. DOI: **10.3390/app16126037**.

- 官方全文：https://www.mdpi.com/2076-3417/16/12/6037
- **官方 PDF，已成功下载**：https://www.mdpi.com/2076-3417/16/12/6037/pdf
- 作者代码：本轮未找到公开源码链接；Data Availability 只列数据来源。
- refs：`turn275828721view1`、`turn768542156view0`（LOSO与Table3）、`turn768542156view3`（Table4）。
- 阅读定位：§3 pp.6–10，§4.1–4.5 pp.11–15，结论/数据声明 pp.24–25；图表实际核图页7/14/15。

### C7. inertial feature-based KD

Mardanpour, Malihe; Sepahvand, Majid; Abdali-Mohammadi, Fardin; Nikouei, Mahya; Sarabi, Homeyra. (2023). **Human activity recognition based on multiple inertial sensors through feature-based knowledge distillation paradigm.** *Information Sciences*, **640**, 119073. DOI: **10.1016/j.ins.2023.119073**.

- 官方：https://www.sciencedirect.com/science/article/pii/S0020025523006588
- 可读官方/作者 PDF、作者代码：本轮未取得，不编造。
- ref：`turn347432275search4`。

### C8. DFKD

Wang, Li; Zhang, Lei; Cheng, Dongzhou; Yao, Minghui; Wu, Hao; Song, Aiguo. (2026). **Diffusion-facilitated knowledge distillation in human activity recognition.** *Neurocomputing*, **669**, 132517. DOI: **10.1016/j.neucom.2025.132517**.

- 官方：https://www.sciencedirect.com/science/article/pii/S0925231225031893
- 摘要：https://www.sciencedirect.com/science/article/abs/pii/S0925231225031893
- 可读 PDF/代码：本轮未取得；DOI 的2025不是正式卷期年份。
- refs：`turn641417467search4`、`turn490265784search3`、`turn631979002search17`。

### C9. KD-HAR

Gonçalves, Paulo H. N.; Bragança, Hendrio; Souto, Eduardo. (2024). **Efficient Human Activity Recognition on Wearable Devices Using Knowledge Distillation Techniques.** *Electronics*, **13**(18), 3612. DOI: **10.3390/electronics13183612**.

- 官方全文：https://www.mdpi.com/2079-9292/13/18/3612
- 官方 PDF：https://www.mdpi.com/2079-9292/13/18/3612/pdf
- 本地已读：`research/kdhar2024.pdf` / `research/kdhar2024_pdf.txt`。
- 本轮无新增作者代码核验。前轮 refs：`turn930433990view0`、`turn655801646view0`、`turn655801646view1`。

### C10. 感受野公式与边界

Araujo, André; Norris, Wade; Sim, Jack. (2019). **Computing Receptive Fields of Convolutional Neural Networks.** *Distill*. DOI: **10.23915/distill.00021**.

- 官方原始研究全文：https://distill.pub/2019/computing-receptive-fields/
- 本轮读公式/边界/路径推导，不声称另有已下载 PDF。
- ref：`turn202264287view0`。

Luo, Wenjie; Li, Yujia; Urtasun, Raquel; Zemel, Richard. (2016). **Understanding the Effective Receptive Field in Deep Convolutional Neural Networks.** *Advances in Neural Information Processing Systems 29*.

- 作者 PDF：https://arxiv.org/pdf/1611.03530
- 作者页：https://arxiv.org/abs/1611.03530
- 本轮仅核原始论文入口与基本区分，不以其视觉实验数值证明 HAR 结果。

### C11. Conv1d / depthwise 官方定义

PyTorch Contributors.（持续更新，访问2026-11-25）. **torch.nn.Conv1d.** *PyTorch Documentation*.

- 官方：https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
- 本轮 stable 跳转的页面：https://docs.pytorch.org/docs/2.15/generated/torch.nn.Conv1d.html
- refs：`turn392326338view2`、`turn170982839view0`、`turn170982839view1`。
- 这里只核实算子语义和权重形状，不要求项目升级 PyTorch。

### C12. InceptionTime 与标准 FCN 基础

Ismail Fawaz, Hassan; Lucas, Benjamin; Forestier, Germain; Pelletier, Charlotte; Schmidt, Daniel F.; Weber, Jonathan; Webb, Geoffrey I.; Idoumghar, Lhassane; Muller, Pierre-Alain; Petitjean, François. (2020). **InceptionTime: Finding AlexNet for Time Series Classification.** *Data Mining and Knowledge Discovery*, **34**, 1936–1962. DOI: **10.1007/s10618-020-00710-y**.

- 作者 PDF：https://arxiv.org/pdf/1909.04939
- 作者记录：https://arxiv.org/abs/1909.04939
- 作者代码：https://github.com/hfawaz/InceptionTime
- 本轮明确核查 bottleneck：https://github.com/hfawaz/InceptionTime/blob/master/classifiers/inception.py
- refs：`turn792950550view0`、`turn392326338view3`。本轮不是全文精读；源码35–51行已核实共享 bottleneck。

Wang, Zhiguang; Yan, Weizhong; Oates, Tim. (2017). **Time series classification from scratch with deep neural networks: A strong baseline.** *2017 International Joint Conference on Neural Networks (IJCNN)*, 1578–1585. DOI: **10.1109/IJCNN.2017.7966039**.

- 作者 PDF：https://arxiv.org/pdf/1611.06455
- 作者记录：https://arxiv.org/abs/1611.06455
- ref：`turn792950550view1`；最终会议/页码由 Crossref 已核。

### C13. FitNets（一般特征蒸馏基线，不是直接 HAR 实证）

Romero, Adriana; Ballas, Nicolas; Kahou, Samira Ebrahimi; Chassang, Antoine; Gatta, Carlo; Bengio, Yoshua. (2015). **FitNets: Hints for Thin Deep Nets.** *International Conference on Learning Representations (ICLR)*.

- 作者 PDF：https://arxiv.org/pdf/1412.6550
- 作者记录：https://arxiv.org/abs/1412.6550
- ref：`turn392326338view0`。本轮核方法摘要；不将 CIFAR 结果当 HAR 改善证据。

### C14. Attention Transfer（一般方法基线）

Zagoruyko, Sergey; Komodakis, Nikos. (2017). **Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer.** *International Conference on Learning Representations (ICLR)*.

- 作者 PDF：https://arxiv.org/pdf/1612.03928
- 作者记录：https://arxiv.org/abs/1612.03928
- 原记录链接代码：https://github.com/szagoruyko/attention-transfer
- ref：`turn392326338view1`。一维改写是我们拟用的对照，不能说原论文直接验证了本项目的 HAR。

### C15. 时序 dense labeling 与本题不同的边界案例

Lee, Yoo-Kyung; Son, Chang-Sik; Kang, Won-Seok. (2025). **Depthwise-Separable U-Net for Wearable Sensor-Based Human Activity Recognition.** *Applied Sciences*, **15**(16), 9134. DOI: **10.3390/app15169134**.

- 官方全文：https://www.mdpi.com/2076-3417/15/16/9134
- 官方 PDF 入口：https://www.mdpi.com/2076-3417/15/16/9134/pdf
- 机构记录：https://scholar.dgist.ac.kr/handle/20.500.11750/59271
- refs：`turn218642775search1`、`turn218642775search2`。本轮只读摘要/模型片段。该文逐时间点 dense prediction，与当前一窗一标签不同，不当即插即用同任务 baseline。

### C16. 2026 后续刊期结构工作，仅摘要核验

Hattab, Abdessalam; Haddad, Tarek Amine; Boudjema, Ali. (2026，出版社卷期标注December). **Achieving more with less: An ultra-lightweight dual-branch multi-level depthwise separable convolution for sensor-based activity recognition.** *Computers and Electrical Engineering*, **140**, 111561. DOI: **10.1016/j.compeleceng.2026.111561**.

- 出版社页面：https://www.sciencedirect.com/science/article/pii/S0045790626006312
- Crossref：https://api.crossref.org/works/10.1016/j.compeleceng.2026.111561
- refs：`turn218642775search0`、`turn641417467search5`。
- 本轮日期早于其列示刊期，只确认出版社页面和摘要可检索；未取得全文/PDF/代码，不虚构 online publication date，也不列为必须复现的成熟实现。

## 8. 最后给根代理的决策建议

1. 报告主标题、研究问题、结构图和实验计划都以 1D CNN 为主，输入是惯性时间序列，不引入3D/视频/skeleton。
2. 最有把握的毕业实验：现有 CNN 三宽度、同结构 CE/LS/KD、MotionSense 重复；增加 662 vs664 参数的结构预算配对即可产生清楚的工程与学习价值。
3. 如果要更研究型，选择“时间上下文与教师目标可学性”这一条，不把 long-kernel teacher 本身称新。
4. 后续投稿应围绕可重复的失败/改善条件和明确机制，或一种经过强 baseline 证明有效的可学时序 hint；泛 feature KD、depthwise、attention、多尺度和多教师均已有先例。
5. Long Kernel Distillation 全文未得是明确的检索边界，不能在最终回答中装作已经排除全部相近实验；但摘要与引言已足以否定长→短范式的原创性主张。

## 附录：对“固定学生局部输入、改变外部上下文”的机制 probe 做短理论审稿

这是本轮已有文献与计算图分析上的推导，没有新增检索或训练。**该 probe 可作为机制诊断，但其最致命漏洞是：把计算图必然造成的学生不变性，加上人工输入下的教师敏感性，直接解释为真实 HAR 中不可学、且妨碍分类的时序知识。** 这三件事并不等价。

1. **先保证 probe 测到的确实是局部依赖。** 固定的是网络实际输入中该位置真实支持集合内的全部通道，而不只是名义跨度；池化偏移、dilation 空洞、padding 和 projector/对齐算子的支持都要计入。用 eval 模式和固定训练集归一化统计量；关闭 dropout，BN 用冻结统计量。不能让逐窗口标准化、跨时间归一化、SE/attention 或全局 projector 重新引入外部依赖。当前无 BN 的两层 CNN 很适合作为这个干净起点。选择远离边界的位置，并报告学生变化是否只在数值误差内。学生不变是必要的实现核验，不是新的实验发现。

2. **人工对不自动代表真实条件分布。** 对平方 hint loss，若局部输入为 U、外部上下文为 V、教师目标为 H=h(U,V)，任何只依赖 U 的无限容量预测器的最优目标为 E[H|U]，其最低期望误差是 E[tr Cov(H|U)]。这是对所考察数据分布的结论。人工固定 U 后任意改 V，只能证明在该人工分布中存在无法同时精确拟合的目标；真实 HAR 中 V 可能由 U 高度预测。有限连续样本还可能被大容量学生记忆。因此 probe 不能直接估计真实不可约误差，也不能单独证明宽度增加无效。

3. **把拼接伪影和任务无关敏感性排除在主要解释之外。** 机械上下文交换可保留作结构压力测试，但另需小幅、平滑、保持通道物理关系的外部扰动，或同活动真实片段中局部近似匹配的配对；后者只能主张近似，不冒充完全相同输入。保留 sham/no-change 对照，比较短 RF 教师与长 RF 教师。教师 feature 差异大并不说明差异有分类价值；需检查教师真实类别预测、置信度以及相应活动混淆是否受到影响。单纯 nuisance 特征也可能被大核教师带入 hint。

4. **决定性实验是“RF × hint 类型”的真实性能交互。** 在同一冻结教师下，取参数相同或接近的短 RF 与扩张 RF 学生；各自比较 logit-only、逐时间点 hint、池化 hint，必要时加入局部 RF 匹配的教师 hint。主结果看相对各自 logit-only 的增量，而不是宽 RF 学生的绝对分数。若逐时刻 hint 的惩罚在短 RF 明显、扩张 RF 后减轻，而池化 hint 也能减轻，才形成支持机制的干预证据。仅宽 RF 的 CE 更好，不能说明知识迁移更好；仅 hint MSE 下降也不够。

5. **以跨人、独立数据复现约束投稿解释，并保留反证。** 先在训练/验证人员设计 probe 和规则，最终检验跨人 macro-F1、配对种子及类/人员分解，在 UCI HAR 与 MotionSense 分别重复。可以检查受上下文影响较强的活动是否更容易出现 hint 负迁移，但不能事后挑类。若教师特征对外部敏感、学生局部不变，却没有可重复的 hint 性能差异，则应接受“这个结构约束未成为当前任务瓶颈”；若不可精确拟合的 hint 仍改善分类，也不矛盾，学生可能学到了有用的条件均值或获得正则化。此时不能把 probe 强行解释为失败机制。

**投稿问题可以成立，但合适表述应是：逐时刻蒸馏目标中的外部上下文依赖，何时会形成学生的有效学习负担，是否能用学生可访问的时间尺度进行汇总来改善迁移？** 数学上的局部不变性本身不新；有价值的贡献须来自真实任务中的可重复交互、可预测的失败条件，以及优于 logit-only / pooled hint / 简单 RF 扩张等强对照的解决方案。毕业论文无需承诺全部完成这套机制研究。
