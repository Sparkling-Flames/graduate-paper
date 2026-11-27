# 1D CNN 蒸馏的后续研究：公开预训练教师与时间／采样预算

检索与实物核验：2026-11-25 UTC。内部研究备忘，不是已经获得的实验结论。本文没有训练模型，没有读取任何封存测试样本；HARNet 的验证只使用合成输入。当前 UCI／MotionSense 实验由其他代理独立执行，本文不以那些尚未完成的结果选择研究假说。

## 1. 明确判断

最值得推进的是：**公开可穿戴预训练 1D ResNet 的知识，在学生容量与物理观测预算同时收紧时，哪些能够保留下来？** 先用相同传感器、相同真实时间窗确认预训练教师值得蒸馏，再分别缩短学生上下文或降低采样率。这里可以研究的是可迁移知识的边界、失败原因和有效传递方式，而不是再包装一个通用 KL／feature MSE 损失。

这比“LS 教师是否更好”更接近时序 CNN 的结构与输入特点，也更有潜力形成后续投稿题目。旧 MLP 的教师目标熵发现只能作为需要控制的混杂因素；不能直接推断 CNN 也由同一机制支配。

优先顺序：

1. **A：公开预训练运动表征能否被极小 CNN 保留？** 用实际可下载 HARNet 权重，不训练大规模教师。固定标注人员／窗口预算，比较 task-only 学生、logit KD、全局特征 KD、时间关系 KD。强点是外部预训练带来的表征知识是否在强压缩后仍有用。
2. **B：时间／采样预算收紧时，教师优势的可继承边界在哪里？** 固定输入位置与活动标签，分别研究高率→低率、长观察窗→短观察窗。把预测收益与学生实际可观测信息联系起来，并考察跨人员稳定性。这比泛称“鲁棒蒸馏”更具体。
3. **C：预训练教师的姿态／缺失鲁棒性能否传给小 CNN？** 作为 A、B 的扩展，必须比较同等增强下的直接学生与普通 KD。clean→corrupted、feature／relation KD 已有直接 HAR 先例，不能当作新发明。

## 2. 公开教师已经核验到什么程度

### 2.1 最现实的教师：Yuan 等人的 HARNet

**正式出处是 npj Digital Medicine 2024，不是 Nature Communications。** [R1] 用约 700,000 person-days 的腕部加速度数据做自监督预训练，正式论文 Methods 说明其编码器为 18 层、1D convolution 的 ResNet-V2；三个预训练任务涉及时间方向、片段排列和时间形变。八个下游数据集的评估主要证明大规模预训练有迁移价值，并未证明该价值能够压缩进几百到几千参数的 CNN。

正式 PDF 已下载并阅读 Methods、数据协议、性能表及 Figure 3。Figure 3 已考察标注人数与预训练规模，所以“预训练可节省标签”本身不是新问题。本文的候选问题应落实到**预训练收益在小容量／小观测预算下是否仍成立**。

官方代码固定在 commit `150550ea5d41800229c95e36f88f5bf0d2e7cf04`。已下载两份官方权重、逐项检查参数形状并进行合成输入推理：

| 模型 | 声明的物理输入 | 编码器参数 | 加一个随机六类头后的参数 | 本地实物检查 |
|---|---|---:|---:|---|
| harnet5 | 3 轴加速度，30 Hz，5 s，即 N×3×150 | 4,227,904 | 4,493,638 | 131 项 feature state 完整加载，输出有限 |
| harnet10 | 3 轴加速度，30 Hz，10 s，即 N×3×300 | 10,457,408 | 10,985,286 | 131 项 feature state 完整加载，输出有限 |
| harnet30 | 3 轴加速度，30 Hz，30 s，即 N×3×900 | 本轮未实测 | 本轮未实测 | 官方 hub 定义和权重文件存在，本轮未下载运行 |

六类头的参数只是可复算的适配示例，不是官方提供了一个已经训练好的六类 HAR 分类器。**预训练权重只赋予 feature extractor；classifier 是随机初始化，必须用源训练人员标签拟合，或直接做无需分类头的表征蒸馏。** 冻结编码器时还要固定 BatchNorm 的运行统计，不能只设置 `requires_grad=False`。

代码对真实输入有实质约束：

- 单位应匹配作者处理流程的 **g**。官方 WISDM parser 把 m/s² 除以 9.81；PAMAP2 parser 除以标准重力常数，并裁剪到 ±3 g。不能把现有九通道 z-score 数据直接扔给预训练教师。
- 这是三个加速度轴的教师，不含陀螺仪。优先使用含重力的加速度，不能把 body acceleration、gyro 或任意三个特征列当作同一种输入。
- 官方代码对标称时长设计了连续降采样阶段。本轮将 2.56 s 转为约 77 个 30 Hz 点后，harnet5 与 harnet10 都在后段卷积／降采样处报长度不足。**现有 UCI 128 点窗不能无改动直接调用。把 77 点插值拉长到 150／300 点会改变运动时间尺度，不是获得了真实 5／10 秒上下文。**
- harnet10 中间输出依次是 64×150、128×75、256×15、512×3、1024×1。最后一个 1024 维向量已经没有展开的时间轴；若研究时间关系，应取 layer2 或 layer3 等中间层，而不是把最终 pooled embedding 称作时间结构。

实物证据：`research/round5/harnet_weight_audit.json`；可重复检查脚本：`research/round5/check_harnet_weights.py`。该检查证明权重可取得、结构可加载和输入要求，不证明任何真实下游准确率。编码器权重 SHA256：harnet5 `74ffaefbafb467b7253d121ef71fe734ee03bc802ba33c33a6aa1aef7f91de5e`；harnet10 `c64f9135d99e2dcdfc9ae7cc0672f2bcc438df9ceb8215665882f92cddd162a6`。

许可文件是 academic non-commercial 许可，不能标注为 MIT 或任意商业使用。毕业论文研究并不因此受阻。

### 2.2 两个更近但不能直接承诺使用的模型

**RelCon，ICLR 2025 [R2]。** 论文采用 1D ResNet-34（约 3.9M 参数），100 Hz、2.56 s 输入，学习与动作关系相关的距离表示，并评估跨身体位置。它与当前短窗 CNN 的概念接口很好，但官方 README 明确说明不能发布预训练数据和对应权重。可以学习其方法、采用其代码，不能宣称现在已能下载其大规模预训练教师。Table 4 的冻结表征并非在所有跨位置任务上胜过监督模型，因此“foundation model 自动跨位置泛化”不是事实。

**Inertia-1，2026 [R3]。** 作者报告在 18.2M 小时数据及 15 个下游数据集上研究采样率、窗口、轴、位置、架构和规模。作者仓库公告 NeurIPS 2026 接收；本轮阅读的是 arXiv 作者版，而非尚未核验的会议定稿页。代码仓库能取得，但 README、树文件和权重 issue 中未核到可直接用的预训练 checkpoint。本轮不把它列为已就绪教师。它也意味着“研究采样率／窗长／轴数的预训练作用”已有强近邻，投稿需要加入**压缩和知识继承的具体问题**。

## 3. 哪些已有工作会阻止虚假创新

| 直接先例 | 核验到的真实内容 | 对我们选题的约束与仍可研究的问题 |
|---|---|---|
| SelfHAR，Tang et al., IMWUT 2021 [R4] | 已将 teacher pseudo-label、自训练、无标签数据、时序增强与自监督学习结合；方法章节 3.2.1–3.2.5 可全文核查 | 不能称首次结合 SSL 与 KD。它不是把现成约千万参数预训练教师压缩进极小 CNN 的完整答案 |
| SMLDist，Chen et al., IEEE Sensors Journal 2024 [R5] | stage-level 时间特征与 FFT 特征蒸馏，结合中间知识与输出；cross-user 数据和 REALDISP ideal→displaced 实验；作者版 pp.4–8 | “中间 feature／频域／跨人／佩戴位置 KD”早已有。仍可研究公开预训练 prior、极端容量及预算变化；原方法还要求 stage 特征尺寸匹配 |
| HKDF，Hu et al., 2025 [R6] | 干净教师指导噪声／连续缺失学生，cross-sample 与 class-prototype 关系，KL+CE；6页全文 | “clean teacher→corrupted student + relation KD”不是新。教师／学生为同结构 Transformer；篇幅内缺少充分的人员划分信息，不能把图中近99%的结果视作严格 unseen-person 可复现基准 |
| Long kernel distillation，Yao et al., KBS 2025 [R7] | publisher 原始摘要明确多个长卷积核教师指导短卷积核 CNN 学生，带标签指导与自适应权重 | 不能宣称首次大感受野→小感受野 CNN。**长 kernel、长物理观察窗、高采样率是三个不同变量**；全文本轮未取得，不伪称已核验细节 |
| Abbaspourazad et al., 2024/2025 [R8] | 用可穿戴 PPG foundation representation 蒸馏加速度表示，目标含健康结局 | “可穿戴 foundation model + KD”不是新。与公开 1D CNN HAR 教师→极小 CNN、物理观测预算的研究设定不同 |
| Mardanpour et al., Information Sciences 2023 [R9] | 三轴教师→单轴学生，response+feature KD 与 Tucker 分解 | “减少轴／传感器用 KD”不是新；要研究可观测信息与跨人收益，而不是只把输入删掉 |
| Mao et al., Computers and Electronics in Agriculture 2023 [R10] | 高采样率教师→低采样率学生，重建与 correlation distillation；对象为动物可穿戴活动 | 直接否定“KD 恢复低采样信息首次”。人类动作、跨人、受控抗混叠处理与公开 pretrained teacher 仍有不同问题，但只换数据集不够 |

SMLDist 作者相关 GitHub 地址可访问 API；当前仓库 README 名称为 TASNeM-HAR，存在训练代码和 RealWorld 配置，没有模型权重。它与 2024 最终版每一项实验是否对应，尚未验证；不能承诺“git clone 后即可严格复现最终论文”。

SMLDist 的时间 MSE 与完整 Fourier L2 也提醒我们：不应仅把 FFT-MSE 加到 time-MSE 上就称作新的时频知识。Parseval 关系意味着某些设置下两个损失主要只是重加权；rFFT、频率选择和归一化会影响是否等价，需要数学核对。

## 4. 数据：具体从哪里开始，不把输入差异藏起来

### 首选一：PAMAP2 腕部连续数据，用于 100→30→低率的受控实验

UCI 官方数据说明 [R11] 明确：三枚 IMU 分别在 dominant-arm wrist、胸口和 dominant-side ankle；标称 100 Hz，有时间戳，含 NaN／丢样。协议 12 类，另有并非人人完成的可选活动，总体最多 18 类。作者 HARNet benchmark 的 PAMAP2 子集仅 8 类、8 人，不能与常用 12 类完整协议成绩混比。

建议先只取 wrist 的 ±16 g 加速度三轴，在每个人、每段连续记录内部构造真实 10 s 窗。按时间戳处理 NaN 和断段，不能先删缺失行再把不连续时间拼成等间隔序列。官方 HARNet 的旧 PAMAP parser 确实先删除 NaN 行后构窗；如果研究的是严格物理时间／采样预算，不应不加审查地照搬此细节。可复现“作者方法”与建立“本研究物理时间正确的管线”必须分开写。

PAMAP2 的优点是同一人多个位置同步、源采样率高，适合精确比较 30／15／7.5 Hz 等输入。缺点是人员少、类别缺失与丢包需要预先固定处理规则。不要按测试成绩选类别。

### 首选二：WISDM 2019 watch，用于更多人员的腕部复制

应使用 **WISDM Smartphone and Smartwatch Activity and Biometrics Dataset，UCI id507，2019** [R12]，不是早期只含手机活动的 WISDM。官方说明是 51 人、18 个活动、每个约3分钟，watch 置于优势手，phone 在口袋。raw/watch/accel 是腕部三轴连续数据，有 timestamp，单位 m/s²；名义频率 20 Hz，官方也明确操作系统实际轮询会有偏差。

HARNet 官方 parser 正是读取 `raw/watch/accel`，按时间重采样到 20 Hz，构造真实 10 s 窗，然后 /9.81 并插值到 30 Hz。这里 **20→30 是升采样，不创造 10 Hz 以上的新信息**；代码中的“downsample”注释不能改变这个事实。

这个数据的人员更多，适合测 label-person budget；但不能把 20→30 的模型接口适配当作高采样观测。若做低率实验，先在真实 20 Hz 序列上抗混叠并降到 10／5 Hz，再按预先说明的接口方式提供教师或学生，不让重采样混入标签信息。不要跨活动片段／长时间空洞构窗。

### 共同约束

先做 teacher 与 student 相同三轴加速度、相同真实窗长，再单独改变预算。现有 UCI／MotionSense 九通道实验是建立 CNN 学习规律的开发验证，不能直接与三轴 HARNet 实验的差异归因于架构或蒸馏方式。训练标准化和所有调参仅见源训练／验证人员；若用未标注目标人员进行 BN 更新或适配，要明确改成 transductive／UDA 设置。

## 5. 三个精确定义的问题与最小实验

### Q1：大规模预训练 prior 是否值得压缩进小 CNN？【优先】

**假说：** 相同标注人员与窗口预算下，冻结 HARNet 表征中至少一部分跨人可迁移结构，可以通过表征／时间关系监督保留在小 CNN 中；收益大小随学生容量而变，并非学生越小越好。

最低可解释矩阵：

- 先在腕部 10 s／30 Hz 三轴输入训练源任务 head，冻结 HARNet encoder，得到明确下游教师。源验证上若教师不能稳定胜过直接学生，停止“优质教师压缩”的前提，转为教师域适配诊断，不默认 foundation model 优越。
- 两个预定学生容量、两个预定标注人员预算；先小规模检查可行性，再按固定协议扩展。
- 同样输入、增强、优化器／epoch／选模规则：CE、LS、标准 logit KD、global feature KD、一个时间关系 KD。feature projector 只在训练时用；部署参数与内存单独计数。
- 比较相同 HARNet 教师下不同蒸馏方式，这是方法效果；另报告 task-only 学生到最佳蒸馏学生的差异，这是“可利用外部预训练资源”的实际效果。**后者包含 UKB 额外无标签数据，不是蒸馏损失的纯因果效应。**
- 若另引入可获得的额外无标签源数据，CE／学生自监督／KD 的数据访问预算须一致。无需为了形式上的数据等量去重训 UKB 大模型，但必须承认无法把数据量与预训练算法完全分离。

**Go：** 至少两个数据集或独立人员划分、多个配对种子，在相同容量与标签预算下，收益超过直接 CE／LS、对训练预算变化稳健，且确实缩小了与教师的差距；容量—性能关系具有可解释转折。

**No-go／降格：** 只有一组 validation 的小数点优势；结果由多读了未标注目标数据、不同输入通道、更多调参或更长训练解释；全局／时间表征蒸馏均不能超过简单 logit KD，而文章仍只写一个复杂新损失。后一种结果仍可成为毕业论文的清楚比较，但不自动形成强投稿贡献。

### Q2：CNN 的采样率与观察时间预算，分别限制了哪种可蒸馏知识？【最具体的扩展】

拆成两个实验，不能把总输入点数当作唯一预算：

1. **采样率轴：固定真实10 s。** 教师30 Hz；学生30／15／7.5 Hz。先低通再降采样，保留相同起止时间、标签和传感器位置。真实低率学生可直接处理较少点数；若上采样到教师长度作接口控制，必须另报它并没有同样减少CNN输入计算。
2. **上下文轴：固定采样率。** 教师观察同一预测时刻之前10 s，学生只观察最后5 s／2.5 s。训练标签对齐到同一预测事件，不能让长窗采用多数标签而短窗采用另一活动标签。优先用共同末端的 causal past windows，避免教师额外未来信息与学生未来不可见被混为一谈。

最小对照：学生独立 CE／LS、同可见输入训练的教师／同视图 KD、富信息教师的 logit KD、富信息教师的一个关系 KD。加入“教师也只看学生视图”的诊断，区别一般教师正则化与额外观察带来的知识。用于机制比较的教师必须预先冻结，不根据每个学生 test 挑选。

应测：按人和活动的差值、随预算收紧的失效点、周期性动作／静态姿态／过渡段的差异、学生与教师在相同可见片段上的分歧、clean 与预算收紧条件的双指标。关系特征对齐必须按实际秒数或时间 bin 的中心，不能把两个采样率的相同数组下标当作同一时刻。

**关键判断：蒸馏不能保证恢复被不可逆采样／裁剪删除的信息。** 如果两种活动在学生可见输入下本来就几乎不可区分，更自信地模仿教师反而可能传入不可学习的目标。可检验的问题是“哪些教师关系与学生保留的时域结构一致时有用”，而非声称恢复了原始高频信号。

最靠近的先例是 [R7] 长 kernel CNN 与 [R10] 动物低采样 KD。定向检索尚未核实人类 wearable HAR 中完全相同的“长真实观察窗→短 causal 窗”研究；这只是本轮检索边界，**不能据此宣称首次**。视频 Temporal-DINO 有过去／未来的教师学生先例，但不是传感器 HAR，不拿它作直接复现基线。

**Go：** 预算下降曲线出现可重复、活动相关的知识继承边界，富信息 KD 超过同可见信息 KD，且时间关系／可见性约束能解释或改善边界；跨人验证成立。

**No-go／转向：** 提升仅来自未经抗混叠的抽点造成伪差异、改变物理运动速度、不同标签构窗、或教师看到了对应测试人员。若降频本身比高频更好，应先解释噪声／优化变化，不假设教师信息总更多。

采集等待时间、滑窗更新步长、CPU forward 时间、内存、传感器采样功耗是不同指标。10 s→2.5 s 缩短启动所需观察时间，不自动等于稳定流式输出每次延迟降低7.5 s；能耗需设备测量，离线降采样不能直接证明硬件省电。

### Q3：时间关系蒸馏是否转移了教师的不变性，而非只复制分类输出？【次选】

先实测 frozen teacher 对姿态／缺失／采样变化是否更稳，只有前提成立才谈继承。比较同等增强下 CE／LS、logit KD、point-feature KD、temporal-relation KD；student consistency-only 是必要对照。训练与评估使用不同但预先固定的扰动强度／类型，并跨人员。

可用的候选表征是中间层时间 bin 间的归一化相似度矩阵，而非强迫不同通道基底一一相等。关系 KD 本身是已有技术；可研究的是：teacher 在多视图下稳定的时间关系，是否比全部关系更可迁移给低容量／低预算学生。这个筛选规则目前只是待检验假说，不能给出“首次”结论。

应区分整体三轴旋转、身体位置变化、缺失一个轴、缺失一个设备、连续丢帧。它们改变的信息不同。旋转也并非任何标签／任务都天然保持可辨性：静态姿态可能依赖重力方向；用任意不变性可能删除有效信号。SMLDist 与 HKDF 已有强相关设计，因此此方向若仅加一种 loss，投稿力度弱于 Q1+Q2 的可继承边界研究。

## 6. 本轮只读代码审查

### UCI 新 CNN 矩阵

已逐段阅读 `research/round5/cnn_experiments/run_cnn.py` 与锁定协议。未发现实质阻塞问题：train-only 缓存／人员索引核验、前80轮旧订单复现、同seed初始化配对、固定best80教师、detach/KL方向/batchmean/T²、80轮导出后恢复当前state续训至160均一致。best160包含前80轮候选，未重置优化器；best／last／last10并报。4人验证仍是开发证据，best160有更多选峰机会。

### MotionSense 独立数据集复制

已读 `research/round5/motionsense_experiments/run_motionsense.py` 与协议。先按人员排除封存 CSV，再逐 trial 构窗，标签按活动目录；9通道顺序显式重排，标准化只用训练窗；教师缓存只有训练数据，学生初值与160轮batch顺序配对，80／160保存路径正确。未发现污染当前结果的实质问题。

报告应保留两点：50 Hz 是标称值，CSV没有物理时间戳，实际间隔未被核验；当前脚本未来重启时直接复用已完成结果、会覆盖未完成目录，复现说明要先归档并保护原结果。此为重启／归档行为限制，不是要求中断当前固定单次运行。两份实验均不能从现有开发结果直接升级为跨数据集迁移结论。

## 7. 完整书目、官方 PDF 与代码

以下仅使用英文原始论文、作者仓库或官方数据文档。全文可得与仅摘要已区分。工具引用 ref 供根代理重新打开核验；若最终引用这些 ref，根代理需按工具规则自行打开来源。

### R1. 公开教师，正式期刊全文与代码实物均核验

Hang Yuan; Shing Chan; Andrew P. Creagh; Catherine Tong; Aidan Acquah; David A. Clifton; Aiden Doherty. **2024. Self-supervised learning for human activity recognition using 700,000 person-days of wearable data. npj Digital Medicine 7, 91.** DOI: 10.1038/s41746-024-01062-3.

- Article: https://www.nature.com/articles/s41746-024-01062-3
- Official PDF: https://www.nature.com/articles/s41746-024-01062-3.pdf
- Code: https://github.com/OxWearables/ssl-wearables
- Pinned hub: https://github.com/OxWearables/ssl-wearables/blob/150550ea5d41800229c95e36f88f5bf0d2e7cf04/hubconf.py
- Pinned backbone: https://github.com/OxWearables/ssl-wearables/blob/150550ea5d41800229c95e36f88f5bf0d2e7cf04/sslearning/models/accNet.py
- 10s weights: https://wearables-files.ndph.ox.ac.uk/files/ssl/mtl_best.mdl
- 5s weights: https://raw.githubusercontent.com/OxWearables/ssl-wearables/150550ea5d41800229c95e36f88f5bf0d2e7cf04/model_check_point/mtl_5_best.mdl
- Local: `papers/yuan2024_final.pdf`; Methods pp.6–7, Table2 p.3, Table4 p.7, Figure3 label/scale analysis.
- Web refs: `turn473287816search3`, `turn789158068search0`, `turn381082766view1`, `turn381082766view2`. Nature HTML 经网页工具遇IDP错误，但正式 PDF 已通过公开官方下载成功。

### R2. 新近大型 1D CNN，全文核验，无公开预训练权重

Maxwell A. Xu; Jaya Narain; Gregory Darnell; Haraldur Hallgrimsson; Hyewon Jeong; Darren Forde; Richard Fineman; Karthik J. Raghuram; James M. Rehg; Shirley Ren. **2025. RelCon: Relative Contrastive Learning for a Motion Foundation Model for Wearable Data. International Conference on Learning Representations (ICLR).**

- Official PDF: https://proceedings.iclr.cc/paper_files/paper/2025/file/83eb339ed42297658fa24b5cec939285-Paper-Conference.pdf
- OpenReview: https://openreview.net/pdf?id=k2uUeLCrQq
- Code/weight limitation: https://github.com/maxxu05/relcon
- Local: `papers/relcon2025.pdf`; §3–4, Tables3–5 and appendix architecture.
- Web refs: `turn381082766view3`, `turn772817936view1`, `turn339555881view2`.

### R3. 2026 强近邻，作者版重点章节阅读

Zongzhe Xu; Aakarsh Anand; Sarah Jiang; Chuntung Zhuang; Zitao Shuai; Sriram Sankararaman; Yuzhe Yang. **2026. Inertia-1: An Open Exploration of Wearable Motion Foundation Models. arXiv:2607.06617.** 作者仓库公告 NeurIPS2026 接收；本轮未核验正式 proceedings bibliographic entry。

- Author PDF: https://arxiv.org/pdf/2607.06617
- Code: https://github.com/yang-ai-lab/Inertia-1
- Weight availability discussion: https://github.com/yang-ai-lab/Inertia-1/issues/1
- Local: `papers/inertia2026.pdf`; methods／sampling／window／placement related sections重点阅读。
- Web refs: `turn582122845view2`, `turn972052856view0`, `turn972052856view1`.

### R4. SSL + 自训练／KD 的直接 HAR 先例

Chi Ian Tang; Ignacio Perez-Pozuelo; Dimitris Spathis; Soren Brage; Nick Wareham; Cecilia Mascolo. **2021. SelfHAR: Improving Human Activity Recognition through Self-training with Unlabeled Data. Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies 5(1), Article36, 1–30.** DOI: 10.1145/3448112.

- DOI: https://doi.org/10.1145/3448112
- Official author PDF: https://www.cl.cam.ac.uk/~cm542/papers/imwut-selfhar.pdf
- Code: https://github.com/iantangc/SelfHAR
- Local: `papers/selfhar2021.pdf`; Figure2 and §3.2.1–3.2.5实际阅读。作者稿页眉残留“January2020”模板，不改变2021正式刊载年份。
- Web refs: `turn680835983search0`, `turn680835983search1`, `turn673513621search47`.

### R5. 时频、多层与跨用户蒸馏的直接先例

Runze Chen; Haiyong Luo; Fang Zhao; Xuechun Meng; Zhiqing Xie; Yida Zhu. **2024. A Lightweight Deep Human Activity Recognition Algorithm Using Multiknowledge Distillation. IEEE Sensors Journal 24(19), 31495–31511.** DOI: 10.1109/JSEN.2024.3443308.

- Publisher: https://ieeexplore.ieee.org/document/10642991
- Author PDF: https://arxiv.org/pdf/2107.07331v5
- Related author code: https://github.com/chenrz925/SMLDist (current README identifies TASNeM-HAR; final-paper code equivalence unverified)
- Local: `papers/smldist2024.pdf`; §III.A and §IV.A/B, author PDF pp.4–8.
- Web refs: `turn691187192search0`, `turn972052856view2`.

### R6. 干净教师→损坏输入的直接先例，6页全文阅读

Di Hu; Xingchen Wu; Dezhu Xiong. **2025. Hierarchical Knowledge Distillation for Human Activity Recognition. Intelligent Manufacturing and Cloud Computing: Proceedings of the 1st International Conference (ICIMCC2024), Advances in Transdisciplinary Engineering 66, 179–184.** DOI: 10.3233/ATDE241354.

- Official article: https://journals.sagepub.com/doi/full/10.3233/ATDE241354
- Official PDF: https://journals.sagepub.com/doi/pdf/10.3233/ATDE241354
- Web refs/full text: `turn974518444view0`, `turn972052856view3`; no author code verified. 本地下载403，网页工具取得全文；未绕过访问控制。

### R7. 长 kernel 教师→短 kernel CNN 学生，仅 publisher 元数据／摘要核验

Minghui Yao; Dongzhou Cheng; Lei Zhang; Li Wang; Hao Wu; Aiguo Song. **2025. Long kernel distillation in human activity recognition. Knowledge-Based Systems 316, 113397.** DOI: 10.1016/j.knosys.2025.113397.

- Official: https://www.sciencedirect.com/science/article/pii/S0950705125004447
- Official DOI: https://doi.org/10.1016/j.knosys.2025.113397
- 本轮未核验公开可下载 PDF／代码，不填猜测地址。
- Web refs: `turn604761964search3`, `turn379569324search2`.

### R8. Wearable foundation model 蒸馏近邻

Salar Abbaspourazad; Anshuman Mishra; Joseph Futoma; Andrew C. Miller; Ian Shapiro. **2024, revised2025. Wearable Accelerometer Foundation Models for Health via Knowledge Distillation. arXiv:2412.11276.** Apple publication page dated February2025；本轮未核验正式会议刊载，不能标为ICLR。

- Author publication: https://machinelearning.apple.com/research/wearable-accelerometer-foundation-models
- Author PDF: https://arxiv.org/pdf/2412.11276
- Web refs: `turn990679318view0`, `turn339555881view0`, `turn221658283view0`.

### R9. 三轴→单轴 HAR，publisher 原始来源核验

Malihe Mardanpour; Majid Sepahvand; Fardin Abdali-Mohammadi; Mahya Nikouei; Homeyra Sarabi. **2023. Human activity recognition based on multiple inertial sensors through feature-based knowledge distillation paradigm. Information Sciences 640, 119073.** DOI: 10.1016/j.ins.2023.119073.

- Official: https://www.sciencedirect.com/science/article/pii/S0020025523006588
- Official DOI: https://doi.org/10.1016/j.ins.2023.119073
- 本轮无可核验开放PDF／代码；Web refs: `turn604761964search4`, `turn143879631search0`.

### R10. 高率→低率可穿戴 KD 的动物先例

Axiu Mao; Meilu Zhu; Endai Huang; Xi Yao; Kai Liu. **2023. A teacher-to-student information recovery method toward energy-efficient animal activity recognition at low sampling rates. Computers and Electronics in Agriculture 213, 108242.** DOI: 10.1016/j.compag.2023.108242.

- Official: https://www.sciencedirect.com/science/article/pii/S0168169923006300
- Author preprint listing: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4288774 (本轮直接打开失败；搜索能取得作者预印本信息)
- Journal mechanism/author refs: `turn604761964search2`, `turn484302498search15`, `turn484302498search16`, `turn143879631search7` (author institution metadata confirms volume213).
- 另已读同一作者组的相关研究 *Robust animal activity recognition using wearable sensors: A correlation distillation-based information recovery method toward data having low sampling rates*，作者为 A. X. Mao、E. D. Huang、M. L. Zhu、K. Liu；author-hosted PDF: https://plf.tennessee.edu/wp-content/uploads/sites/229/2024/01/Robust-animal-activity-recognition-using-wearable-sensors.-A-correlation-distillation-based-information-recovery-method-toward-data-having-low-sampling-rates.pdf 。本轮未独立核验该短稿会议刊名，因此不把它冒充上面五作者 journal 论文的正式PDF。
- 该短稿 Methods 明确 CNN 的高／低采样输入、上采样接口、时间／轴相关矩阵；实验讨论中25Hz好于100Hz。这说明“更高率必然更好”亦需实测。其全文没给出足够信息证明使用了本研究所需的抗混叠处理，不替作者补写。

### R11. PAMAP2 官方数据与采样／位置定义

Attila Reiss. **2012. PAMAP2 Physical Activity Monitoring [Dataset]. UCI Machine Learning Repository.** DOI: 10.24432/C5NW2H.

- Dataset: https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring
- Official downloadable readme PDF: https://archive.ics.uci.edu/ml/machine-learning-databases/00231/readme.pdf
- Dataset-associated paper: **Attila Reiss; Didier Stricker. 2012. Introducing a New Benchmarked Dataset for Activity Monitoring. 16th IEEE International Symposium on Wearable Computers (ISWC).** 该书目由官方 readme 列出，本轮不猜论文PDF。
- Web refs: `turn604761964search1`, `turn484302498view1`; official readme local `papers/pamap2_readme.pdf`.

### R12. WISDM 2019 官方腕部连续数据

Gary M. Weiss. **2019. WISDM Smartphone and Smartwatch Activity and Biometrics Dataset. UCI Machine Learning Repository, Dataset507; accompanying technical description.** Dataset DOI: 10.24432/C5HK59（已核对 UCI 官方 citation）。

- Dataset: https://archive.ics.uci.edu/dataset/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset
- Official PDF: https://archive.ics.uci.edu/ml/machine-learning-databases/00507/WISDM-dataset-description.pdf
- Official download: https://archive.ics.uci.edu/static/public/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset.zip
- Web refs: `turn604761964search0`, `turn484302498view0`, `turn143879631view0`; local `papers/wisdm2019_description.pdf`. 来源定位：§1 devices／subjects，§3 raw timing／units；不是旧WISDM手机数据。

## 8. 交付与尚未完成事项

完成：英文一手文献地图、关键方法章节核读、HARNet 权重实物及输入／层级验证、两个连续腕部数据源落实、两个正在运行实验脚本的只读实质审查。未做：真实 HARNet 下游训练、预训练模型效果复现、能耗测量、封存测试评估。未核验开放全文的论文明确标注摘要层级；未核验 checkpoint 的模型没有列为已就绪。

建议根报告把 Q1+Q2 组成一条后续路线：先证明公开预训练知识能压缩，再研究在更少采样／更短观察下哪些知识保留。Q3 是可解释这一边界的候选机制，而不是另铺一条泛泛的鲁棒性系统工程。
