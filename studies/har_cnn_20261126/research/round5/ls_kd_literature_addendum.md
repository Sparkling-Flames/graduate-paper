# LS 教师与 KD：本轮窄问题补遗

核验日期：2026-11-26。没有新增训练；仅复核原始论文、作者页面和已有本地 PDF。

## 核心结论

**“LS 教师能够改善学生”已经有明确的已发表先例，不能作为本项目的新方法或首次发现。** 当前 UCI HAR 1D CNN 中，LS 教师的 w8 学生在本次4人开发划分、3个配对种子下，相对直接LS的 best160 平均提升约0.81个百分点，适合表述为一种已有机制在当前具体设置中的经验结果。它没有推翻普遍不相容定理，因为2019论文没有证明那样的定理；2021和2022论文已经明确给出有条件的正面结果。

## 1. 2019：负面证据的原始来源

**Rafael Müller; Simon Kornblith; Geoffrey E. Hinton. 2019. When Does Label Smoothing Help? Advances in Neural Information Processing Systems 32 (NeurIPS2019).**

- 官方页面：https://proceedings.neurips.cc/paper_files/paper/2019/hash/f1748d6b0fd9d439f71450117eba2725-Abstract.html
- 官方PDF：https://proceedings.neurips.cc/paper_files/paper/2019/file/f1748d6b0fd9d439f71450117eba2725-Paper.pdf
- 定位：§4，Figure5，PDF pp.6–7；§6结论。
- 可核查结论：在作者的图像蒸馏实验中，LS即使提高教师准确率，也可能降低其学生效果；作者用更紧密的类内表征、logit中类别相似性信息减少解释这一现象。Figure5为CIFAR-10上ResNet-56→AlexNet。**这是条件化经验结果，不应转述成“所有LS教师都不适合KD”。**
- 本轮primary refs：`turn583891202search2`, `turn583891202search6`, `turn583891202search8`。本地全文 `research/round3/literature/muller2019.pdf` 已复核。

## 2. 2021：直接正面反例，不只是 arXiv 预印本

**Zhiqiang Shen; Zechun Liu; Dejia Xu; Zitian Chen; Kwang-Ting Cheng; Marios Savvides. 2021. Is Label Smoothing Truly Incompatible with Knowledge Distillation: An Empirical Study. International Conference on Learning Representations (ICLR2021).**

- 正确官方页面：https://openreview.net/forum?id=PObuuGVrGaZ
- 正确官方PDF：https://openreview.net/pdf?id=PObuuGVrGaZ
- 可下载作者版：https://arxiv.org/pdf/2104.00676
- 作者项目：https://zhiqiangshen.com/projects/LS_and_KD/index.html
- 定位：§4–5，Table2（PDF p.7），AppendixE。
- 可核查结论：LS教师在多种图像／翻译设置下可训练出更好学生；作者也指出长尾分布与类别数增大时的限制。Table2的ImageNet ResNet-50→ResNet-18学生Top-1由71.425±0.038提升至71.816±0.017，直接否定“LS教师不能改善KD”的普遍说法。主体图像实验多为T=1、纯教师软监督，不能与当前T=2、硬标签+KD混合目标视作同协议。
- LS定义注意：其公式真类1−α、其他类α/(K−1)，不同于PyTorch uniform-inclusive `(1−ε)y+ε/K`；不能只看“0.1”就说同一平滑程度。
- 本轮primary refs：`turn253508592view0`, `turn837755041view0`, `turn253508592search2`。官方项目页的Paper链接已通过工具点击核实为PObuuGVrGaZ。**旧round4笔记中的Hyt_7MbpkOu地址错误，已更正。** 本次OpenReview有浏览器验证墙，正文核验使用公开arXiv作者稿，不把验证页当论文。

## 3. 2022：最直接的温度交互先例

**Keshigeyan Chandrasegaran; Ngoc-Trung Tran; Yunqing Zhao; Ngai-Man Cheung. 2022. Revisiting Label Smoothing and Knowledge Distillation Compatibility: What was Missing? Proceedings of the 39th International Conference on Machine Learning, PMLR162:2890–2916.**

- 官方页面：https://proceedings.mlr.press/v162/chandrasegaran22a.html
- 官方PDF：https://proceedings.mlr.press/v162/chandrasegaran22a/chandrasegaran22a.pdf
- 作者代码：https://github.com/sutd-visual-computing-group/LS-KD-compatibility
- 定位：§3–5、Table2/3、§7；正文明确给出LS教师配低温度T=1的实践建议。
- 可核查结论：**教师训练时的LS与蒸馏温度之间存在交互。** 作者将较高温度下学生表征朝语义相近类别发生的变化称为systematic diffusion，并以此解释LS教师收益为什么可能被高温度削弱。其工作已经直接研究并调和2019与2021结论；“发现LS与温度交互”不能算本项目新意。
- 本轮primary refs：`turn683082152view2`, `turn583891202search7`, `turn583891202search5`。本地全文 `research/round4/papers/chandrasegaran2022.pdf` 已复核相关方法／表格。
- 本轮没有在所核正文中发现作者做了“把CE教师单标量缩放到LS教师训练目标平均熵”的同一精确对照；这不构成对该对照的新颖性证明，也没有必要把它包装成新方法。

## 4. 更早的已发表正面反例：有噪声标签时

**Michal Lukasik; Srinadh Bhojanapalli; Aditya Krishna Menon; Sanjiv Kumar. 2020. Does Label Smoothing Mitigate Label Noise? Proceedings of the 37th International Conference on Machine Learning, PMLR119:6448–6458.**

- 官方页面：https://proceedings.mlr.press/v119/lukasik20a.html
- 官方PDF：https://proceedings.mlr.press/v119/lukasik20a/lukasik20a.pdf
- 原始摘要明确报告：从带噪标签数据蒸馏时，平滑教师有利于学生；作者将它与此前干净标签设置的结果区分。
- 这是最直接的已发表反例之一，但**不能据此声称当前UCI改善已被证明来自标签噪声**。当前实验没有把噪声机制作为已辨识的因果变量。
- 本轮核验范围：官方书目、摘要和PDF首页，未声称重新阅读全文。Primary refs：`turn713735337search0`, `turn713735337search2`。

## 5. 当前最合适的后续诊断：只改变教师目标的标量软度

这是**已有LS/KD温度文献启发的诊断计划，本轮未执行，不是新方法**。

在既有CE教师训练logits上求一个正标量a，使源训练窗上的平均熵满足：

`mean H(softmax(a * z_CE / 2)) ≈ mean H(softmax(z_LS / 2))`。

学生温度始终固定为2，蒸馏权重固定0.5，T²系数固定4，使用相同学生初始化／batch顺序／优化预算。这样改变的是教师目标分布；不要同时改变学生温度和损失倍数，否则标准T扫描还会改变学生梯度的条件，难以孤立目标平滑的作用。该标量只利用源训练logits，不读封存人员，不通过验证性能反复挑选。

该诊断区分的问题：

- 若标量软化后的CE教师接近LS教师的学生效果，说明单纯的logit尺度／目标软度是有解释力的候选因素，但不是证明所有信息关系都无用。
- 若仍明显落后，说明**匹配平均熵不足以复现LS教师作用**；可以进一步怀疑每样本软度分布、非目标类别关系或错误结构，但不能仅凭这个差值宣布其中某一个已被证明。
- 正标量保留CE教师每个样本的类别排名，不能复制LS训练改变的类别关系。平均熵相同也不等于逐样本熵、真类概率、类条件分布或梯度范数相同。这些边界应写进报告，不把“熵匹配”称为全部因素已控制。

当前两个CNN教师的验证准确率相同，LS教师硬标签NLL更高；故不能把学生提升解释为“教师更准确／校准更好”。本次结果支持的是**更换教师训练目标所产生的输出分布整体干预改善了当前学生**；机制仍待上述对照。

## 可直接纳入报告的措辞

> 本次UCI HAR开发实验中，采用标签平滑训练的教师，使w8学生相对直接标签平滑获得约0.81个百分点的平均macro-F1提升。该结果与既有研究所揭示的LS教师在特定条件下有益相符，不能视为首次发现或新蒸馏方法。Müller等（2019）报告了LS教师在其设置中的负面影响，Shen等（2021）给出正面反例，Chandrasegaran等（2022）进一步指出教师平滑与蒸馏温度存在交互。因此，后续将采用文献启发的目标软度诊断：只对CE教师logits进行单标量缩放，使训练目标平均熵接近LS教师，同时固定学生温度、蒸馏权重和训练预算，以检验仅改变输出尺度能否解释当前收益。平均熵匹配仍不能控制全部类别关系；现阶段不将收益归因于教师准确率或某一单独机制。
