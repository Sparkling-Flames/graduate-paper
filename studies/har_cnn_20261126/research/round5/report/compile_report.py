#!/usr/bin/env python3
"""Fill the research report from the completed experiment summaries."""
from pathlib import Path
import json
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'output/har_cnn_20261126'
HERE=Path(__file__).resolve().parent

def main():
    summary=json.loads((OUT/'analysis_summary.json').read_text())
    def metric(dataset,w,m,b,s='best'):
        return next(x for x in summary['aggregate'][dataset] if x['width']==w and x['method']==m and x['budget']==b and x['selection']==s)
    def contrast(dataset,w,c,b,s='best'):
        return next(x for x in summary['paired'][dataset] if x['width']==w and x['contrast']==c and x['budget']==b and x['selection']==s)
    def score(d):return f"{d['mean']*100:.2f} ± {d['std']*100:.2f}"
    def signed(v):return f'{v:+.3f}'
    def mdtable(headers,rows):
        return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows])
    rows=[]
    for w,p in [(4,406),(8,1126),(16,3526)]:
        for b in [80,160]:
            rows.append([f'w{w} / {p:,}',b]+[score(metric('uci',w,m,b)) for m in ['ce','ls','kd_ce']]+[
                signed(contrast('uci',w,'kd_ce-ce',b)['mean_pp']),signed(contrast('uci',w,'kd_ce-ls',b)['mean_pp'])])
    utable='下表的 KD 均使用原 CE 教师；LS 指学生直接使用标签平滑训练。差值单位为百分点。\n\n'+mdtable(['宽度 / 参数','预算','CE','LS','KD','KD-CE pp','KD-LS pp'],rows)
    rows=[]
    for b in [80,160]:
        rows.append([b]+[score(metric('motion',8,m,b)) for m in ['ce','ls','kd']]+[
            signed(contrast('motion',8,'kd-ce',b)['mean_pp']),signed(contrast('motion',8,'kd-ls',b)['mean_pp'])])
    mtable=mdtable(['预算','CE','LS','KD','KD-CE pp','KD-LS pp'],rows)
    r=[]
    for b in [80,160]:
        r.append([b,score(metric('uci',8,'kd_ce',b)),score(metric('uci',8,'kd_ls',b)),
            signed(contrast('uci',8,'kd_ls-kd_ce',b)['mean_pp']),signed(contrast('uci',8,'kd_ls-ls',b)['mean_pp'])])
    teacher_table=mdtable(['预算','KD 用 CE 教师','KD 用 LS 教师','换教师差值 pp','LS教师KD-直接LS pp'],r)
    teacher_table+='\n\n两位 CNN 教师的验证准确率同为 93.7931%，macro-F1 仅由 94.0825% 变为 94.0961%；LS 教师的硬标签 NLL 反而较高。换教师的诊断必须解释为整条训练及输出目标改变后的效果，不能自动归因于教师更准确。此诊断不改变论文主线，也不证明 MLP 中的机制已经推广到 CNN。'
    teacher_table+='\n\n这个改善并非只出现在最佳检查点：160 轮的最后一轮，LS 教师 KD 对直接 LS 平均提高 '+signed(contrast('uci',8,'kd_ls-ls',160,'last')['mean_pp'])+' 个百分点；末 10 轮平均提高 '+signed(contrast('uci',8,'kd_ls-ls',160,'last10_mean')['mean_pp'])+' 个百分点，均为三个种子同方向。逐类核查也限制了机制解释：相对 CE 教师 KD，三次验证预测中非坐站互混错误从 61 降到 13，坐站互混却从 252 增到 257；改善主要避免了其他类别的错误，没有解决静态姿态瓶颈。'
    uci_interpret='在当前固定划分和预定超参数下，使用原 CE 教师的普通 KD 在三档容量的 best160 平均分均低于各自 CE，也低于 LS。w4、w8、w16 的 KD-CE 分别为 '+', '.join(signed(contrast('uci',w,'kd_ce-ce',160)['mean_pp']) for w in [4,8,16])+' 个百分点。这直接反驳了“把学生缩小，KD 就会稳定更有利”的简单判断。'
    uci_interpret+='\n\n原 w8 首个种子在 80 轮时，CE 教师 KD 相对直接 LS 高 +0.6456 个百分点，延至 160 轮后差值为 -0.0515；三个种子的总体结论也没有支持原单次优势。与此同时，406 参数的 CE/LS 已达到约 93.65%，扩大通道并未带来稳定单调收益。这个现象支持控制模型容量和训练，而不是继续无目的增宽。'
    uci_interpret+='\n\n也需要保留最后一轮的不同结论：w16 在 last160 时，原 CE 教师 KD 相对 CE 为 '+signed(contrast('uci',16,'kd_ce-ce',160,'last')['mean_pp'])+' 个百分点，相对 LS 为 '+signed(contrast('uci',16,'kd_ce-ls',160,'last')['mean_pp'])+' 个百分点。默认 KD 在部分设置下缓和了后期退化，但没有提高按相同选择规则得到的最佳验证性能；不能把两类指标混为一谈。'
    uci_interpret+='\n\n教师训练目标的诊断给出了正面结果：使用 LS 教师后，w8 学生的 best160 为 '+score(metric('uci',8,'kd_ls',160))+'%，相对同宽直接 LS 为 '+signed(contrast('uci',8,'kd_ls-ls',160)['mean_pp'])+' 个百分点，三种子均为正；相对原 CE 教师 KD 为 '+signed(contrast('uci',8,'kd_ls-kd_ce',160)['mean_pp'])+' 个百分点。1,126 参数的该学生与 3,526 参数直接 LS 学生的平均分接近，值得进一步独立验证。'
    uci_interpret+='\n\n这些结果要求把教师的训练方式列为明确实验因素，而不只是比较教师准确率。它们没有检验更长时间教师、时序中层蒸馏或预训练教师；也不说明一般 KD 无效。毕业论文可以连贯呈现同一 CNN 结构下的失败条件、诊断与改善，再用独立数据和最终人员评价检验结论。'
    motion_interpret='MotionSense 的 best160 均值中，KD 相对 CE 提高 '+signed(contrast('motion',8,'kd-ce',160)['mean_pp'])+' 个百分点，相对 LS 提高 '+signed(contrast('motion',8,'kd-ls',160)['mean_pp'])+' 个百分点。KD-CE 的三种子方向是 2 正 1 负；KD-LS 三种子均为正，但只有约 0.142-0.188 个百分点。这里存在值得跟进的小信号，尚不能把 CE 到 KD 的全部收益解释为教师独有知识，因为直接 LS 已获得相当部分的改善。'
    motion_interpret+='\n\n训练预算影响明显：best80 的 KD-LS 均值为 '+signed(contrast('motion',8,'kd-ls',80)['mean_pp'])+' 个百分点，best160 变为正。固定最后一轮时，160 轮 KD-LS 的均值仅 '+signed(contrast('motion',8,'kd-ls',160,'last')['mean_pp'])+' 个百分点，三个种子差值跨越正负。不能把“所有种子最佳检查点略胜 LS”误写成在全部训练时点都稳定领先。'
    motion_interpret+='\n\n两套数据说明了同一 CNN 配方的表现会依赖数据与训练条件。MotionSense 没有重新调 T、KD 权重或增强来追求优胜；其完整负结果和种子波动同时保留。当前结论是开发阶段的比较，仍需在冻结方案后做独立人员评价。'
    headline='本轮新增 41 次训练后，原 CE 教师的普通 logit KD 没有在 UCI 三档 CNN 容量上形成平均优势；更换为 LS 教师后，1,126 参数学生达到 '+score(metric('uci',8,'kd_ls',160))+'%，相对同宽直接 LS 的均值优势为 '+signed(contrast('uci',8,'kd_ls-ls',160)['mean_pp'])+' 个百分点。独立 MotionSense 实验中，原 CE 教师 KD 相对 CE 提高 '+signed(contrast('motion',8,'kd-ce',160)['mean_pp'])+' 个百分点，但相对 LS 只增加 '+signed(contrast('motion',8,'kd-ls',160)['mean_pp'])+' 个百分点。证据支持继续研究 CNN 蒸馏，同时明确控制教师、强基线、容量与训练预算。'
    values={'HEADLINE_RESULTS':headline,'UCI_TABLE':utable,'MOTION_TABLE':mtable,
        'UCI_INTERPRETATION':uci_interpret,'MOTION_INTERPRETATION':motion_interpret,'UCI_TEACHER_TABLE':teacher_table,
        'UCI_FIGURE':'![图 1 同一固定人员划分上的容量与预算比较。误差条为三个配对种子的样本标准差；不能视作人群置信区间。](uci_capacity_budget.png)',
        'MOTION_FIGURE':'![图 2 MotionSense 每个随机种子的 CE、LS、KD 结果。连线只连接相同初始化和批次顺序的训练；两面板保持教师不变。](motionsense_paired_seeds.png)',
        'BIBLIOGRAPHY':(HERE/'references.md').read_text()}
    report=(HERE/'report_draft.md').read_text()
    for key,value in values.items():report=report.replace('{{'+key+'}}',value)
    assert '{{' not in report
    report=report.replace('普通 logit KD | 0.5','普通 logit KD [21] | 0.5')
    report=report.replace('更近的运动预训练工作也已核查：','SelfHAR 早已将无标签数据、自训练和时序增强用于 HAR，因此“自监督与蒸馏结合”也需要进一步明确差异。[19]\n\n更近的运动预训练工作也已核查：')
    path=OUT/'HAR_1D_CNN_研究进展_20261126.md';path.write_text(report)
    print(path)

if __name__=='__main__':main()
