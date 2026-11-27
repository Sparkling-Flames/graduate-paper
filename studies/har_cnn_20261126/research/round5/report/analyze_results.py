#!/usr/bin/env python3
"""Independent summary from completed development runs; no test data access."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[3]
R5=ROOT/'research'/'round5'
OUT=ROOT/'output'/'har_cnn_20261126'
OUT.mkdir(parents=True,exist_ok=True)

def writej(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def main():
    assert (R5/'cnn_experiments'/'completion.json').exists(),'UCI matrix is still running'
    assert (R5/'motionsense_experiments'/'completion.json').exists(),'MotionSense matrix is still running'
    u=pd.read_csv(R5/'cnn_experiments'/'summary_checkpoints.csv')
    u=u[u.role=='student']
    m=pd.read_csv(R5/'motionsense_experiments'/'summary_runs.csv')
    m=m[m.width==8].copy()
    m['budget']=m.snapshot.str.extract(r'(\d+)').astype(int)
    m['selection']=m.snapshot.str.replace(r'\d+','',regex=True)
    tables={};deltas={}
    for name,df in [('uci',u),('motion',m)]:
        df.to_csv(OUT/f'{name}_student_metrics.csv',index=False)
        summary=df.groupby(['width','method','budget','selection']).val_macro_f1.agg(['size','mean','std']).reset_index()
        summary.to_csv(OUT/f'{name}_aggregates.csv',index=False)
        tables[name]=summary.to_dict('records')
        pairs=[]
        kd='kd_ce' if name=='uci' else 'kd'
        for (width,budget,sel),g in df.groupby(['width','budget','selection']):
            pivot=g.pivot(index='seed',columns='method',values='val_macro_f1')
            for a,b in [(kd,'ce'),(kd,'ls'),('ls','ce')]+([('kd_ls','ce'),('kd_ls','ls'),('kd_ls','kd_ce')] if name=='uci' else []):
                if a not in pivot or b not in pivot:continue
                d=(pivot[a]-pivot[b]).dropna()*100
                pairs.append(dict(width=int(width),budget=int(budget),selection=sel,contrast=a+'-'+b,
                    n=len(d),mean_pp=float(d.mean()),sd_pp=float(d.std(ddof=1)),min_pp=float(d.min()),
                    max_pp=float(d.max()),positive=int((d>0).sum()),by_seed={str(k):float(v) for k,v in d.items()}))
        deltas[name]=pairs
        pd.DataFrame(pairs).to_csv(OUT/f'{name}_paired_summary.csv',index=False)
    writej(OUT/'analysis_summary.json',dict(aggregate=tables,paired=deltas))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':190})
    palette={'ce':'#526778','ls':'#B95D18','kd_ce':'#247B73','kd':'#247B73','kd_ls':'#735599'}
    labels={'ce':'CE','ls':'LS','kd_ce':'KD (CE teacher)','kd':'KD (CE teacher)','kd_ls':'KD (LS teacher)'}
    fig,axes=plt.subplots(1,2,figsize=(9.2,3.7),sharey=True)
    for ax,budget in zip(axes,[80,160]):
        for method in ['ce','ls','kd_ce']:
            d=u[(u.budget==budget)&(u.selection=='best')&(u.method==method)]
            agg=d.groupby('width').val_macro_f1.agg(['mean','std'])
            xs=np.arange(len(agg))
            ax.errorbar(xs,agg['mean']*100,yerr=agg['std']*100,color=palette[method],marker='o',capsize=3,label=labels[method],linewidth=1.6)
        ax.set_xticks([0,1,2],['w4\n406 params','w8\n1,126 params','w16\n3,526 params'])
        ax.set_title(f'Best checkpoint within {budget} epochs')
        ax.grid(axis='y',alpha=.2);ax.set_xlabel('Student width / parameter count')
    axes[0].set_ylabel('Validation macro-F1 (%)')
    axes[0].legend(loc='best',fontsize=8)
    fig.suptitle('UCI HAR: fixed subject split, three paired seeds',fontsize=12)
    fig.tight_layout();fig.savefig(OUT/'uci_capacity_budget.png');fig.savefig(OUT/'uci_capacity_budget.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9.2,3.7),sharey=True)
    for ax,budget in zip(axes,[80,160]):
        d=m[(m.budget==budget)&(m.selection=='best')]
        for seed,g in d.groupby('seed'):
            v=g.set_index('method').loc[['ce','ls','kd'],'val_macro_f1'].to_numpy()*100
            ax.plot([0,1,2],v,marker='o',label=str(seed),linewidth=1.2,alpha=.85)
        ax.set_xticks([0,1,2],['CE','LS','KD'])
        ax.set_title(f'Best checkpoint within {budget} epochs');ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Validation macro-F1 (%)');axes[1].legend(title='Paired seed',fontsize=8)
    fig.suptitle('MotionSense: independent training, w8 student (1,126 params)',fontsize=12)
    fig.tight_layout();fig.savefig(OUT/'motionsense_paired_seeds.png');fig.savefig(OUT/'motionsense_paired_seeds.pdf');plt.close(fig)
    print(json.dumps(dict(aggregate=tables,paired=deltas),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
