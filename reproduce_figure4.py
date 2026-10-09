from pathlib import Path
import pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
base=Path(__file__).resolve().parent/'meta_acp/runs_gpu'
f=pd.read_csv(base/'few_shot_strict_analysis/few_shot_paired_comparisons.csv')
a=pd.read_csv(base/'ablation_esm2_150m_summary/ablation_paired_bootstrap.csv')
print(f.method.unique())
plt.rcParams.update({'font.family':'Arial','font.size':8,'axes.labelsize':8,'axes.titlesize':9,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'xtick.labelsize':7,'ytick.labelsize':7,'svg.fonttype':'none','pdf.fonttype':42})
fig,axs=plt.subplots(2,2,figsize=(7.2,5.8))
fig.subplots_adjust(left=.19,right=.98,bottom=.13,top=.85,hspace=.80,wspace=.32)
methods=['prototype','classifier','head'];colors=['#0072B2','#009E73','#D55E00'];markers=['o','s','^']
settings=[('Test1.fasta',5),('Test1.fasta',10),('Test2.fasta',5),('Test2.fasta',10)]
for col,metric in enumerate(['mcc','auc']):
 ax=axs[0,col]
 for j,(m,c,mk) in enumerate(zip(methods,colors,markers)):
  d=pd.concat([f[(f.source_file==s)&(f.k_shot==k)&(f.method==m)&(f.metric==metric)] for s,k in settings]);assert len(d)==4
  x=d.mean_difference.to_numpy()*100;lo=d.difference_ci_low.to_numpy()*100;hi=d.difference_ci_high.to_numpy()*100
  ax.errorbar(x,np.arange(4)+(j-1)*.20,xerr=[x-lo,hi-x],fmt=mk,color=c,markersize=3.5,lw=.9,capsize=2)
 ax.set_yticks(range(4),[s.replace('.fasta','')+' / '+str(k) for s,k in settings] if col==0 else ['']*4);ax.invert_yaxis()
 ax.set_xlabel('Δ'+metric.upper()+' (× 0.01)');ax.set_title(('a' if col==0 else 'b')+'   Few-shot adaptation',loc='left',weight='bold',pad=13)
 if col==0:ax.set_ylabel('Source / examples per class')
for col,metric in enumerate(['auc','ap']):
 ax=axs[1,col];d=a[a.metric==metric].set_index('model_b').loc[['no_supcon','no_proto','classifier_only']]
 x=d.difference_a_minus_b.to_numpy()*100;lo=d.difference_ci_low.to_numpy()*100;hi=d.difference_ci_high.to_numpy()*100
 ax.errorbar(x,np.arange(3),xerr=[x-lo,hi-x],fmt='D',color='#0072B2',markersize=4,lw=1,capsize=2)
 ax.set_yticks(range(3),['Without contrastive','Without prototype','Focal only'] if col==0 else ['']*3);ax.invert_yaxis()
 ax.set_xlabel('Δ'+('AUC' if col==0 else 'AP')+' (× 0.01)');ax.set_title(('c' if col==0 else 'd')+'   Auxiliary loss ablation',loc='left',weight='bold',pad=13)
for ax in axs.flat:
 ax.axvline(0,color='#8B959C',ls='--',lw=.8);ax.grid(axis='x',color='#E4E9ED',lw=.5);ax.set_axisbelow(True)
 for y in range(len(ax.get_yticks())):
  if y%2==0:ax.axhspan(y-.45,y+.45,color='#F1F5F7',zorder=-2)
handles=[Line2D([],[],color=c,marker=m,lw=1,markersize=4,label=l) for c,m,l in zip(colors,markers,['Prototype','Classifier','Projection + classifier'])]
fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.57,.995),frameon=False,ncol=3,fontsize=8)
fig.text(.19,.927,'Adapted − baseline | 20 paired draws per setting',fontsize=7,color='#697780')
fig.text(.19,.46,'Full objective − ablation | Three-seed ensembles',fontsize=7,color='#697780')
fig.text(.19,.02,'Mean differences with pointwise 95% bootstrap CIs',fontsize=7,color='#697780')
p=Path(__file__).resolve().parent/'reproduced/figures';p.mkdir(parents=True,exist_ok=True)
fig.savefig(p/'Figure_4.png',dpi=450);fig.savefig(p/'Figure_4.svg');fig.savefig(p/'Figure_4.pdf');plt.close(fig)

