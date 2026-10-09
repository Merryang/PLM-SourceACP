"""Regenerate quantitative manuscript figures from frozen data."""
from pathlib import Path
import argparse,sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parent;RUN=ROOT/'meta_acp/runs_gpu'
sys.path.insert(0,str(ROOT/'meta_acp'))
from metrics import binary_metrics
plt.rcParams.update({'font.family':'Arial','font.size':9,'axes.labelsize':9,'axes.titlesize':10,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'xtick.labelsize':8,'ytick.labelsize':8,'svg.fonttype':'none','pdf.fonttype':42})
BLUE='#0072B2';GREEN='#009E73';ORANGE='#D55E00';GRAY='#727C81'
def read(n):return pd.read_csv(RUN/n)
def save(fig,out,name):
 for ext in ['png','pdf','svg','tiff']:fig.savefig(out/(name+'.'+ext),dpi=450,facecolor='white')
 plt.close(fig)
def stripes(ax,n):
 for i in range(n):
  if i%2==0:ax.axhspan(i-.45,i+.45,color='#F0F4F6',zorder=-2)
 ax.grid(axis='x',color='#E1E7EA',lw=.5);ax.set_axisbelow(True)
def figure3(out):
 fig,axs=plt.subplots(2,2,figsize=(7.5,6.3));fig.subplots_adjust(left=.18,right=.98,bottom=.14,top=.84,wspace=.63,hspace=.88)
 back=['esm2_150m','esm2_650m','esmc','ankh','protbert','prott5'];labels=['ESM2-150M','ESM2-650M','ESM-C 300M','Ankh-base','ProtBERT','ProtT5']
 ax=axs[0,0]
 for i,b in enumerate(back):
  vals=[]
  for j,(seed,m) in enumerate(zip([42,2024,2026],['o','s','^'])):
   df=read(f'ensemble_with_hardcase_strict/member_predictions/test/{b}_seed{seed}.csv');v=binary_metrics(df.binary_label,df.score)['auc'];vals.append(v);ax.scatter(v,i+(j-1)*.17,marker=m,c=BLUE,s=23,edgecolor='white',lw=.4,zorder=3)
  ax.plot([min(vals),max(vals)],[i,i],c='#A2C6D7',lw=1.5);ax.plot(np.mean(vals),i,marker='|',ms=15,mew=2,c='#243D49')
 ax.set_yticks(range(6),labels);ax.invert_yaxis();ax.set_xlim(.92,.98);ax.set_xlabel('Integrated-test AUC');ax.set_title('a   Backbone comparison',loc='left',weight='bold',pad=15);stripes(ax,6)
 src=read('frozen_manuscript_tables/source_wise_mean_std.csv');ax=axs[0,1];sources=['Train.fasta','Test1.fasta','Test2.fasta']
 for i,s in enumerate(sources):
  vals=[float(src[(src.source_file==s)&(src.model==model)].auc_mean.iloc[0]) for model in ['esm2_150m','strict_logistic_stack_C3']]
  ax.plot(vals,[i,i],c='#B7C5CB',lw=1.5)
  for v,m,c in zip(vals,['o','D'],[BLUE,ORANGE]):ax.scatter(v,i,marker=m,c=c,s=34,zorder=3)
 ax.set_yticks(range(3),['Train (366)','Test1 (102)','Test2 (41)']);ax.invert_yaxis();ax.set_xlim(.88,1);ax.set_xlabel('Source-wise AUC');ax.set_title('b   Sources represented in training',loc='left',weight='bold',pad=15);stripes(ax,3)
 ax.legend(handles=[Line2D([],[],color=c,marker=m,ls='',label=l) for c,m,l in [(BLUE,'o','ESM2-150M mean'),(ORANGE,'D','Primary stack')]],loc='upper left',bbox_to_anchor=(0,-.28),frameon=False,fontsize=7)
 loso=read('loso_summary_esm2_150m/loso_seed_metrics.csv');ax=axs[1,0]
 for i,s in enumerate(sources):
  vals=[]
  for j,(seed,m) in enumerate(zip([42,2024,2026],['o','s','^'])):
   v=float(loso[(loso.heldout_source==s)&(loso.seed==seed)].auc.iloc[0]);vals.append(v);ax.scatter(v,i+(j-1)*.17,marker=m,c=GREEN,s=24,edgecolor='white',lw=.4,zorder=3)
  ax.plot([min(vals),max(vals)],[i,i],c='#A6CFC3',lw=1.5);ax.plot(np.mean(vals),i,marker='|',ms=15,mew=2,c='#243D49')
 ax.set_yticks(range(3),['Train (1,861)','Test1 (482)','Test2 (227)']);ax.invert_yaxis();ax.set_xlim(.87,1);ax.set_xlabel('Held-out-source AUC');ax.set_title('c   Sources excluded from training',loc='left',weight='bold',pad=15);stripes(ax,3)
 ci=read('external_public_summary/external_full_head_ensemble_bootstrap_ci.csv');cmp=read('external_public_summary/external_full_head_vs_linear_probe.csv');ax=axs[1,1]
 for i,t in enumerate(['antiangiopred','pepbenchmark_cpp']):
  r=ci[(ci.task==t)&(ci.metric=='auc')].iloc[0];probe=cmp[(cmp.task==t)&(cmp.metric=='auc')].iloc[0].estimate_b
  ax.errorbar(r.estimate,i,xerr=[[r.estimate-r.ci_low],[r.ci_high-r.estimate]],fmt='o',c=GREEN,lw=1.4,capsize=2,ms=5);ax.scatter(probe,i,c=GRAY,marker='s',s=28,zorder=4)
 ax.set_yticks(range(2),['AntiAngioPred\n(n = 56)','CPP\n(n = 229)']);ax.set_ylim(1.5,-.5);ax.set_xlim(.6,1);ax.set_xlabel('Related-task AUC');ax.set_title('d   Representation reuse',loc='left',weight='bold',pad=15);stripes(ax,2)
 ax.legend(handles=[Line2D([],[],c=GRAY,marker='s',ls='',label='Linear probe'),Line2D([],[],c=GREEN,marker='o',label='Task head (95% CI)')],loc='upper left',bbox_to_anchor=(0,-.29),frameon=False,fontsize=7)
 fig.legend(handles=[Line2D([],[],c=GRAY,marker=m,ls='',label=str(s)) for m,s in zip(['o','s','^'],[42,2024,2026])]+[Line2D([],[],c='#243D49',marker='|',mew=2,ms=12,ls='',label='Mean')],loc='upper center',ncol=4,frameon=False,title='Training seeds in a and c')
 save(fig,out,'Figure_3')
def figure5(out):
 fig,axs=plt.subplots(2,2,figsize=(7.5,6.4));fig.subplots_adjust(left=.20,right=.98,bottom=.12,top=.92,wspace=.58,hspace=.75)
 features=['net_charge_per_residue','isoelectric_point','hydrophobic_moment','length','gravy','hydrophobic_fraction'];labels=['Charge / residue','Theoretical pI','Hydrophobic moment','Length','GRAVY','Hydrophobic fraction']
 cls=read('physchem_provenance/class_comparison.csv').set_index('feature').loc[features];ax=axs[0,0]
 ax.errorbar(cls.cliffs_delta,np.arange(6),xerr=[cls.cliffs_delta-cls.delta_ci_low,cls.delta_ci_high-cls.cliffs_delta],fmt='o',c=BLUE,capsize=2,ms=4,lw=1)
 ax.set_yticks(range(6),labels);ax.invert_yaxis();ax.set_xlim(-1,1);ax.axvline(0,c=GRAY,ls='--',lw=.8);ax.set_xlabel("Cliff's δ (ACP versus non-ACP)");ax.set_title('a   Class-associated properties',loc='left',weight='bold',pad=15);stripes(ax,6)
 errors=read('physchem_provenance/error_comparison.csv');v=np.array([[float(errors[(errors.feature==f)&(errors.comparison==c)].cliffs_delta.iloc[0]) for c in ['FN_vs_TP','FP_vs_TN']] for f in features]);ax=axs[0,1]
 im=ax.imshow(v,vmin=-1,vmax=1,cmap='PuOr_r',aspect='auto')
 for i in range(6):
  for j in range(2):ax.text(j,i,f'{v[i,j]:+.2f}' if abs(v[i,j])>=.005 else '0.00',ha='center',va='center',color='white' if abs(v[i,j])>.6 else '#243D49',fontsize=8)
 ax.set_xticks([0,1],['FN vs TP','FP vs TN']);ax.set_yticks([])
 ax.set_xticks(np.arange(-.5,2,1),minor=True);ax.set_yticks(np.arange(-.5,6,1),minor=True);ax.grid(which='minor',c='white',lw=2);ax.tick_params(which='both',length=0)
 for s in ax.spines.values():s.set_visible(False)
 fig.colorbar(im,ax=ax,orientation='horizontal',fraction=.07,pad=.22,ticks=[-1,0,1]).set_label("Cliff's δ (point estimates)",fontsize=8)
 ax.set_title('b   Error-associated properties',loc='left',weight='bold',pad=15)
 df=read('physchem_provenance/test_predictions_with_physchem.csv');ax=axs[1,0]
 for outcome,col,m in [('TN','#CDD3D6','o'),('TP','#75B8D9','o'),('FP',GREEN,'s'),('FN',ORANGE,'^')]:
  t=df[df.outcome==outcome];ax.scatter(t.net_charge_per_residue,t.gravy,s=12 if outcome in ['TN','TP'] else 30,c=col,marker=m,alpha=.7 if outcome in ['TN','TP'] else 1,edgecolors='white' if outcome in ['FP','FN'] else 'none',linewidths=.4,label=f'{outcome} (n={len(t)})')
 ax.set_xlabel('Net charge per residue (pH 7)');ax.set_ylabel('GRAVY');ax.set_title('c   Prediction outcomes',loc='left',weight='bold',pad=15);ax.legend(loc='upper left',bbox_to_anchor=(-.04,-.24),frameon=False,ncol=2,fontsize=7,columnspacing=1,handletextpad=.3)
 holder=axs[1,1];spec=holder.get_subplotspec().subgridspec(2,1,height_ratios=[4,1],hspace=.08);holder.remove();ax=fig.add_subplot(spec[0]);hist=fig.add_subplot(spec[1],sharex=ax)
 bins=read('physchem_provenance/calibration_bins.csv');ax.plot([0,1],[0,1],c=GRAY,ls='--',lw=1);ax.plot(bins.mean_predicted_probability,bins.observed_fraction,'o-',c=BLUE,ms=4,lw=1)
 ax.set_ylim(-.03,1.06);ax.set_ylabel('Observed ACP fraction');ax.set_title('d   Prediction reliability',loc='left',weight='bold',pad=15);ax.text(.02,.89,'Brier = 0.0530',transform=ax.transAxes,fontsize=8);ax.tick_params(labelbottom=False)
 hist.hist(df.score,bins=np.linspace(0,1,11),color='#A3CADD');hist.set_xlabel('Predicted ACP probability');hist.set_ylabel('n');save(fig,out,'Figure_5')
 fig,ax=plt.subplots(figsize=(4,3.6));ax.plot([0,1],[0,1],'--',c=GRAY);ax.plot(bins.mean_predicted_probability,bins.observed_fraction,'o-',c=BLUE);ax.set_xlabel('Mean predicted ACP probability');ax.set_ylabel('Observed ACP fraction');ax.text(.04,.92,'Brier = 0.0530',transform=ax.transAxes);fig.tight_layout();save(fig,out,'Figure_S1')
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=ROOT/'reproduced/figures');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);figure3(a.out);figure5(a.out)
 print('Figures 3, 5 and S1 regenerated from archived numerical data.')
if __name__=='__main__':main()
