"""Recompute descriptive length strata and optional stratified AUC intervals."""
from pathlib import Path
import argparse,sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'meta_acp'))
from metrics import binary_metrics,_roc_auc
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bootstrap',type=int,default=0);p.add_argument('--out',type=Path,default=ROOT/'reproduced/length');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 run=ROOT/'meta_acp/runs_gpu';primary=pd.read_csv(run/'ensemble_with_hardcase_strict/strict_logistic_stack/predictions.csv');rows=[]
 rng=np.random.default_rng(20260910)
 for label,mask in [('11–50',primary.sequence.str.len()<=50),('51–255',primary.sequence.str.len()>50)]:
  d=primary[mask];r={'length':label,**binary_metrics(d.binary_label,d.score)}
  if a.bootstrap:
   y=d.binary_label.to_numpy();s=d.score.to_numpy();pos=np.flatnonzero(y==1);neg=np.flatnonzero(y==0);est=[]
   for _ in range(a.bootstrap):
    idx=np.r_[rng.choice(pos,len(pos),replace=True),rng.choice(neg,len(neg),replace=True)];est.append(_roc_auc(y[idx],s[idx]))
   r['ci_low'],r['ci_high']=np.quantile(est,[.025,.975])
  rows.append(r)
 assert [r['n'] for r in rows]==[535,42]
 assert [round(r['auc'],4) for r in rows]==[.9696,.9944]
 pd.DataFrame(rows).to_csv(a.out/'integrated_length.csv',index=False)
 rows=[]
 for source in ['Test1','Test2']:
  for seed in [42,2024,2026]:
   d=pd.read_csv(run/f'loso/{source}_fasta_seed{seed}/predictions.csv')
   for group,mask in [('All',np.ones(len(d),dtype=bool)),('11–50',d.sequence.str.len()<=50),('>50',d.sequence.str.len()>50)]:
    part=d[mask]
    if len(part):rows.append({'source':source,'seed':seed,'length':group,**binary_metrics(part.binary_label,part.score)})
 raw=pd.DataFrame(rows);raw.to_csv(a.out/'loso_length_by_seed.csv',index=False);raw.groupby(['source','length'])[['n','pos','neg','auc','mcc','acc']].mean().to_csv(a.out/'loso_length_means.csv')
 print('Length subgroup counts and AUCs agree with the manuscript.')
if __name__=='__main__':main()
