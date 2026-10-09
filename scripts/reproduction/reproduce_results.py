"""Recompute manuscript estimates from frozen predictions without model downloads."""
from pathlib import Path
import argparse,json,sys,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'meta_acp'))
from metrics import binary_metrics

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=ROOT/'reproduced');args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
 runs=ROOT/'meta_acp/runs_gpu'
 entries=[('integrated_primary',runs/'ensemble_with_hardcase_strict/strict_logistic_stack/predictions.csv',0.5,0.9709),('integrated_auxiliary',runs/'ensemble_with_hardcase_strict/logistic_stack/predictions.csv',0.5,0.9737),('selected_single',runs/'ensemble_with_hardcase_strict/member_predictions/test/esm2_150m_seed2024.csv',0.5,0.9562),('aegis',runs/'final_statistics/aegis_esm2_150m_3seed_ensemble_predictions.csv',0.4423266351222992,0.9849),('antiangiogenic',runs/'external_public_summary/antiangiopred_full_head_3seed_ensemble_predictions.csv',0.4422,0.8036),('cell_penetrating',runs/'external_public_summary/pepbenchmark_cpp_full_head_3seed_ensemble_predictions.csv',0.4438,0.7979)]
 rows=[]
 for name,path,threshold,expected in entries:
  df=pd.read_csv(path);assert not df.sequence.duplicated().any(),name
  m=binary_metrics(df.binary_label,df.score,threshold);assert round(m['auc'],4)==expected,(name,m)
  if 'prediction' in df:assert np.array_equal((df.score>=threshold).astype(int),df.prediction)
  rows.append({'evaluation':name,'threshold':threshold,**m})
  if name=='integrated_primary':
   assert round(m['ap'],4)==0.9659 and round(m['mcc'],4)==0.8600
   assert round(float(np.mean((df.score-df.binary_label)**2)),4)==0.0530
  if name=='aegis':assert round(m['ap'],4)==0.9960
 pd.DataFrame(rows).to_csv(args.out/'performance_recomputed.csv',index=False)
 seeds=[]
 for f in (runs/'ensemble_with_hardcase_strict/member_predictions/test').glob('*.csv'):
  import re
  match=re.fullmatch(r'(ankh|esm2_150m|esm2_650m|esmc|protbert|prott5)_seed(42|2024|2026)',f.stem)
  if match:
   df=pd.read_csv(f);seeds.append({'backbone':match[1],'seed':int(match[2]),**binary_metrics(df.binary_label,df.score)})
 raw=pd.DataFrame(seeds);assert len(raw)==18
 raw.to_csv(args.out/'backbone_seed_metrics.csv',index=False)
 summary=raw.groupby('backbone')[['auc','ap','mcc','acc']].agg(['mean','std']);summary.columns=['_'.join(c) for c in summary.columns];summary.to_csv(args.out/'backbone_summary.csv')
 splits={n:pd.read_csv(ROOT/'data/splits'/(n+'.csv')) for n in ['train','validation','test']}
 assert [len(splits[n]) for n in splits]==[1728,577,577]
 for a,b in [('train','validation'),('train','test'),('validation','test')]:assert not set(splits[a].sequence)&set(splits[b].sequence)
 master=pd.read_csv(ROOT/'data/processed/merged_all_cdhit08.csv');assert len(master)==2882 and int(master.binary_label.sum())==1036
 loso=[]
 for name in ['Train','Test1','Test2']:
  for seed in [42,2024,2026]:
   df=pd.read_csv(runs/f'loso/{name}_fasta_seed{seed}/predictions.csv');loso.append({'source':name,'seed':seed,**binary_metrics(df.binary_label,df.score)})
 pd.DataFrame(loso).to_csv(args.out/'loso_seed_metrics.csv',index=False)
 report={'status':'passed','checks':['six reported AUC estimates','primary AP and MCC','Aegis AP','primary Brier score','archived binary predictions','18 standard backbone predictions','nine LOSO predictions','integrated split sizes and exact sequence separation'],'not_tested':['full model training','GPU embedding extraction','bitwise equivalence across software and hardware'],'metric_implementation':'meta_acp/metrics.py preserves the archived computation, including its treatment of score ties'}
 (args.out/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
