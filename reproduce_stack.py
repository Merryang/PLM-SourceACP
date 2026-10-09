"""Refit the manuscript stack using archived validation member predictions."""
from pathlib import Path
import argparse,json,sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'meta_acp'))
from metrics import binary_metrics
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=ROOT/'reproduced/stack');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 base=ROOT/'meta_acp/runs_gpu/ensemble_with_hardcase_strict';cfg=json.loads((base/'config.json').read_text());members=[m.split(':')[0] for m in cfg['models']]
 def matrix(split):
  dfs=[pd.read_csv(base/'member_predictions'/split/(m+'.csv')) for m in members]
  reference=dfs[0].set_index('sequence')
  arrays=[]
  for df in dfs:
   assert not df.sequence.duplicated().any()
   df=df.set_index('sequence').loc[reference.index];assert np.array_equal(df.binary_label,reference.binary_label)
   arrays.append(df.score.to_numpy())
  return np.column_stack(arrays),reference.binary_label.to_numpy(),reference.index
 xv,yv,_=matrix('validation');xt,yt,seq=matrix('test');candidates=[];models={}
 for c in [.01,.03,.1,.3,1.,3.]:
  model=LogisticRegression(C=c,max_iter=5000,class_weight='balanced',solver='lbfgs').fit(xv,yv)
  candidates.append({'C':c,'validation_auc':binary_metrics(yv,model.predict_proba(xv)[:,1])['auc']});models[c]=model
 selected=max(candidates,key=lambda r:r['validation_auc'])['C'];assert selected==3.,candidates
 scores=models[selected].predict_proba(xt)[:,1];m=binary_metrics(yt,scores)
 archive=pd.read_csv(base/'strict_logistic_stack/predictions.csv').set_index('sequence').loc[seq]
 delta=float(np.max(np.abs(scores-archive.score.to_numpy())))
 pd.DataFrame(candidates).to_csv(a.out/'validation_selection.csv',index=False)
 pd.DataFrame({'sequence':seq,'binary_label':yt,'score':scores}).to_csv(a.out/'test_predictions.csv',index=False)
 (a.out/'stack_coefficients.json').write_text(json.dumps({'members':members,'C':selected,'intercept':models[selected].intercept_.tolist(),'coefficients':models[selected].coef_[0].tolist(),'metrics':m,'maximum_score_difference_from_archive':delta},indent=2),encoding='utf-8')
 assert round(m['auc'],4)==.9709 and round(m['mcc'],4)==.8600,m
 print(json.dumps({'selected_C':selected,'AUC':m['auc'],'AP':m['ap'],'MCC':m['mcc'],'maximum_score_difference_from_archive':delta},indent=2))
if __name__=='__main__':main()
