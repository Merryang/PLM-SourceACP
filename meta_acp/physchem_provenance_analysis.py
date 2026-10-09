"""Reproducible physicochemical, provenance, error and calibration analysis."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from scipy.stats import mannwhitneyu, kruskal, spearmanr
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from statsmodels.stats.multitest import multipletests

AA20=set("ACDEFGHIKLMNPQRSTVWY")
KD={"I":4.5,"V":4.2,"L":3.8,"F":2.8,"C":2.5,"M":1.9,"A":1.8,"G":-0.4,"T":-0.7,"S":-0.8,"W":-0.9,"Y":-1.3,"P":-1.6,"H":-3.2,"E":-3.5,"Q":-3.5,"D":-3.5,"N":-3.5,"K":-3.9,"R":-4.5}
FEATURES=["length","molecular_weight","molecular_weight_per_residue","isoelectric_point","gravy","aromaticity","net_charge_ph7","net_charge_per_residue","helix_fraction","turn_fraction","sheet_fraction","hydrophobic_moment","hydrophobic_fraction","positive_fraction","negative_fraction","aliphatic_index"]

def hydrophobic_moment(seq, angle=100.0):
    theta=np.deg2rad(angle); vals=np.array([KD[a] for a in seq]); idx=np.arange(len(seq))
    return float(np.hypot(np.sum(vals*np.cos(idx*theta)),np.sum(vals*np.sin(idx*theta)))/len(seq))

def features(seq):
    seq=str(seq).strip().upper(); bad="".join(sorted(set(seq)-AA20))
    if not seq or bad: return None,bad or "empty"
    p=ProteinAnalysis(seq); n=len(seq); helix,turn,sheet=p.secondary_structure_fraction()
    counts=p.count_amino_acids()
    out={"length":n,"molecular_weight":p.molecular_weight(),"molecular_weight_per_residue":p.molecular_weight()/n,
         "isoelectric_point":p.isoelectric_point(),"gravy":p.gravy(),"aromaticity":p.aromaticity(),
         "net_charge_ph7":p.charge_at_pH(7.0),"net_charge_per_residue":p.charge_at_pH(7.0)/n,
         "helix_fraction":helix,"turn_fraction":turn,"sheet_fraction":sheet,
         "hydrophobic_moment":hydrophobic_moment(seq),
         "hydrophobic_fraction":sum(a in "AILMFWVY" for a in seq)/n,
         "positive_fraction":sum(a in "KRH" for a in seq)/n,
         "negative_fraction":sum(a in "DE" for a in seq)/n,
         "aliphatic_index":100*(counts["A"]/n+2.9*counts["V"]/n+3.9*(counts["I"]+counts["L"])/n)}
    out.update({f"aac_{a}":counts[a]/n for a in sorted(AA20)})
    return out,""

def cliff_delta(x,y):
    x=np.asarray(x);y=np.asarray(y); ys=np.sort(y)
    return float(np.sum(np.searchsorted(ys,x,"left")-(len(y)-np.searchsorted(ys,x,"right")))/(len(x)*len(y)))

def bootstrap_delta(x,y,rng,n=2000):
    vals=[cliff_delta(rng.choice(x,len(x),True),rng.choice(y,len(y),True)) for _ in range(n)]
    return np.percentile(vals,[2.5,97.5])

def class_stats(df,rng):
    rows=[]
    for f in FEATURES:
        a=df.loc[df.binary_label==1,f].dropna().to_numpy(); b=df.loc[df.binary_label==0,f].dropna().to_numpy()
        u,p=mannwhitneyu(a,b,alternative="two-sided"); d=cliff_delta(a,b); lo,hi=bootstrap_delta(a,b,rng)
        rows.append({"feature":f,"n_acp":len(a),"median_acp":np.median(a),"iqr_acp":np.subtract(*np.percentile(a,[75,25])),
                     "n_non_acp":len(b),"median_non_acp":np.median(b),"iqr_non_acp":np.subtract(*np.percentile(b,[75,25])),
                     "mann_whitney_u":u,"p_value":p,"cliffs_delta":d,"delta_ci_low":lo,"delta_ci_high":hi})
    out=pd.DataFrame(rows); out["p_bh"]=multipletests(out.p_value,method="fdr_bh")[1]; return out

def source_stats(df):
    rows=[]
    for label in [0,1]:
        part=df[df.binary_label==label]
        for f in FEATURES:
            groups=[x[f].dropna().to_numpy() for _,x in part.groupby("source_file") if len(x)>=2]
            if len(groups)<2: continue
            h,p=kruskal(*groups); n=sum(map(len,groups)); k=len(groups)
            eps=max(0,(h-k+1)/(n-k)) if n>k else np.nan
            rows.append({"binary_label":label,"feature":f,"n_sources":k,"n":n,"kruskal_h":h,"p_value":p,"epsilon_squared":eps})
    out=pd.DataFrame(rows); out["p_bh"]=out.groupby("binary_label").p_value.transform(lambda x:multipletests(x,method="fdr_bh")[1]); return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data",required=True); ap.add_argument("--predictions",required=True); ap.add_argument("--out",required=True); ap.add_argument("--bootstrap",type=int,default=2000); args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True); rng=np.random.default_rng(20260713)
    base=pd.read_csv(args.data); rows=[];invalid=[]
    for _,r in base.iterrows():
        z,bad=features(r.sequence)
        if z is None: invalid.append({"seq_id":r.get("seq_id",""),"sequence":r.sequence,"invalid_symbols":bad}); continue
        rows.append({**r.to_dict(),**z})
    feat=pd.DataFrame(rows); feat.to_csv(out/"physchem_per_sequence.csv",index=False);pd.DataFrame(invalid).to_csv(out/"invalid_sequences.csv",index=False)
    cs=class_stats(feat,rng);cs.to_csv(out/"class_comparison.csv",index=False)
    ss=source_stats(feat);ss.to_csv(out/"source_stratified_tests.csv",index=False)
    summary=feat.groupby(["source_file","binary_label"])[FEATURES].agg(["count","median","mean","std"]).reset_index();summary.to_csv(out/"source_class_summary.csv",index=False)
    pred=pd.read_csv(args.predictions).drop(columns=[c for c in FEATURES if c in pd.read_csv(args.predictions,nrows=0).columns])
    test=pred.merge(feat[["seq_id"]+FEATURES],on="seq_id",how="left",validate="one_to_one")
    test["outcome"]=np.select([(test.binary_label==1)&(test.prediction==1),(test.binary_label==0)&(test.prediction==0),(test.binary_label==0)&(test.prediction==1)],["TP","TN","FP"],default="FN")
    test.to_csv(out/"test_predictions_with_physchem.csv",index=False)
    er=[]
    for f in FEATURES:
        for target,ref in [("FN","TP"),("FP","TN")]:
            x=test.loc[test.outcome==target,f].dropna();y=test.loc[test.outcome==ref,f].dropna()
            if len(x) and len(y):
                u,p=mannwhitneyu(x,y,alternative="two-sided");er.append({"comparison":f"{target}_vs_{ref}","feature":f,"n_target":len(x),"n_reference":len(y),"median_target":x.median(),"median_reference":y.median(),"cliffs_delta":cliff_delta(x,y),"p_value":p})
    er=pd.DataFrame(er);er["p_bh"]=er.groupby("comparison").p_value.transform(lambda x:multipletests(x,method="fdr_bh")[1]);er.to_csv(out/"error_comparison.csv",index=False)
    cr=[]
    for f in FEATURES:
        rho,p=spearmanr(test[f],test.score,nan_policy="omit");cr.append({"feature":f,"spearman_rho":rho,"p_value":p})
    cr=pd.DataFrame(cr);cr["p_bh"]=multipletests(cr.p_value,method="fdr_bh")[1];cr.to_csv(out/"score_correlations.csv",index=False)
    prob_true,prob_pred=calibration_curve(test.binary_label,test.score,n_bins=10,strategy="quantile")
    pd.DataFrame({"mean_predicted_probability":prob_pred,"observed_fraction":prob_true}).to_csv(out/"calibration_bins.csv",index=False)
    meta={"n_input":len(base),"n_valid":len(feat),"n_invalid":len(invalid),"pH":7.0,"hydropathy_scale":"Kyte-Doolittle","hydrophobic_moment_angle_degrees":100.0,"bootstrap_iterations":args.bootstrap,"brier_score":brier_score_loss(test.binary_label,test.score)}
    (out/"analysis_metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    print(json.dumps(meta,indent=2))

if __name__=="__main__": main()
