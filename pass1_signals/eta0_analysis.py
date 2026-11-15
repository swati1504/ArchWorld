#!/usr/bin/env python
"""Add Eric's eta=0 base run (eric/vggt_base_eta0_dev_results.csv) to swati's sweep: gain over no
intervention, noise floor (eta=1 vs eta=0), LOSO with eta=0 allowed, leakage vs base error.
Writes pass1_out/labels_with_eta0.csv.   python pass1_signals/eta0_analysis.py
"""
import sys; sys.path.insert(0,"pass1_signals")
import numpy as np, pandas as pd
from scipy.stats import spearmanr, wilcoxon
from selection import load_curves, run_all, sel_fixed, sel_best_fixed, sel_oracle, regret
sw=pd.read_csv("swati/results_vggt_dev/per_scene.csv")
b=pd.read_csv("eric/vggt_base_eta0_dev_results.csv")
print("status:",b.status.value_counts().to_dict())
e0=b[["scene","acc_mean","acc_med","comp_mean","comp_med","cd"]].rename(columns={"acc_mean":"Acc-mean","acc_med":"Acc-med","comp_mean":"Comp-mean","comp_med":"Comp-med","cd":"Score"}); e0["eta"]=0
lab=pd.concat([e0[sw.columns],sw]); lab.to_csv("pass1_out/labels_with_eta0.csv",index=False)
scenes,etas,X=load_curves("pass1_out/labels_with_eta0.csv"); print(len(scenes),"scenes, grid",etas)
base=X[:,0]
print("\n== each eta vs eta=0 (no intervention), per scene relative change of CD")
for j,e in enumerate(etas[1:],1):
    d=X[:,j]/base-1
    print(f"eta={e:5d}: median {np.median(d):+.2%}  mean {d.mean():+.2%}  better on {(d<0).sum()}/39  wilcoxon p={wilcoxon(X[:,j],base).pvalue:.3f}")
d=X[:,1:].min(1)/base-1; print(f"oracle over eta>0: median {np.median(d):+.2%}, better on {(d<0).sum()}/39")
d=X.min(1)/base-1; print(f"oracle incl eta=0: median {np.median(d):+.2%}; eta=0 is best on {(X.argmin(1)==0).sum()}/39")
n=np.abs(X[:,1]/base-1); print(f"\n== noise floor: |eta=1 vs eta=0| per scene: median {np.median(n):.2%}, 75% {np.percentile(n,75):.2%}, max {n.max():.2%}")
sp=X[:,1:].max(1)/X[:,1:].min(1)-1; print(f"   for scale: spread worst/best over eta>0: median {np.median(sp):.1%}")
srt=np.sort(X[:,1:],1); gap=srt[:,1]/srt[:,0]-1; print(f"   gap best vs runner-up eta: median {np.median(gap):.2%}; scenes with gap < median noise: {(gap<np.median(n)).sum()}/39")
print("\n== LOSO with eta=0 allowed (ref = eta=0)")
sel=[(f"eta={e}",sel_fixed(j)) for j,e in enumerate(etas)]+[("best-fixed",sel_best_fixed),("oracle",sel_oracle)]
pd.set_option("display.width",220); print(run_all(sel,None,X,etas,ref=0).to_string(index=False,float_format=lambda v:f"{v:.4f}"))
# leakage vs base error
F=pd.read_csv("pass1_out/pass1_features.csv").set_index("scene").loc[scenes]
y=b.set_index("scene").loc[scenes,"cd_norm"].values
print("\n== leakage claim: Spearman(feature, base CD_norm), n=39")
for c in [c for c in F.columns if "leak" in c or "sky" in c or "nonarch" in c]:
    r,p=spearmanr(F[c],y); print(f"{c:24s} rho {r:+.3f} p {p:.3f}")
