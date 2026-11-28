"""Synthesize fixed-cohort analyses and render scientific figures."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,default=ROOT/'sources')
args=parser.parse_args();SOURCE=args.source
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180,'pdf.fonttype':42})
figdir=ROOT/'figures';figdir.mkdir(exist_ok=True)
data=pd.read_csv(SOURCE/'paired_hypotheses.csv')
effects=pd.read_csv(ROOT/'effects/paired_effects.csv')
corr=pd.read_csv(ROOT/'correlations/primary_correlations.csv')
groups=pd.read_csv(ROOT/'groups/categorical_omnibus_tests.csv')
group_effects=pd.read_csv(ROOT/'groups/group_effects.csv')

def adjust(p,kind):
    p=np.asarray(p,float);order=np.argsort(p,kind='stable');m=len(p);out=np.empty(m)
    if kind=='bh':out[order]=np.minimum.accumulate((p[order]*m/np.arange(1,m+1))[::-1])[::-1]
    else:out[order]=np.maximum.accumulate(p[order]*(m-np.arange(m)))
    return np.clip(out,0,1)
tests=[]
for r in groups.to_dict('records'):
    tests.append({'family':'categorical','test':f"{r['grouping']} | {r['method']} - {r['control']}",'p':r['permutation_p'],'within_family_q':r['bh_q_18_tests']})
for r in corr.to_dict('records'):
    tests.append({'family':'continuous','test':f"{r['feature']} | {r['hypothesis']} - {r['comparator']}",'p':r['permutation_p_two_sided'],'within_family_q':r['bh_q_within_family']})
alltests=pd.DataFrame(tests);assert len(alltests)==46
alltests['bh_q_all_46']=adjust(alltests.p,'bh');alltests['holm_p_all_46']=adjust(alltests.p,'holm')
alltests=alltests.sort_values('p').reset_index(drop=True);alltests.to_csv(ROOT/'all_exploratory_primary_tests.csv',index=False)
global_summary={'primary_exploratory_tests':46,'categorical_tests':18,'continuous_tests':28,
 'global_bh_discoveries_at_0_05':int((alltests.bh_q_all_46<.05).sum()),
 'global_holm_discoveries_at_0_05':int((alltests.holm_p_all_46<.05).sum()),
 'smallest_global_bh_q':float(alltests.bh_q_all_46.min()),'strongest_test':alltests.iloc[0].to_dict(),
 'scope':'Global sensitivity across all primary exploratory group/feature tests; six prespecified method/control sign tests remain their own family. Log-ratio sensitivities and explicitly coupled baseline diagnostics are not promoted to discoveries.'}
(ROOT/'global_multiplicity.json').write_text(json.dumps(global_summary,indent=2),encoding='utf-8')

colors={'H1':'#2563A6','H2':'#C26924','H3':'#7A5292'}
fig,axes=plt.subplots(1,2,figsize=(11.8,4.3),sharex=True,sharey=True)
for ax,control,title in zip(axes,['baseline','scalar'],['Compared with original VGGT','Compared with scalar control']):
    ax.axvline(0,color='#555',lw=1,ls='--')
    for y,h in enumerate(['H1','H2','H3']):
        r=effects[(effects.hypothesis==h)&(effects.control==control)].iloc[0]
        ax.errorbar(r.mean_delta,y,xerr=[[r.mean_delta-r.mean_delta_ci95_low],[r.mean_delta_ci95_high-r.mean_delta]],fmt='o',capsize=5,color=colors[h],lw=2,ms=8)
        ax.text(-1.65,y+.23,f"{int(r.wins)} wins / {int(r.ties)} ties / {int(r.losses)} losses",fontsize=9,color='#555')
    ax.set_yticks([0,1,2],['H1','H2','H3']);ax.set_ylim(2.5,-.5);ax.set_xlim(-1.75,2.55)
    ax.set_title(title,loc='left',fontweight='bold',fontsize=12);ax.set_xlabel('Mean paired change in CD_norm (negative is better)');ax.grid(axis='x',alpha=.15)
fig.suptitle('All six mean-effect intervals include zero',x=.065,ha='left',fontweight='bold',fontsize=16)
fig.text(.065,.018,'36 confirmation landmarks; 100,000 paired-bootstrap samples. Pointwise 95% intervals; selected cohort, not population guarantees.',fontsize=9,color='#555')
fig.tight_layout(rect=[0,.07,1,.91]);fig.savefig(figdir/'paired_effects.png');fig.savefig(figdir/'paired_effects.pdf');plt.close(fig)

# Every landmark remains visible; percent change is a secondary scale, not the official mean estimand.
d=data.sort_values('scene').reset_index(drop=True)
heat=np.column_stack([100*(d[h]/d[c]-1) for c in ['baseline','scalar'] for h in ['H1','H2','H3']])
fig,ax=plt.subplots(figsize=(11,14.2));im=ax.imshow(heat,cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-55,vcenter=0,vmax=55),aspect='auto')
ax.set_yticks(range(len(d)),[s.replace('_',' ') for s in d.scene],fontsize=8.4)
ax.set_xticks(range(6),['H1 vs original','H2 vs original','H3 vs original','H1 vs scalar','H2 vs scalar','H3 vs scalar'],rotation=35,ha='right')
ax.axvline(2.5,color='white',lw=4)
for i in range(len(d)):
    for j in range(6):ax.text(j,i,f'{heat[i,j]:+.1f}' if abs(heat[i,j])>=.05 else '0.0',ha='center',va='center',fontsize=7.4,color='white' if abs(heat[i,j])>30 else '#182532')
ax.set_title('Individual landmark changes (%)\nNegative = lower error; positive = higher error',loc='left',fontweight='bold',fontsize=14,pad=14)
cb=fig.colorbar(im,ax=ax,pad=.02,shrink=.7);cb.set_label('100 × (hypothesis / control − 1)')
fig.text(.03,.01,'All 36 confirmation landmarks. H3 reuses the scalar prediction on 33 landmarks. Percentages are per-landmark descriptive outcomes.',fontsize=8.5)
fig.tight_layout(rect=[0,.025,1,1]);fig.savefig(figdir/'landmark_changes.png');fig.savefig(figdir/'landmark_changes.pdf');plt.close(fig)

# Strongest raw continuous associations, disclosed as an exploratory subset of the full table.
top=corr.sort_values('permutation_p_two_sided').head(6)
pretty={'mean_applied_relative_update':'Relative update magnitude','scalar_mean_applied_relative_update':'Scalar relative update','coefficient_max_to_min':'Maximum/minimum strength','coefficient_max':'Maximum strength','coefficient_min':'Minimum strength','extent_m':'Physical extent'}
fig,ax=plt.subplots(figsize=(11.8,5.2));ax.axvline(0,color='#555',ls='--',lw=1)
labels=[]
for y,r in enumerate(top.to_dict('records')):
    labels.append(f"{r['hypothesis']} − {'original' if r['comparator']=='baseline' else 'scalar'}\n{pretty[r['feature']]}")
    ax.errorbar(r['spearman_rho'],y,xerr=[[r['spearman_rho']-r['bootstrap_ci_low']],[r['bootstrap_ci_high']-r['spearman_rho']]],fmt='o',capsize=4,color=colors[r['hypothesis']],ms=7,lw=1.8)
    key=f"{r['feature']} | {r['hypothesis']} - {r['comparator']}";q=alltests.loc[alltests.test==key,'bh_q_all_46'].iloc[0]
    ax.text(.76,y,f"q = {q:.3f}",va='center',fontsize=10)
ax.set_yticks(range(len(top)),labels,fontsize=9);ax.set_ylim(len(top)-.5,-.6);ax.set_xlim(-.8,.98);ax.set_xlabel('Spearman correlation with signed error difference (positive = worse effect)')
ax.set_title('Exploratory correlations do not survive global correction',loc='left',fontweight='bold',fontsize=14,pad=18)
ax.grid(axis='x',alpha=.15);fig.text(.03,.025,'Six smallest raw p-values among 28 continuous tests; pointwise bootstrap intervals. q pools all 46 primary group/feature tests.',fontsize=9,color='#555')
fig.tight_layout(rect=[0,.06,1,1]);fig.savefig(figdir/'exploratory_correlations.png');fig.savefig(figdir/'exploratory_correlations.pdf');plt.close(fig)

# Display the regional signal with individual observations, including small-group/outlier effects.
joined=pd.read_csv(ROOT/'groups/confirmation_with_groups.csv')
regions=sorted(joined.region.unique());rng=np.random.default_rng(20261126)
fig,ax=plt.subplots(figsize=(11.8,4.8));ax.axhline(0,color='#555',ls='--',lw=1)
for x,region in enumerate(regions):
    rows=joined[joined.region==region];delta=100*(rows.H1/rows.scalar-1)
    ax.scatter(x+rng.uniform(-.12,.12,len(rows)),delta,s=38,color=colors['H1'],alpha=.8)
    ax.plot(x,float(delta.median()),'D',ms=7,color='#111')
ax.set_xticks(range(len(regions)),[f"{r}\n(n={int((joined.region==r).sum())})" for r in regions],fontsize=9)
ax.set_ylabel('H1 change vs scalar (%)\nNegative is better')
ax.set_title('Regional differences in H1 versus scalar are exploratory',loc='left',fontweight='bold',fontsize=14,pad=15);ax.grid(axis='y',alpha=.15)
region_test=alltests[alltests.test=='region | H1 - scalar'].iloc[0]
fig.text(.03,.02,f'Dots: landmarks. Diamonds: medians. Rank test p = {region_test.p:.5f}; within-family q = {region_test.within_family_q:.5f}; global q = {region_test.bh_q_all_46:.5f}.',fontsize=9,color='#555')
fig.tight_layout(rect=[0,.06,1,1]);fig.savefig(figdir/'h1_regional_pattern.png');fig.savefig(figdir/'h1_regional_pattern.pdf');plt.close(fig)
print(json.dumps(global_summary,indent=2))
