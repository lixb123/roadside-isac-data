"""Seed-cluster summaries, conservative loss bounds, figures and numeric tables."""
from pathlib import Path
import csv,json
import numpy as np
from scipy.stats import beta,binomtest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'analysis';OUT.mkdir(exist_ok=True)

def read(path):
 rows=list(csv.DictReader(path.open()))
 for r in rows:
  for k in list(r):
   try:r[k]=float(r[k])
   except ValueError:pass
 return rows

def flip_p(d):
 d=np.asarray(d);d=d[np.abs(d)>1e-12]
 if not len(d):return 1.
 sums=np.zeros(1)
 for v in d:sums=np.concatenate([sums+v,sums-v])
 return float(np.mean(np.abs(sums)>=abs(d.sum())-1e-10))

def boot(d):
 rng=np.random.default_rng(20261024);d=np.asarray(d)
 x=d[rng.integers(len(d),size=(20000,len(d)))].mean(axis=1)
 return np.quantile(x,[.025,.975]).tolist()

def main():
 rows=read(ROOT/'results-0.0025/trajectories.csv');assert len(rows)==504
 seeds=sorted({int(r['seed']) for r in rows});assert len(seeds)==24
 policies=['periodic-local','sparse-local','ai-mean'];metrics=['joint_delivery_5s','availability','requests','total_compute_s','decision_s','solver_s','setup_s','timeouts']
 # Average training seeds inside each physical trajectory first.
 normalized=[r for r in rows if not str(r['policy']).startswith('ai-')]
 for seed in seeds:
  for regime in ['cruise','turn_blockage','ood']:
   rr=[r for r in rows if r['seed']==seed and r['regime']==regime and str(r['policy']).startswith('ai-')];assert len(rr)==5
   normalized.append(dict(seed=seed,regime=regime,policy='ai-mean',**{m:float(np.mean([r[m] for r in rr])) for m in metrics}))
 summary=[]
 for policy in policies:
  for regime in ['cruise','turn_blockage','ood','all']:
   rr=[r for r in normalized if r['policy']==policy and (regime=='all' or r['regime']==regime)]
   summary.append(dict(policy=policy,regime=regime,**{m:float(np.mean([r[m] for r in rr])) for m in metrics}))
 def cluster(policy,metric,adverse=False):
  return np.array([np.mean([r[metric] for r in normalized if r['seed']==s and r['policy']==policy and (not adverse or r['regime']!='cruise')]) for s in seeds])
 comparisons=[]
 for baseline in policies[:2]:
  ai=cluster('ai-mean','joint_delivery_5s',True);base=cluster(baseline,'joint_delivery_5s',True)
  difference=ai-base;loss=np.maximum(0,-difference);k=int(np.sum(loss>1e-10));upper=float(beta.ppf(.975,k+1,24-k)) if k<24 else 1.
  saving=cluster(baseline,'total_compute_s')-cluster('ai-mean','total_compute_s')
  nonzero=saving[np.abs(saving)>1e-12]
  comparisons.append(dict(baseline=baseline,clusters=24,adverse_delivery_difference=float(difference.mean()),delivery_difference_ci=boot(difference),observed_mean_nonnegative_loss=float(loss.mean()),seeds_with_loss=k,simultaneous_loss_upper=upper,NI_1pp=upper<.01,NI_2pp=upper<.02,NI_5pp=upper<.05,compute_saved_s=float(saving.mean()),compute_saved_ci=boot(saving),relative_compute_saving=float(saving.mean()/cluster(baseline,'total_compute_s').mean()),signflip_p=flip_p(saving),sign_test_p=float(binomtest(int(np.sum(nonzero>0)),len(nonzero),.5).pvalue) if len(nonzero) else 1.,compute_difference_sd=float(saving.std(ddof=1)),paired_standardized_effect=float(saving.mean()/saving.std(ddof=1)) if saving.std(ddof=1)>0 else None))
 order=np.argsort([c['signflip_p'] for c in comparisons]);last=0.
 for rank,j in enumerate(order):
  last=max(last,min(1.,(2-rank)*comparisons[j]['signflip_p']));comparisons[j]['holm_p']=last
 learning=[]
 for p in sorted({r['policy'] for r in rows if str(r['policy']).startswith('ai-')}):
  rr=[r for r in rows if r['policy']==p]
  learning.append(dict(policy=p,requests=float(np.mean([r['requests'] for r in rr])),compute=float(np.mean([r['total_compute_s'] for r in rr])),adverse_delivery=float(np.mean([r['joint_delivery_5s'] for r in rr if r['regime']!='cruise']))))
 margin=json.loads((ROOT/'service-margin.json').read_text());proposed=ROOT/'service-margin-proposed.json'
 (OUT/'statistics.json').write_text(json.dumps(dict(summary=summary,comparisons=comparisons,learning_seeds=learning,initial_proposed_primary_margin=json.loads(proposed.read_text()) if proposed.exists() else margin,user_decision_margin=margin,unit='24 physical seed clusters; three correlated regimes; five learners averaged within trajectory'),indent=2))
 def write(path,rr):
  with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
 write(OUT/'summary.csv',summary)
 table='|Policy|Regime|5s delivery %|Availability %|CSI requests|Total compute s|Timeouts|\n|---|---|---:|---:|---:|---:|---:|\n'
 for r in summary:table+=f"|{r['policy']}|{r['regime']}|{100*r['joint_delivery_5s']:.3f}|{100*r['availability']:.3f}|{r['requests']:.4f}|{r['total_compute_s']:.6f}|{r['timeouts']:.4f}|\n"
 (OUT/'numeric-table.md').write_text(table)
 fig,ax=plt.subplots(figsize=(6.8,3.8));colors=['#607080','#df9a43','#167e89']
 for i,p in enumerate(policies):
  v=cluster(p,'total_compute_s');ci=boot(v);ax.bar(i,v.mean(),color=colors[i],width=.62);ax.errorbar(i,v.mean(),yerr=[[v.mean()-ci[0]],[ci[1]-v.mean()]],fmt='none',color='black',capsize=4)
 ax.set_xticks(range(3),['Periodic local','Sparse local','Learned (mean of 5)']);ax.set_ylabel('Compute per 120 s mission (s)');ax.set_title('Initialization, decisions, validation and setup included');ax.spines[['top','right']].set_visible(False);fig.tight_layout();fig.savefig(OUT/'resource-comparison.png',dpi=180);plt.close(fig)
 fig,ax=plt.subplots(figsize=(7,3.8));x=np.arange(3)
 for i,p in enumerate(policies):
  v=[next(r['joint_delivery_5s'] for r in summary if r['policy']==p and r['regime']==reg) for reg in ['cruise','turn_blockage','ood']]
  ax.bar(x+(i-1)*.24,np.array(v)*100,width=.23,color=colors[i],label=p)
 ax.set_xticks(x,['Cruise','Turn/blockage','OOD']);ax.set_ylim(0,110);ax.set_ylabel('Joint-compliant 5 s delivery (%)');ax.legend(frameon=False,ncol=3,fontsize=8);ax.set_title('Descriptive means; noninferiority requires a separate bound');ax.spines[['top','right']].set_visible(False);fig.tight_layout();fig.savefig(OUT/'service-breakdown.png',dpi=180);plt.close(fig)
 print(json.dumps(comparisons,indent=2));print(table)

if __name__=='__main__':main()
