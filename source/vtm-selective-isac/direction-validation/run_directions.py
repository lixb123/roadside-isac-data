"""Frozen sequential screening; no test-dependent parameter tuning."""
from __future__ import annotations
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import numpy as np
from scipy.stats import shapiro, binomtest
from direction_env import ROOT, Asset, Env, load_assets

OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
BASE=['guard','periodic20','periodic10','periodic5']
STAGE1=BASE+['timing','budget','joint']

def write_csv(path:Path,rows:list[dict])->None:
 with path.open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def trigger(x:np.ndarray,threshold:float=.08)->bool:
 return x[1]>=1 or x[2]>=threshold or x[3]<.08 or x[6]<.08

def choose(x:np.ndarray,name:str,parameter:float=.08)->int:
 now=round(x[0]*120);risk=trigger(x,parameter)
 if name.startswith('periodic'):return 2 if now%int(name[8:])==0 else 0
 if name=='guard':return 2 if now%20==0 and risk else 0
 if name=='timing':return 2 if risk else 0
 # Select faster local computation if the predicted use window is short.
 action=1 if x[7]*40<5 else 2
 if name=='budget':return action if now%20==0 else 0
 if name=='joint':return action if risk else 0
 if name=='robust':return 3 if risk else 0
 raise ValueError(name)

def rollout(asset:Asset,name:str,parameter:float=.08,model:dict|None=None)->tuple[dict,list[dict]]:
 env=Env(asset);x=env.observe()
 while env.k<24:
  allowed=env.allowed()
  if name=='voi':
   assert model is not None
   values=[float(features(x,a)@model['weights']) for a in allowed if a<3]
   allowed=[a for a in allowed if a<3];action=allowed[int(np.argmax(values))]
  elif name.startswith('rl'):
   assert model is not None
   allowed=[a for a in allowed if a<3];q=model['q'].get(state_key(x),np.zeros(3));action=max(allowed,key=lambda a:q[a])
  else:action=choose(x,name,parameter)
  if action not in allowed:action=0
  x,_,_,_=env.step(action)
 result=env.result();result['policy']=name
 rows=[dict(policy=name,**r) for r in env.rows]
 return result,rows

def score(rows:list[dict])->float:
 adverse=[r['joint_delivery_5s'] for r in rows if r['regime']!='cruise']
 return float(np.mean(adverse))+.001*float(np.mean([r['availability'] for r in rows]))-.0001*float(np.mean([r['compute'] for r in rows]))

def features(x:np.ndarray,a:int)->np.ndarray:
 v=np.r_[x,x*x,np.maximum(0,x[6]),x[8]*x[0]];out=np.zeros(3*len(v));out[a*len(v):(a+1)*len(v)]=v;return out

def fit_voi(train:list[Asset])->dict:
 rng=np.random.default_rng(241024);xx=[];yy=[]
 for asset in train:
  for episode in range(24):
   env=Env(asset);x=env.observe()
   while env.k<24:
    action=int(rng.choice([a for a in env.allowed() if a<3]));old=x
    x,reward,_,_=env.step(action);xx.append(features(old,action));yy.append(reward)
 X=np.array(xx);y=np.array(yy);w=np.linalg.solve(X.T@X+.1*np.eye(X.shape[1]),X.T@y)
 np.savez(OUT/'voi-model.npz',weights=w,train_seeds=np.array([a.seed for a in train]))
 return dict(weights=w)

def state_key(x:np.ndarray)->str:
 bins=[np.digitize(x[0],[.33,.67]),np.digitize(x[1],[.25,.75]),np.digitize(x[6],[-.1,.1]),
       np.digitize(x[2],[.08,.25]),np.digitize(x[7],[.05,.2]),np.digitize(x[8],[.17,.5]),round(x[10]*4)]
 return ','.join(map(str,bins))

def train_rl(train:list[Asset],validation:list[Asset])->list[dict]:
 models=[];curves=[]
 for seed in [31,47,59,71,89]:
  rng=np.random.default_rng(seed);q={};best=None;best_score=-math.inf
  for episode in range(1,601):
   asset=train[int(rng.integers(len(train)))];env=Env(asset);x=env.observe();total=0.
   while env.k<24:
    key=state_key(x);values=q.setdefault(key,np.zeros(3));allowed=[a for a in env.allowed() if a<3]
    eps=max(.05,.6*(1-episode/600));action=int(rng.choice(allowed)) if rng.random()<eps else max(allowed,key=lambda a:values[a])
    nx,r,done,_=env.step(action);nkey=state_key(nx);nvalues=q.setdefault(nkey,np.zeros(3));nallowed=[a for a in env.allowed() if a<3]
    target=r+(0 if done else .95*max(nvalues[a] for a in nallowed));values[action]+=.15*(target-values[action]);x=nx;total+=r
   if episode in [100,300,600]:
    model=dict(q=q);results=[rollout(a,'rl',model=model)[0] for a in validation];s=score(results)
    curves.append(dict(learning_seed=seed,episode=episode,validation_score=s,episode_reward=total,states=len(q)))
    if s>best_score:best_score=s;best=dict(q={k:v.copy() for k,v in q.items()},learning_seed=seed,episode=episode)
  assert best is not None;models.append(best)
  (OUT/f'rl-model-{seed}.json').write_text(json.dumps(dict(seed=seed,episode=best['episode'],q={k:v.tolist() for k,v in best['q'].items()}),indent=2),encoding='utf-8')
  print(f'RL seed {seed} trained; validation checkpoint {best["episode"]}',flush=True)
 write_csv(OUT/'rl-training.csv',curves);return models

def ci(x:np.ndarray,seed:int=241024)->tuple[float,float]:
 rng=np.random.default_rng(seed);b=x[rng.integers(0,len(x),(20000,len(x)))].mean(axis=1);return tuple(map(float,np.quantile(b,[.025,.975])))

def contrast(rows:list[dict],method:str,baseline:str)->dict:
 aa={(r['seed'],r['regime']):r for r in rows if r['policy']==method}
 bb={(r['seed'],r['regime']):r for r in rows if r['policy']==baseline}
 assert aa.keys()==bb.keys()
 seeds=sorted({k[0] for k in aa});d=[];cr=[]
 for seed in seeds:
  d.append(np.mean([aa[seed,reg]['joint_delivery_5s']-bb[seed,reg]['joint_delivery_5s'] for reg in ['turn_blockage','ood']]))
  cr.append(aa[seed,'cruise']['availability']-bb[seed,'cruise']['availability'])
 d=np.array(d);low,high=ci(d)
 flips=np.array(list(itertools.product([-1,1],repeat=len(d))))@d/len(d);p=float(np.mean(np.abs(flips)>=abs(d.mean())-1e-12))
 normal=float(shapiro(d).pvalue) if np.ptp(d)>1e-12 else None
 sd=float(d.std(ddof=1));nz=d[np.abs(d)>1e-12];signp=float(binomtest(int((nz>0).sum()),len(nz)).pvalue) if len(nz) else 1.
 return dict(method=method,baseline=baseline,n_seed_clusters=len(d),mean_difference=float(d.mean()),ci_low=low,ci_high=high,
   signflip_p=p,bonferroni4_p=min(1.,p*4),sign_test_p=signp,shapiro_p=normal,dz=float(d.mean()/sd) if sd>1e-12 else None,
   cruise_availability_difference=float(np.mean(cr)),clear_advantage=bool(d.mean()>=.05 and low>0 and p<.0125 and np.mean(cr)>=-.01))

def run_policies(assets:list[Asset],names:list[str],parameters:dict,models:dict|None=None)->tuple[list[dict],list[dict]]:
 rows=[];logs=[]
 for asset in assets:
  for name in names:
   r,log=rollout(asset,name,parameters.get(name,.08),(models or {}).get(name));rows.append(r);logs.extend(log)
  print(f'Evaluated {asset.regime} {asset.seed}, {len(names)} policies',flush=True)
 return rows,logs

def main()->None:
 parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');args=parser.parse_args()
 paths=[ROOT/'direction_env.py',Path(__file__),ROOT/'protocol.md',ROOT/'export_direction_assets.m']
 hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
 if args.freeze:
  (ROOT/'implementation-freeze.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8');print('All stage algorithms frozen before policy test evaluation.');return
 frozen=json.loads((ROOT/'implementation-freeze.json').read_text());assert hashes==frozen,'Implementation changed after freeze; record and justify before evaluation'
 assert len(list((ROOT/'assets'/'train').glob('*.mat')))==8
 assert len(list((ROOT/'assets'/'validation').glob('*.mat')))==4
 assert len(list((ROOT/'assets'/'test').glob('*.mat')))==24
 train=load_assets('train');val=load_assets('validation');test=load_assets('test')
 parameters={};validation_rows=[]
 for name in STAGE1+['robust']:
  choices=[.04,.08,.16] if name in ['timing','joint','robust'] else [.08]
  best_score=-math.inf
  for parameter in choices:
   rows=[rollout(a,name,parameter)[0] for a in val]
   for r in rows:r['parameter']=parameter
   validation_rows.extend(rows);s=score(rows)
   if s>best_score:best_score=s;parameters[name]=parameter
 base=max(BASE,key=lambda name:score([r for r in validation_rows if r['policy']==name]))
 (OUT/'locked-parameters.json').write_text(json.dumps(dict(parameters=parameters,primary_baseline=base,hashes=hashes),indent=2),encoding='utf-8')
 write_csv(OUT/'validation.csv',validation_rows)
 rows,logs=run_policies(test,STAGE1,parameters)
 primary=contrast(rows,'joint',base);ablations=[contrast(rows,'joint',x) for x in ['timing','budget']]
 stage1=bool(primary['clear_advantage'] and all(x['mean_difference']>0 for x in ablations))
 decisions=dict(stage1=primary,joint_ablations=ablations,stage1_pass=stage1)
 write_csv(OUT/'test-trajectories.csv',rows);write_csv(OUT/'test-decisions.csv',logs)
 (OUT/'stage-decisions.json').write_text(json.dumps(decisions,indent=2),encoding='utf-8')
 if not stage1:
  voi=fit_voi(train);r,l=run_policies(test,['robust','voi'],parameters,{'voi':voi});rows+=r;logs+=l
  stages=[contrast(rows,name,base) for name in ['robust','voi']];decisions['stage2']=stages
  decisions['stage2_pass']=any(x['clear_advantage'] for x in stages)
  write_csv(OUT/'test-trajectories.csv',rows);write_csv(OUT/'test-decisions.csv',logs)
  (OUT/'stage-decisions.json').write_text(json.dumps(decisions,indent=2),encoding='utf-8')
  if not decisions['stage2_pass']:
   rl=train_rl(train,val);models={f'rl-{m["learning_seed"]}':m for m in rl};r,l=run_policies(test,list(models),parameters,models);rows+=r;logs+=l
   # Average learning seeds within each propagation trajectory; never multiply n.
   averaged=[]
   for key in sorted({(r['seed'],r['regime']) for r in r}):
    rr=[v for v in r if (v['seed'],v['regime'])==key];record={k:(float(np.mean([v[k] for v in rr])) if isinstance(rr[0][k],(int,float)) else rr[0][k]) for k in rr[0]};record['policy']='rl-mean';averaged.append(record)
   rows+=averaged;decisions['stage3']=contrast(rows,'rl-mean',base)
 write_csv(OUT/'test-trajectories.csv',rows);write_csv(OUT/'test-decisions.csv',logs)
 (OUT/'stage-decisions.json').write_text(json.dumps(decisions,indent=2),encoding='utf-8')
 print(json.dumps(decisions,indent=2),flush=True)

if __name__=='__main__':main()
