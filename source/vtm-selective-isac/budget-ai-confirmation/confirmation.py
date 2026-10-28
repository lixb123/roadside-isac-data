"""Isolated follow-up using frozen historical physics, with causal learned decisions."""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
import joblib

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'direction-validation'
sys.path.insert(0,str(OLD))
from direction_env import Asset,Env,metrics
from run_directions import choose,trigger,state_key,write_csv

SEEDS=[31,47,59,71,89]
BASELINES=['hold','guard','periodic20','periodic-local','budget','joint','paced-trigger']

class ReplayEnv(Env):
 """Parameterize declared deployment delays without modifying frozen physics."""
 def __init__(self,asset:Asset,delays:tuple[float,float]=(0.7,2.2))->None:
  self.delays=delays
  super().__init__(asset)

 def step(self,action:int)->tuple[np.ndarray,float,bool,dict]:
  assert action in self.allowed() and action<3
  a=self.asset;k=self.k;old=self.state;cost=[0.,.5,2.][action]
  delay=0. if not action else self.delays[action-1]
  assert delay<5,'Cross-opportunity jobs require a pending-job simulator'
  accepted=False
  if action:
   self.requests+=1;self.compute+=cost;self.stamp=k*5
   candidate=a.candidate(k,action);accepted=a.accept[candidate]
  before=a.outcome(old,k);data={key:value.copy() for key,value in before.items()}
  if accepted:
   after=a.outcome(candidate,k);cut=round(delay/a.step)
   for key in data:data[key][cut:]=after[key][cut:]
   self.state=candidate
  radio=np.full(len(data['rate']),.96);debit=.6*(int(bool(action))+int(k==0));radio[:min(len(radio),round(debit/a.step))]=0
  joint=data['rate']*radio*data['feasible'];good=data['secrecy']*radio*data['feasible']
  self.parts.append(dict(joint=joint,rate=data['rate']*radio,feasible=data['feasible'],good=good,qos=data['qos'],echo=data['echo'],sic=data['sic']))
  bits=float(joint.sum()*a.step);delivered=float(bits>=1-1e-9) if k%4==0 else 0.
  reward=delivered+.1*float(data['feasible'].mean())-.015*cost-.01*bool(action)
  row=dict(seed=a.seed,regime=a.regime,k=k,action=action,accepted=int(accepted),requests=self.requests,compute=self.compute,delay=delay,bits=bits,reward=reward,state=self.state)
  self.rows.append(row);self.k+=1
  return self.observe(),reward,self.k==24,row

def policy_action(env:Env,name:str)->int:
 x=env.observe();now=env.k*5
 if name=='hold':return 0
 if name=='periodic-local':return 1 if now%20==0 else 0
 if name=='paced-trigger':
  # At most one request per 20-second business interval, using only past usage.
  return (1 if x[7]*40<5 else 2) if trigger(x,.08) and env.requests<1+now//20 else 0
 return choose(x,name,.16 if name=='joint' else .08)

def causal_features(env:Env,expanded:bool=True)->np.ndarray:
 x=env.observe()
 if not expanded:return x
 a=env.asset;now=env.k*5;lo=max(0,now-19);p=np.maximum(a.d['probes'][lo:now+1],1e-8)
 ref=np.maximum(a.d['probes'][0],1e-8);norm=p/ref
 # Only recorded scalar power probes and prior action outcomes; no future truth.
 extra=np.r_[norm[-1],norm.mean(axis=0),norm.std(axis=0),
             int(env.rows[-1]['accepted']) if env.rows else 1,
             sum(r['accepted'] for r in env.rows)/max(1,sum(bool(r['action']) for r in env.rows))]
 return np.r_[x,extra]

def fork(env:Env)->ReplayEnv:
 other=copy.copy(env)
 # outcome arrays are immutable, whereas rows and parts lists must be independent.
 other.rows=env.rows.copy();other.parts=env.parts.copy()
 return other

def counterfactual(env:ReplayEnv,action:int)->tuple[float,np.ndarray]:
 branch=fork(env);start=branch.k
 for j in range(4):branch.step(action if j==0 else 0)
 delivered=int(branch.rows[start]['bits']>=1-1e-9)
 availability=np.mean([p['feasible'].mean() for p in branch.parts[start:]])
 # Correctly value the 20-second service period, with the same main delivery target.
 score=delivered+.1*availability-.015*[0.,.5,2.][action]-.01*bool(action)
 return float(score),causal_features(branch) if branch.k<24 else np.zeros(26)

def load_model(path:Path)->dict:
 return joblib.load(path)

def learned_action(env:Env,model:dict)->int:
 if env.k%4:return 0
 x=causal_features(env,model['expanded']);values=model['regressor'].predict(x[None,:])[0]
 allowed=[a for a in env.allowed() if a<3]
 return max(allowed,key=lambda a:values[a])

def rollout(asset:Asset,name:str,model:dict|None=None,delays:tuple[float,float]=(0.7,2.2),schedule:list[int]|None=None)->tuple[dict,list[dict]]:
 env=ReplayEnv(asset,delays);inference=[]
 while env.k<24:
  start=time.perf_counter_ns()
  if schedule is not None:action=schedule[env.k]
  elif name.startswith('ai-'):
   assert model is not None
   action=learned_action(env,model)
  elif name.startswith('qold-'):
   assert model is not None
   allowed=[v for v in env.allowed() if v<3];q=model['q'].get(state_key(env.observe()),np.zeros(3));action=max(allowed,key=lambda v:q[v])
  else:action=policy_action(env,name)
  inference.append((time.perf_counter_ns()-start)/1e6)
  if action not in env.allowed():action=0
  env.step(action)
 row=env.result();row['policy']=name;row['inference_ms_mean']=float(np.mean(inference));row['inference_ms_max']=float(np.max(inference))
 req=[r['k']*5 for r in env.rows if r['action']]
 row['last_request_s']=float(req[-1]) if req else -1.;row['sixth_request_s']=float(req[5]) if len(req)>5 else -1.
 row['accepted_requests']=sum(r['accepted'] for r in env.rows)
 return row,[dict(policy=name,**r) for r in env.rows]

def build_training()->None:
 out=ROOT/'models';out.mkdir(exist_ok=True);xx=[];yy=[];small=[];meta=[]
 for path in sorted((OLD/'assets/train').glob('*.mat')):
  a=Asset(path,step=.01);rng=np.random.default_rng(501000+a.seed)
  # 48 random full trajectories supply varying incumbents and measurement histories.
  for episode in range(48):
   env=ReplayEnv(a)
   for epoch in range(6):
    xx.append(causal_features(env));small.append(causal_features(env,False))
    outcomes=[counterfactual(env,v)[0] for v in [0,1,2]];yy.append(outcomes)
    meta.append(dict(seed=a.seed,regime=a.regime,episode=episode,epoch=epoch,best=int(np.argmax(outcomes))))
    # Include deterministic endpoint behaviors so persistent hold/local/joint are covered.
    action=episode if episode<3 else int(rng.integers(3))
    for j in range(4):env.step(action if j==0 else 0)
  print(f'Training labels {a.regime} {a.seed}: {len(yy)} states',flush=True);del a
 X=np.array(xx);Y=np.array(yy);S=np.array(small)
 np.savez(out/'counterfactual-training.npz',X=X,Y=Y,small=S)
 write_csv(out/'training-states.csv',meta)
 counts=np.bincount(np.argmax(Y,axis=1),minlength=3)
 (out/'training-summary.json').write_text(json.dumps(dict(states=len(X),expanded_features=X.shape[1],base_features=S.shape[1],argmax_counts=counts.tolist()),indent=2))
 validation=[];candidates=[]
 for expanded in [True,False]:
  for depth in [4,8]:
   for leaf in [8,24]:
    for seed in SEEDS:
     reg=ExtraTreesRegressor(n_estimators=96,max_depth=depth,min_samples_leaf=leaf,random_state=seed,n_jobs=1)
     reg.fit(X if expanded else S,Y)
     model=dict(regressor=reg,expanded=expanded,depth=depth,leaf=leaf,seed=seed)
     key=f'{"full" if expanded else "base"}-d{depth}-l{leaf}-s{seed}'
     path=out/f'{key}.joblib';joblib.dump(model,path);candidates.append((key,path,expanded,seed,depth,leaf))
 for path in sorted((OLD/'assets/validation').glob('*.mat')):
  a=Asset(path,step=.005)
  for key,file,expanded,seed,depth,leaf in candidates:
   row,_=rollout(a,'ai-'+key,load_model(file));row.update(model_key=key,expanded=int(expanded),learning_seed=seed,depth=depth,leaf=leaf);validation.append(row)
  print(f'Validation {a.regime} {a.seed}',flush=True);del a
 write_csv(out/'validation.csv',validation)
 chosen={}
 for family in ['full','base']:
  configs=[]
  for depth in [4,8]:
   for leaf in [8,24]:
    rr=[r for r in validation if r['expanded']==int(family=='full') and r['depth']==depth and r['leaf']==leaf]
    # Single configuration chosen across all seeds; never pick a lucky learning seed.
    score=np.mean([r['joint_delivery_5s'] for r in rr if r['regime']!='cruise'])+.001*np.mean([r['availability'] for r in rr])-.0001*np.mean([r['compute'] for r in rr])
    configs.append((float(score),-depth,leaf))
  score,negdepth,leaf=max(configs);chosen[family]=dict(depth=-negdepth,leaf=leaf,validation_score=score,models=[f'{family}-d{-negdepth}-l{leaf}-s{s}.joblib' for s in SEEDS])
 (out/'selected.json').write_text(json.dumps(chosen,indent=2))
 freeze={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'confirmation.py',ROOT/'protocol.md',ROOT/'export_confirmation_assets.m',out/'selected.json']+[out/f for v in chosen.values() for f in v['models']]}
 freeze.update({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [OLD/'direction_env.py',OLD/'run_directions.py']})
 (ROOT/'test-freeze.json').write_text(json.dumps(freeze,indent=2))
 print('Models selected and frozen before independent evaluation',chosen,flush=True)

def evaluate(step:float)->None:
 frozen=json.loads((ROOT/'test-freeze.json').read_text());assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in frozen.items())
 selected=json.loads((ROOT/'models/selected.json').read_text());out=ROOT/f'results-{step:g}';out.mkdir(exist_ok=True)
 models={}
 for family,config in selected.items():
  for seed,name in zip(SEEDS,config['models']):models[f'ai-{family}-{seed}']=load_model(ROOT/'models'/name)
 for seed in SEEDS:
  raw=json.loads((OLD/f'results/rl-model-{seed}.json').read_text());models[f'qold-{seed}']=dict(q={k:np.array(v) for k,v in raw['q'].items()})
 paths=sorted((ROOT/'assets/test').glob('*.mat'));assert len(paths)==24
 rows=[];logs=[];diagnostics=[]
 for path in paths:
  asset=Asset(path,step=step);stored={}
  for name in BASELINES+list(models):
   row,log=rollout(asset,name,models.get(name));rows.append(row);logs.extend(log);stored[name]=log
  # Replay identical actions with intervention on delay only; feedback cannot change actions.
  for source in ['budget','joint','periodic20','periodic-local']:
   schedule=[r['action'] for r in stored[source]]
   for tag,delays in [('zero-delay',(0.,0.)),('equal-fast',(.7,.7)),('double-delay',(1.4,4.4))]:
    row,_=rollout(asset,source+'-'+tag,delays=delays,schedule=schedule);diagnostics.append(row)
  write_csv(out/'test-trajectories.csv',rows);write_csv(out/'test-decisions.csv',logs);write_csv(out/'delay-diagnostics.csv',diagnostics)
  print(f'Independent scoring {step:g}: {asset.regime} {asset.seed}',flush=True);del asset
 print('Independent evaluation complete',flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['train','evaluate']);parser.add_argument('--step',type=float,default=.0025);args=parser.parse_args()
 if args.mode=='train':build_training()
 else:evaluate(args.step)
