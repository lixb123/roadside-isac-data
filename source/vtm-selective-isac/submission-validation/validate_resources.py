"""Frozen-model prospective screening with recorded full candidate costs."""
from pathlib import Path
import sys,time,json,hashlib,argparse,csv
import numpy as np
from scipy.io import loadmat
from scipy.stats import beta
ROOT=Path(__file__).resolve().parent
PREV=ROOT.parent/'budget-ai-confirmation'
sys.path.insert(0,str(PREV))
from confirmation import Asset,Env,causal_features,load_model,write_csv
from direction_env import channel

class LocalAsset(Asset):
 def __post_init__(self):
  # Preserve physics/verification; omit unused robust and joint codebook evaluation.
  self.d=loadmat(self.path,simplify_cells=True);self.c=self.d['c'];self.regime=str(self.d['meta']['regime']);self.seed=int(self.path.stem.split('-')[-1])
  self.hs,self.hr=channel(self.d,np.arange(0,120,self.step));self.Hsr=self.d['z']['Hsr']
  self.W=[self.d['initial']['W']];self.V=[self.d['initial']['V']];self.accept=[True];self.actual=[0.]
  for k in range(24):
   for a in range(2):
    self.W.append(self.d['Ws'][:,:,k,a]);self.V.append(self.d['Vs'][:,k,a]);self.accept.append(bool(self.d['accepted'][k,a]));self.actual.append(float(self.d['runtime'][k,a]))
  self.cache={};self.obs_cache={};self.verify()

class TimedEnv(Env):
 def __init__(self,asset):
  self.pending=None;self.decision_s=0.;self.solver_s=0.;self.timeouts=0
  super().__init__(asset)

 def allowed(self):
  return [0,1] if self.pending is None and self.requests<6 and self.compute<12-1e-9 else [0]

 def timed_step(self,action,overhead):
  a=self.asset;k=self.k;now=5*k;old=self.state;accepted=False;launch=bool(action)
  remaining=max(0,12-self.compute);decision_used=min(remaining,overhead)
  self.compute+=decision_used;self.decision_s+=overhead;finish=now+overhead
  if overhead>remaining:
   action=0;launch=False;self.timeouts+=1
  if action:
   self.requests+=1;self.stamp=now;candidate=a.candidate(k,1)
   actual=a.actual[candidate];used=min(max(0,12-self.compute),actual)
   self.compute+=used;self.solver_s+=used;timedout=used+1e-9<actual
   self.timeouts+=int(timedout)
   self.pending=dict(finish=finish+.2+used,state=candidate,accepted=a.accept[candidate] and not timedout)
  data={key:v.copy() for key,v in a.outcome(old,k).items()}
  if self.pending is not None and self.pending['finish']<=now+5+1e-9:
   accepted=self.pending['accepted']
   if accepted:
    candidate=self.pending['state'];after=a.outcome(candidate,k)
    cut=int(np.clip(np.ceil((self.pending['finish']-now)/a.step-1e-10),0,len(data['rate'])))
    for key in data:data[key][cut:]=after[key][cut:]
    self.state=candidate
   self.pending=None
  radio=np.full(len(data['rate']),.96);debit=.6*(int(launch)+int(k==0));radio[:min(len(radio),round(debit/a.step))]=0
  self.parts.append(dict(joint=data['rate']*radio*data['feasible'],rate=data['rate']*radio,feasible=data['feasible'],good=data['secrecy']*radio*data['feasible'],qos=data['qos'],echo=data['echo'],sic=data['sic']))
  self.rows.append(dict(seed=a.seed,regime=a.regime,k=k,action=action,accepted=int(accepted),requests=self.requests,compute=self.compute,decision_s=overhead,state=self.state,busy=int(self.pending is not None)))
  self.k+=1

def action(env,name,model):
 if env.k%4:return 0
 if env.allowed()==[0]:return 0
 x=env.observe()
 if name=='periodic-local':return 1 if 1 in env.allowed() else 0
 if name=='sparse-local':return int((x[1]*40>=120 or x[2]>=.32 or x[3]<.08) and 1 in env.allowed())
 features=causal_features(env,True)
 values=model['regressor'].predict(features[None])[0]
 return max(env.allowed(),key=lambda v:values[v])

def freeze():
 paths=[ROOT/'protocol.md',ROOT/'export_submission_assets.m',ROOT/'validate_resources.py',ROOT/'explore_initializer.py',PREV/'confirmation.py',ROOT.parent/'direction-validation/direction_env.py',PREV/'models/selected.json']
 paths += list((PREV/'models').glob('full-d8-l24-s*.joblib'))
 paths += [ROOT.parent/'solve_inherited.m',ROOT.parent/'evaluate_isac.m',ROOT.parent/'mobility-stress/mobility_initializer.m']
 (ROOT/'pretest-freeze.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2))
 (ROOT/'service-margin.json').write_text(json.dumps(dict(margin=.01,origin='proposed engineering screen; no user response yet; not a service standard',sensitivities=[.01,.02,.05]),indent=2))
 print('Prospective code/models/protocol frozen before scoring')

def run(step,replay=None):
 for p,h in json.loads((ROOT/'pretest-freeze.json').read_text()).items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
 paths=sorted((ROOT/'assets/test').glob('*.mat'));assert len(paths)==72
 out=ROOT/f'results-{step:g}';out.mkdir(exist_ok=True);models={};loads={}
 for s in [31,47,59,71,89]:
  t=time.perf_counter();models[f'ai-{s}']=load_model(PREV/f'models/full-d8-l24-s{s}.joblib');loads[f'ai-{s}']=time.perf_counter()-t
 rows=[];logs=[];prior={};prior_setup={}
 if replay:
  for r in csv.DictReader((Path(replay)/'decisions.csv').open()):prior[(int(r['seed']),r['regime'],r['policy'],int(r['k']))]=float(r['decision_s'])
  for r in csv.DictReader((Path(replay)/'trajectories.csv').open()):prior_setup[(int(r['seed']),r['regime'],r['policy'])]=float(r['setup_s'])
 for path in paths:
  asset=LocalAsset(path,step=step)
  for name in ['periodic-local','sparse-local']+list(models):
   env=TimedEnv(asset)
   while env.k<24:
    t=time.perf_counter();choice=action(env,name,models.get(name));overhead=time.perf_counter()-t
    if replay:overhead=prior[(asset.seed,asset.regime,name,env.k)]
    env.timed_step(choice,overhead)
   r=env.result();r.update(policy=name,decision_s=env.decision_s,solver_s=env.solver_s,setup_s=float(asset.d['setupTime'])+loads.get(name,0.),timeouts=env.timeouts)
   if replay:r['setup_s']=prior_setup[(asset.seed,asset.regime,name)]
   r['total_compute_s']=r['decision_s']+r['solver_s']+r['setup_s'];rows.append(r)
   logs += [dict(policy=name,**v) for v in env.rows]
  write_csv(out/'trajectories.csv',rows);write_csv(out/'decisions.csv',logs)
  print('Resource screening',step,asset.regime,asset.seed,flush=True);del asset
 (ROOT/'asset-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['freeze','run']);p.add_argument('--step',type=float,default=.0025);p.add_argument('--replay');a=p.parse_args()
 if a.mode=='freeze':freeze()
 else:run(a.step,a.replay)
