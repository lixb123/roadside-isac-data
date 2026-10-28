"""Causal resource-limited replay using MATLAB-exported solver candidates."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.io import loadmat

ROOT=Path(__file__).resolve().parent

def channel(d:dict,t:np.ndarray)->tuple[np.ndarray,np.ndarray]:
 c,z,meta=d['c'],d['z'],d['meta'];t=np.atleast_1d(t);lam=c['lightSpeed']/c['carrierHz'];K=10**(c['ricianKdB']/10)
 hs=np.zeros((len(t),4,6),complex);hr=np.zeros((len(t),4,8),complex)
 for u in range(4):
  pos=np.column_stack([np.interp(t,np.arange(121),meta['positions'][:,u,j]) for j in range(3)])
  delta=pos-c['initialPositions'][u]
  phase=z['phases'][:,u,None]+2*np.pi/lam*(z['dx'][:,u,None]*delta[:,0]+z['dy'][:,u,None]*delta[:,1])
  sos=np.exp(1j*phase);direct=pos-c['bsPosition'];dist=np.linalg.norm(direct,axis=1)
  db=np.zeros(len(t))
  if u==0 and meta['regime']!='cruise':
   dur=c['oodBlockageDuration'] if meta['regime']=='ood' else c['blockageDuration']
   loss=c['oodBlockageDb'] if meta['regime']=='ood' else c['blockageDb']
   db[(t>=meta['eventSeconds'])&(t<meta['eventSeconds']+dur)]=loss
  los=np.exp(1j*np.pi*np.arange(6)[None,:]*(direct[:,1]/dist)[:,None])*np.exp(-1j*2*np.pi*dist/lam)[:,None]/np.sqrt(6)
  hs[:,u]=(np.sqrt(K/(K+1))*los+np.sqrt(1/(K+1))*(z['coef'][:,:,u]@sos).T)*(c['referenceDistance']/dist*10**(-db/20))[:,None]
  refl=pos-c['irsPosition'];dr=np.linalg.norm(refl,axis=1);ref=np.linalg.norm(c['initialPositions'][u]-c['irsPosition'])
  losr=np.exp(-1j*np.pi*np.arange(8)[None,:]*(refl[:,1]/dr)[:,None])*np.exp(-1j*2*np.pi*dr/lam)[:,None]/np.sqrt(8)
  hr[:,u]=(np.sqrt(K/(K+1))*losr+np.sqrt(1/(K+1))*(z['coefIRS'][:,:,u].T@sos).T)*(.05*ref/dr)[:,None]
 return hs,hr

def metrics(hs:np.ndarray,hr:np.ndarray,Hsr:np.ndarray,W:np.ndarray,V:np.ndarray,c:dict)->dict[str,np.ndarray]:
 H=np.einsum('tun,n,nm->tum',hr,np.conj(V),Hsr)+np.conj(hs)
 A=np.abs(H@W)**2;noise=c['noiseW']/c['powerW'];rates=[];eve=[]
 for k in range(3):
  rates.append(np.min(np.log2(1+A[:,k:3,k]/(A[:,k:3,k+1:3].sum(axis=2)+noise)),axis=1))
  eve.append(np.log2(1+A[:,3,k]/(A[:,3,:].sum(axis=1)-A[:,3,k]+noise)))
 rates=np.array(rates).T;eve=np.array(eve).T
 jam=np.log2(1+A[:,:3,3]/(A[:,:3,:3].sum(axis=2)+noise))
 tol=1e-8*np.maximum(noise,A[:,:3,:].max(axis=2))
 ordering=np.all((A[:,:3,3]>=A[:,:3,0]-tol)&(A[:,:3,0]>=A[:,:3,1]-tol)&(A[:,:3,1]>=A[:,:3,2]-tol),axis=1)
 echo=c['rcs']*np.sum(np.abs(H[:,3])**2,axis=1)*A[:,3].sum(axis=1)/(c['sensingNoiseW']/c['powerW'])
 qos=np.any(rates<c['rateMin']-1e-5,axis=1);echo_fail=echo<c['echoThreshold']*(1-1e-5)
 sic=(~ordering)|np.any(jam<1-1e-5,axis=1)|np.any(np.sum(np.abs(W[:,:3])**2,axis=0)>np.sum(np.abs(W[:,3])**2)+1e-5)
 feasible=(~qos)&(~echo_fail)&(~sic)&(np.sum(np.abs(W)**2)<=1+1e-5)&(np.max(np.abs(np.abs(V)-1))<=1e-6)
 margin=np.minimum.reduce([rates.min(axis=1)/c['rateMin']-1,echo/c['echoThreshold']-1,jam.min(axis=1)-1])
 return dict(feasible=feasible,secrecy=np.maximum(rates-eve,0).sum(axis=1),rate=rates[:,0],qos=qos,echo=echo_fail,sic=sic,margin=margin,echo_value=echo)

def robust_candidate(hs:np.ndarray,hr:np.ndarray,Hsr:np.ndarray,c:dict)->tuple[np.ndarray,np.ndarray]:
 """Finite uncertain-angle/amplitude max-min codebook, not a robustness certificate."""
 V=np.ones(8,complex);H=np.einsum('un,n,nm->um',hr,np.conj(V),Hsr)+np.conj(hs)
 directions=[np.sum(H.conj().T/np.maximum(np.linalg.norm(H,axis=1),1e-10),axis=1)]
 for j in range(1,17):directions.append(np.linalg.pinv(H)@np.exp(1j*2*np.pi*np.arange(4)*j/16))
 sh=[];rh=[]
 for db in [-2.,0.,2.]:
  for angle in [-.02,0.,.02]:
   h=hs.copy();h[0]*=10**(db/20)*np.exp(1j*np.pi*np.arange(6)*angle)
   sh.append(h);rh.append(hr.copy())
 best=None;score=-np.inf
 for q in directions:
  if np.linalg.norm(q)<1e-12:continue
  W=q[:,None]/np.linalg.norm(q)*np.sqrt(c['initialPowerFractions'])[None,:]
  m=metrics(np.array(sh),np.array(rh),Hsr,W,V,c)
  value=m['margin'].min()+.001*m['secrecy'].mean()
  if value>score:score=value;best=W
 assert best is not None
 return best,V

@dataclass
class Asset:
 path:Path
 step:float=.01

 def __post_init__(self)->None:
  self.d=loadmat(self.path,simplify_cells=True);self.c=self.d['c'];self.regime=str(self.d['meta']['regime']);self.seed=int(self.path.stem.split('-')[-1])
  self.hs,self.hr=channel(self.d,np.arange(0,120,self.step));self.Hsr=self.d['z']['Hsr']
  self.W=[self.d['initial']['W']];self.V=[self.d['initial']['V']];self.accept=[True];self.actual=[0.]
  for k in range(24):
   for a in range(2):
    self.W.append(self.d['Ws'][:,:,k,a]);self.V.append(self.d['Vs'][:,k,a]);self.accept.append(bool(self.d['accepted'][k,a]));self.actual.append(float(self.d['runtime'][k,a]))
  for k in range(24):
   i=int(k*5/self.step);W,V=robust_candidate(self.hs[i],self.hr[i],self.Hsr,self.c)
   m=metrics(self.hs[i:i+1],self.hr[i:i+1],self.Hsr,W,V,self.c)
   self.W.append(W);self.V.append(V);self.accept.append(bool(m['feasible'][0]));self.actual.append(0.)
  self.cache:dict[tuple[int,int],dict]={};self.obs_cache={}
  self.verify()

 def verify(self)->None:
  ix=np.array([0,20,40,80,120]);hs,hr=channel(self.d,ix)
  m=metrics(hs,hr,self.Hsr,self.W[0],self.V[0],self.c)
  got=np.column_stack([m['secrecy'],m['feasible'],m['echo_value'],m['rate'],m['margin']])
  assert np.max(np.abs(got-self.d['testMetrics']))<1e-7, 'MATLAB/Python evaluator mismatch'
  assert all(np.isfinite(w).all() for w in self.W)

 def outcome(self,state:int,k:int)->dict:
  key=state,k
  if key not in self.cache:
   lo=round(k*5/self.step);hi=round((k+1)*5/self.step)
   self.cache[key]=metrics(self.hs[lo:hi],self.hr[lo:hi],self.Hsr,self.W[state],self.V[state],self.c)
  return self.cache[key]

 def candidate(self,k:int,a:int)->int:
  return 49+k if a==3 else 1+2*k+a-1


class Env:
 def __init__(self,asset:Asset)->None:
  self.asset=asset;self.reset()

 def reset(self)->np.ndarray:
  self.k=0;self.state=0;self.stamp=0;self.requests=0;self.compute=0.;self.rows=[];self.parts=[];return self.observe()

 def observe(self)->np.ndarray:
  a=self.asset;c=a.c;now=self.k*5;stamp=self.stamp
  if self.k==24:return np.zeros(12)
  # Only probes through now and CSI at last paid acquisition.
  probe=a.d['probes'][now];oldprobe=a.d['probes'][stamp]
  residual=np.max(np.abs(probe-oldprobe)/np.maximum(oldprobe,1e-8))
  i=round(stamp/a.step);m=metrics(a.hs[i:i+1],a.hr[i:i+1],a.Hsr,a.W[self.state],a.V[self.state],c)
  margin=float(m['margin'][0]);first=max(0,now-4);tt=np.arange(first,now+1);pdb=20*np.log10(np.maximum(a.d['probes'][tt,0],1e-8))
  drop=-float(np.polyfit(tt-tt.mean(),pdb,1)[0]) if len(tt)>1 else 0.
  pos=a.d['meta']['positions'][:now+1,0,:]
  rng=np.random.default_rng(930000+a.seed);noisy=pos+rng.normal(0,.2,pos.shape)
  velocity=np.polyfit(tt-tt.mean(),noisy[first:],1)[0] if len(tt)>1 else np.zeros(3)
  current=noisy[-1]-c['bsPosition'];future=current+5*velocity
  mu0=current[1]/np.linalg.norm(current);mu1=future[1]/np.linalg.norm(future)
  loss=1-abs(np.mean(np.exp(1j*np.pi*np.arange(6)*(mu1-mu0))))**2
  ratio=min(1.,(probe[0]/max(oldprobe[0],1e-8))**2)
  risk=margin-(1-ratio)-max(0.,drop)*.12-loss
  # Approximate seconds until proxy margin crosses zero, capped at 40.
  decay=max(.002,max(0.,drop)*.12+loss/5)
  window=np.clip(max(0.,risk)/decay,0,40)
  return np.array([now/120,(now-stamp)/40,residual,np.clip(margin,-1,5),np.clip(drop,-5,5),loss,
    np.clip(risk,-2,5),window/40,(6-self.requests)/6,(12-self.compute)/12,(self.k%4)/4,1.])

 def allowed(self)->list[int]:
  allowed=[0]
  if self.requests<6:
   if self.compute+.5<=12+1e-9:allowed.append(1)
   if self.compute+2<=12+1e-9:allowed.extend([2,3])
  return allowed

 def step(self,action:int)->tuple[np.ndarray,float,bool,dict]:
  assert action in self.allowed();a=self.asset;k=self.k;old=self.state
  costs=[0.,.5,2.,2.];delay=[0.,.7,2.2,2.2][action]
  accepted=False;candidate=old
  if action:
   self.requests+=1;self.compute+=costs[action];self.stamp=k*5;candidate=a.candidate(k,action);accepted=a.accept[candidate]
  before=a.outcome(old,k);data={key:value.copy() for key,value in before.items()}
  if accepted:
   after=a.outcome(candidate,k);cut=round(delay/a.step)
   for key in data:data[key][cut:]=after[key][cut:]
   self.state=candidate
  # Continuous 4% observation rate reduction, explicit acquisition airtime debit.
  radio=np.full(len(data['rate']),.96);debit=.6*(int(bool(action))+int(k==0));radio[:min(len(radio),round(debit/a.step))]=0
  joint=data['rate']*radio*data['feasible'];good=data['secrecy']*radio*data['feasible']
  self.parts.append(dict(joint=joint,rate=data['rate']*radio,feasible=data['feasible'],good=good,qos=data['qos'],echo=data['echo'],sic=data['sic']))
  bits=float(joint.sum()*a.step);delivered=float(bits>=1-1e-9) if k%4==0 else 0.
  reward=delivered+.1*float(data['feasible'].mean())-.015*costs[action]-.01*bool(action)
  row=dict(seed=a.seed,regime=a.regime,k=k,action=action,accepted=int(accepted),requests=self.requests,compute=self.compute,delay=delay,bits=bits,reward=reward,state=self.state)
  self.rows.append(row);self.k+=1
  return self.observe(),reward,self.k==24,row

 def result(self)->dict:
  assert self.k==24;a=self.asset
  cat=lambda key:np.concatenate([p[key] for p in self.parts])
  row=dict(seed=a.seed,regime=a.regime,availability=float(cat('feasible').mean()),goodput=float(cat('good').mean()),
    qos_violation=float(cat('qos').mean()),echo_violation=float(cat('echo').mean()),sic_violation=float(cat('sic').mean()),
    requests=self.requests,compute=self.compute,limited=sum(r['action']==1 for r in self.rows),joint=sum(r['action'] in [2,3] for r in self.rows),rejected=sum(r['action']>0 and not r['accepted'] for r in self.rows))
  for deadline in [1,5,20]:
   count=[];radio=[]
   for k in range(0,24,4):
    count.append(cat('joint')[round(k*5/a.step):round((k*5+deadline)/a.step)].sum()*a.step>=1-1e-9)
    radio.append(cat('rate')[round(k*5/a.step):round((k*5+deadline)/a.step)].sum()*a.step>=1-1e-9)
   row[f'joint_delivery_{deadline}s']=float(np.mean(count));row[f'delivery_{deadline}s']=float(np.mean(radio))
  return row

def load_assets(split:str)->list[Asset]:
 return [Asset(p) for p in sorted((ROOT/'assets'/split).glob('*.mat'))]
