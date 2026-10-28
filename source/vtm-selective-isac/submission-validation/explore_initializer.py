"""Historical train/validation pilot only; no independent test trajectories."""
from pathlib import Path
import sys,time,json,csv
import numpy as np
from scipy.io import loadmat,savemat
from sklearn.ensemble import ExtraTreesClassifier
import joblib

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'direction-validation'))
from direction_env import channel,metrics

def candidates(hs,hr,Hsr,c):
    V=np.ones(8,complex)
    H=np.einsum('un,n,nm->um',hr,np.conj(V),Hsr)+np.conj(hs)
    ref=np.ones((6,1))/np.sqrt(6)*np.sqrt(c['initialPowerFractions'])[None,:]
    qs=[np.sum(H.conj().T/np.maximum(np.linalg.norm(H,axis=1),1e-15),axis=1)]
    inv=np.linalg.pinv(H)
    qs += [inv@np.exp(1j*2*np.pi*np.arange(4)*j/16) for j in range(1,17)]
    Ws=[ref]+[q[:,None]/max(np.linalg.norm(q),1e-15)*np.sqrt(c['initialPowerFractions'])[None,:] for q in qs]
    return H,Ws,V

def feature(hs,hr,Hsr,c):
    H=np.einsum('un,n,nm->um',hr,np.ones(8),Hsr)+np.conj(hs)
    return np.r_[H.real.ravel(),H.imag.ravel(),np.linalg.norm(H,axis=1)]

def evaluate(hs,hr,Hsr,W,V,c):
    m=metrics(hs[None],hr[None],Hsr,W,V,c)
    H=np.einsum('un,n,nm->um',hr,np.conj(V),Hsr)+np.conj(hs)
    A=np.abs(H@W)**2;noise=c['noiseW']/c['powerW']
    jam=np.log2(1+A[:3,3]/(A[:3,:3].sum(axis=1)+noise))
    jmin=1+c['optimizationJamMargin']
    margin=min(float(m['rate'][0])/c['rateMin']-1, float(m['echo_value'][0])/c['echoThreshold']-1,float(jam.min())/jmin-1)
    # m['rate'] is stream one; include every legitimate stream in the margin.
    rates=[min(np.log2(1+A[k:3,k]/(A[k:3,k+1:3].sum(axis=1)+noise))) for k in range(3)]
    margin=min(margin,min(rates)/c['rateMin']-1)
    return bool(m['feasible'][0] and np.all(jam>=jmin-1e-5)),margin

def choose_scan(hs,hr,Hsr,c):
    _,Ws,V=candidates(hs,hr,Hsr,c)
    scores=[evaluate(hs,hr,Hsr,w,V,c) for w in Ws]
    best=max(range(len(Ws)),key=lambda j:(scores[j][0],scores[j][1],-j))
    return Ws,V,scores,best

def run():
    old=ROOT.parent/'direction-validation/assets';out=ROOT/'initializer-pilot';out.mkdir(exist_ok=True)
    X=[];Y=[]
    for path in sorted((old/'train').glob('*.mat')):
        d=loadmat(path,simplify_cells=True);c=d['c'].copy()
        c.update(rateMin=c['rateMin']+c['optimizationRateMargin'],echoThreshold=c['echoThreshold']*c['optimizationEchoFactor'])
        hs,hr=channel(d,np.arange(0,120,20))
        for h,r in zip(hs,hr):
            _,_,_,best=choose_scan(h,r,d['z']['Hsr'],c)
            X.append(feature(h,r,d['z']['Hsr'],c));Y.append(best)
    fit=time.perf_counter();model=ExtraTreesClassifier(n_estimators=96,max_depth=8,min_samples_leaf=2,random_state=31,n_jobs=1).fit(X,Y);fit=time.perf_counter()-fit
    joblib.dump(model,out/'selector.joblib');rows=[]
    for path in sorted((old/'validation').glob('*.mat')):
        d=loadmat(path,simplify_cells=True);c=d['c'].copy();c.update(rateMin=c['rateMin']+c['optimizationRateMargin'],echoThreshold=c['echoThreshold']*c['optimizationEchoFactor'])
        hs,hr=channel(d,np.arange(0,120,20));Wout=np.zeros((6,4,6,3),complex);chosen=np.zeros((6,3),int)
        for epoch,(h,r) in enumerate(zip(hs,hr)):
            Hsr=d['z']['Hsr']
            for method in range(3):
                start=time.perf_counter();fallback=False
                if method==0:
                    Ws,V,ss,ix=choose_scan(h,r,Hsr,c)
                else:
                    ix=int(model.predict(feature(h,r,Hsr,c)[None,:])[0]) if method==1 else 1
                    _,Ws,V=candidates(h,r,Hsr,c)
                    # Avoid crediting common codebook construction as a learned reduction.
                    feasible,margin=evaluate(h,r,Hsr,Ws[ix],V,c)
                    if not feasible:
                        Ws,V,ss,ix=choose_scan(h,r,Hsr,c);fallback=True
                elapsed=time.perf_counter()-start;feasible,margin=evaluate(h,r,Hsr,Ws[ix],V,c)
                Wout[:,:,epoch,method]=Ws[ix];chosen[epoch,method]=ix
                rows.append(dict(trajectory=path.stem,epoch=epoch,method=['scan','learned','simple'][method],index=ix,feasible=int(feasible),margin=margin,time_s=elapsed,fallback=int(fallback)))
        savemat(out/(path.stem+'.mat'),dict(Winit=Wout,chosen=chosen,source=str(path)))
    with (out/'initializers.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/'training.json').write_text(json.dumps(dict(states=len(Y),label_counts=np.bincount(Y,minlength=18).tolist(),fit_s=fit,features=len(X[0]),scope='historical validation pilot; Python uses all design margins; MATLAB independently rechecks constraints'),indent=2))
    print('Initializer pilot prepared',len(rows),'rows',flush=True)

if __name__=='__main__':run()
