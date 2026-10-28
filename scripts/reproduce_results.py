"""Recompute manuscript results from frozen data. No solver, training, or new timing."""
from pathlib import Path
import argparse, csv, hashlib, json, sys
import numpy as np
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
REGIMES = ['cruise', 'turn_blockage', 'ood']
POLICIES = ['periodic-local', 'sparse-local', 'ai-mean']
METRICS = ['joint_delivery_5s', 'availability', 'requests', 'total_compute_s',
           'decision_s', 'solver_s', 'setup_s', 'timeouts']

def read(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            try: r[k] = float(v)
            except (ValueError, TypeError): pass
    return rows

def write(path, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def boot(x):
    x = np.asarray(x); rng = np.random.default_rng(20261024)
    return np.quantile(x[rng.integers(len(x), size=(20000, len(x)))].mean(axis=1), [.025, .975]).tolist()

def compare(a, b, label):
    if isinstance(a, dict):
        assert set(a) == set(b), label
        for k in a: compare(a[k], b[k], label + '/' + str(k))
    elif isinstance(a, list):
        assert len(a) == len(b), label
        for i, (x, y) in enumerate(zip(a, b)): compare(x, y, label + '/' + str(i))
    elif isinstance(a, (float, int)) and not isinstance(a, bool):
        assert np.isclose(a, b, rtol=1e-10, atol=1e-11), (label, a, b)
    else: assert a == b, (label, a, b)

def main(out, mat_check):
    out.mkdir(parents=True, exist_ok=True)
    checked = 0
    for line in (ROOT/'checksums.sha256').read_text().splitlines():
        h, name = line.split('  ', 1)
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == h, name
        checked += 1
    for item in json.loads((ROOT/'metadata/frozen-inputs.json').read_text()):
        assert hashlib.sha256((ROOT/item['file']).read_bytes()).hexdigest()==item['sha256'], item['file']
    training=np.load(ROOT/'models/counterfactual-training.npz')
    assert training['X'].shape==(2304,26) and training['Y'].shape==(2304,3) and training['small'].shape==(2304,12)
    rows = read(ROOT/'data/main/grid-0.0025/trajectories.csv')
    fine = read(ROOT/'data/main/grid-0.00125/trajectories.csv')
    assert len(rows) == len(fine) == 504
    keys = lambda r: (int(r['seed']), r['regime'], r['policy'])
    assert len({keys(r) for r in rows}) == 504
    seeds = sorted({int(r['seed']) for r in rows}); assert seeds == list(range(16101, 16125))
    normalized = [r for r in rows if not r['policy'].startswith('ai-')]
    for s in seeds:
        for regime in REGIMES:
            rr = [r for r in rows if r['seed'] == s and r['regime'] == regime and r['policy'].startswith('ai-')]
            assert {r['policy'] for r in rr} == {'ai-'+str(s) for s in [31,47,59,71,89]}
            normalized.append(dict(seed=s, regime=regime, policy='ai-mean',
                                   **{m:float(np.mean([r[m] for r in rr])) for m in METRICS}))
    summary = []
    for p in POLICIES:
        for regime in REGIMES + ['all']:
            rr = [r for r in normalized if r['policy']==p and (regime=='all' or r['regime']==regime)]
            summary.append(dict(policy=p, regime=regime, **{m:float(np.mean([r[m] for r in rr])) for m in METRICS}))
    def cluster(p,m,adverse=False):
        return np.array([np.mean([r[m] for r in normalized if r['seed']==s and r['policy']==p and (not adverse or r['regime']!='cruise')]) for s in seeds])
    comparisons = []; paired = []; resource = []
    for p in POLICIES:
        x=cluster(p,'total_compute_s'); ci=boot(x)
        resource.append(dict(policy=p, compute_mean_s=float(x.mean()), ci_low_s=ci[0], ci_high_s=ci[1]))
    for p in POLICIES[:2]:
        d=cluster('ai-mean','joint_delivery_5s',True)-cluster(p,'joint_delivery_5s',True)
        loss=np.maximum(0,-d); k=int(np.sum(loss>1e-10)); upper=float(beta.ppf(.975,k+1,24-k)) if k<24 else 1.
        saving=cluster(p,'total_compute_s')-cluster('ai-mean','total_compute_s')
        # The observed vectors have a common positive sign. Only the two extreme
        # assignments attain the absolute observed sum among 2**24 sign flips.
        assert np.all(saving>1e-10)
        flip=2.0/2**len(saving)
        comparisons.append(dict(baseline=p, clusters=24, adverse_delivery_difference=float(d.mean()),
            delivery_difference_ci=boot(d), observed_mean_nonnegative_loss=float(loss.mean()),
            seeds_with_loss=k, simultaneous_loss_upper=upper, NI_1pp=upper<.01,
            NI_2pp=upper<.02, NI_5pp=upper<.05, compute_saved_s=float(saving.mean()),
            compute_saved_ci=boot(saving), relative_compute_saving=float(saving.mean()/cluster(p,'total_compute_s').mean()),
            signflip_p=flip, sign_test_p=flip, compute_difference_sd=float(saving.std(ddof=1)),
            paired_standardized_effect=float(saving.mean()/saving.std(ddof=1))))
        for s,c,v in zip(seeds,saving,d): paired.append(dict(seed=s,baseline=p,compute_saved_s=float(c),adverse_delivery_difference=float(v)))
    order=np.argsort([c['signflip_p'] for c in comparisons]); last=0.
    for rank,j in enumerate(order):
        last=max(last,min(1.,(2-rank)*comparisons[j]['signflip_p'])); comparisons[j]['holm_p']=last
    reference=json.loads((ROOT/'reference/statistics.json').read_text())
    compare(summary,reference['summary'],'main-summary'); compare(comparisons,reference['comparisons'],'paired-statistics')
    learning=[]
    for p in sorted({r['policy'] for r in rows if r['policy'].startswith('ai-')}):
        rr=[r for r in rows if r['policy']==p]
        learning.append(dict(policy=p,requests=float(np.mean([r['requests'] for r in rr])),compute=float(np.mean([r['total_compute_s'] for r in rr])),adverse_delivery=float(np.mean([r['joint_delivery_5s'] for r in rr if r['regime']!='cruise']))))
    compare(learning,reference['learning_seeds'],'learning-seeds')
    for r in rows:
        compare(r['total_compute_s'],r['decision_s']+r['solver_s']+r['setup_s'],'compute-accounting')
    fine_map={keys(r):r for r in fine}; assert set(fine_map)=={keys(r) for r in rows}
    grid={m:max(abs(r[m]-fine_map[keys(r)][m]) for r in rows) for m in ['joint_delivery_1s','joint_delivery_5s','joint_delivery_20s','availability','goodput','requests','total_compute_s']}
    dec=read(ROOT/'data/main/grid-0.0025/decisions.csv'); decfine=read(ROOT/'data/main/grid-0.00125/decisions.csv')
    dk=lambda r:(*keys(r),int(r['k']))
    dm={dk(r):r for r in dec}; fm={dk(r):r for r in decfine}
    assert len(dec)==len(decfine)==len(dm)==len(fm)==12096 and set(dm)==set(fm)
    grid.update(changed_actions=sum(dm[k]['action']!=fm[k]['action'] for k in dm),changed_acceptance=sum(dm[k]['accepted']!=fm[k]['accepted'] for k in dm),trajectory_rows_per_grid=len(rows),decision_rows_per_grid=len(dec))
    compare(grid,json.loads((ROOT/'reference/grid-check.json').read_text()),'grid')
    hist=read(ROOT/'data/historical/guarded-hold-grid-0.0025.csv'); assert len(hist)==18
    hist_summary=[]
    for s in [1,2,3]:
        rr=[r for r in hist if r['scenario']==s]; assert len(rr)==6
        h=dict(scenario=s,**{m:float(np.mean([r[m] for r in rr])) for m in ['availability','acquired_feasible_aged_failure','unresolved_failure']})
        assert np.isclose(sum(h[k] for k in h if k!='scenario'),1)
        hist_summary.append(h)
    compare(hist_summary,json.loads((ROOT/'reference/historical-diagnosis.json').read_text()),'historical')
    pilot=read(ROOT/'data/initializer/solver-validation.csv'); assert len(pilot)==72
    init={}
    initref=json.loads((ROOT/'reference/initializer-summary.json').read_text())
    for method in ['scan','learned','simple']:
        rr=[r for r in pilot if r['method']==method]; assert len(rr)==24
        init[method]={m:float(np.mean([r[m] for r in rr])) for m in initref[method] if m!='snapshots'}
        init[method]['snapshots']=len(rr)
    compare(init,initref,'initializer')
    write(out/'table-1.csv',[dict(regime=r['regime'],policy=r['policy'],compute_s=r['total_compute_s'],delivery_percent=100*r['joint_delivery_5s']) for r in summary if r['regime']!='all'])
    write(out/'main-summary.csv',summary); write(out/'paired-seed-differences.csv',paired)
    write(out/'figure-4-data.csv',resource); write(out/'figure-5-data.csv',[r for r in summary if r['regime']!='all']); write(out/'figure-6-data.csv',hist_summary)
    result=dict(summary=summary,comparisons=comparisons,learning_seeds=learning,grid_check=grid,initializer=init,historical=hist_summary)
    (out/'recomputed-results.json').write_text(json.dumps(result,indent=2)+'\n')
    if mat_check:
        from scipy.io import loadmat
        sys.path.insert(0,str(ROOT/'source/vtm-selective-isac/direction-validation'))
        from direction_env import channel, metrics
        assets=list((ROOT/'data/main/assets').glob('*.mat')); assert len(assets)==72
        timings=read(ROOT/'data/main/candidate-timings.csv');assert len(timings)==432
        timer_map={(int(r['seed']),r['regime'],int(r['k'])):r for r in timings};assert len(timer_map)==432
        errors=[]
        for p in assets:
            d=loadmat(p,simplify_cells=True); hs,hr=channel(d,np.array([0,20,40,80,120]))
            m=metrics(hs,hr,d['z']['Hsr'],d['initial']['W'],d['initial']['V'],d['c'])
            got=np.column_stack([m['secrecy'],m['feasible'],m['echo_value'],m['rate'],m['margin']])
            errors.append(float(np.max(np.abs(got-d['testMetrics']))))
            for k in range(0,24,4):
                timer=timer_map[(int(p.stem.split('-')[-1]),str(d['meta']['regime']),k)]
                compare(timer['solve_validation_s'],float(d['runtime'][k,0]),'candidate-timer')
                compare(timer['accepted'],int(d['accepted'][k,0]),'candidate-acceptance')
                compare(timer['failed'],int(d['failed'][k,0]),'candidate-failure')
                compare(timer['common_setup_s'],float(d['setupTime']),'candidate-setup')
        assert max(errors)<1e-7
        result['mat_reference_max_abs_error']=max(errors)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=['#596d7c','#c48532','#168078']; labels=['Periodic','Sparse','Learned']
    fig,ax=plt.subplots(figsize=(5,3)); v=np.array([r['compute_mean_s'] for r in resource]); ax.bar(range(3),v,color=colors)
    ax.errorbar(range(3),v,yerr=np.array([[r['compute_mean_s']-r['ci_low_s'] for r in resource],[r['ci_high_s']-r['compute_mean_s'] for r in resource]]),fmt='none',color='black',capsize=3); ax.set_xticks(range(3),labels); ax.set_ylabel('Control computation per mission (s)'); fig.tight_layout(); fig.savefig(out/'figure-4.png',dpi=200); plt.close(fig)
    fig,ax=plt.subplots(figsize=(5,3));x=np.arange(3)
    for i,p in enumerate(POLICIES): ax.bar(x+(i-1)*.24,[100*next(r['joint_delivery_5s'] for r in summary if r['policy']==p and r['regime']==reg) for reg in REGIMES],.23,color=colors[i],label=labels[i])
    ax.set_xticks(x,['Cruise','Turn/block','OOD']);ax.set_ylabel('Joint-compliant 5 s delivery (%)');ax.legend(frameon=False);fig.tight_layout();fig.savefig(out/'figure-5.png',dpi=200);plt.close(fig)
    fig,ax=plt.subplots(figsize=(5,3));left=np.zeros(3)
    for key,label,color in zip(['availability','acquired_feasible_aged_failure','unresolved_failure'],['Available','Acquired-feasible, later failed','Unresolved'],['#a9cfca','#c48532','#596d7c']):
        v=np.array([r[key]*100 for r in hist_summary]);ax.barh(range(3),v,left=left,label=label,color=color);left+=v
    ax.set_yticks(range(3),['Cruise','Turn/block','OOD']);ax.invert_yaxis();ax.set_xlabel('Historical service interval (%)');ax.legend(fontsize=7,loc='lower center',bbox_to_anchor=(.5,1));fig.tight_layout();fig.savefig(out/'figure-6.png',dpi=200);plt.close(fig)
    report=dict(status='PASS',hashed_files_checked=checked,trajectory_rows_per_grid=504,decision_rows_per_grid=12096,independent_seed_clusters=24,historical_rows=18,initializer_rows=72,mat_reference_check=bool(mat_check),mat_reference_max_abs_error=result.get('mat_reference_max_abs_error'),checks=['raw-file checksums','pre-scoring frozen input hashes','training dimensions','table 1 and figures 4-6','paired compute statistics and bootstrap intervals','five learner summaries','full compute accounting','resolution repeat','initializer pilot']+(['432 recorded candidate timers and flags'] if mat_check else []))
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'reproduced');p.add_argument('--mat-check',action='store_true');args=p.parse_args();main(args.out,args.mat_check)
