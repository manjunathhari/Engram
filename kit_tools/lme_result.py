import json,random,collections,math
import numpy as np
rows=[json.loads(x) for x in open("ATTACHMENTS/5cb997f8-lme_answers.jsonl")]
keys=json.load(open("SCRATCH/lme_key/keys.json"))
lab={}
for i in range(8): lab.update(json.load(open(f"SCRATCH/lme_score/labels{i}.json")))
aud=json.load(open("SCRATCH/lme_score/audit_labels.json"))
print('keys',len(keys),'labels',len(lab),'missing',len(set(keys.values())-set(lab)),'bad',[v for v in lab.values() if v not in (0,1)][:3])
common=set(aud)&set(lab);agree=sum(aud[k]==lab[k] for k in common)
a=np.array([lab[k] for k in common]);b=np.array([aud[k] for k in common]);po=(a==b).mean();pe=a.mean()*b.mean()+(1-a.mean())*(1-b.mean())
print('audit',len(common),'agree',agree,round(po,3),'kappa',round((po-pe)/(1-pe),3),'first=1 second=0',sum(lab[k]==1 and aud[k]==0 for k in common),'first=0 second=1',sum(lab[k]==0 and aud[k]==1 for k in common))
for r in rows: r['m']=lab[keys["%s\t%s"%(r['id'],r['pred'].strip())]]
ids=sorted({r['id'] for r in rows})
get=lambda s:{r['id']:r for r in rows if r['system']==s}
N,C=get('no_facts'),get('current')
def boot(idx,seed=20261106,n=10000):
    d=np.array([N[i]['m']-C[i]['m'] for i in idx],float)
    g=np.random.default_rng(seed);m=d[g.integers(0,len(d),(n,len(d)))].mean(1)
    return round(100*d.mean(),1),round(100*np.percentile(m,2.5),1),round(100*np.percentile(m,97.5),1)
def sign(idx):
    w=sum(N[i]['m']>C[i]['m'] for i in idx);l=sum(N[i]['m']<C[i]['m'] for i in idx);n=w+l
    p=min(1,2*sum(math.comb(n,k) for k in range(min(w,l)+1))/2**n) if n else 1
    return w,l,round(p,5)
out={}
def line(name,idx):
    d=dict(n=len(idx),current=round(100*np.mean([C[i]['m'] for i in idx]),1),networks=round(100*np.mean([N[i]['m'] for i in idx]),1),diff_range=boot(idx),won_lost_p=sign(idx))
    out[name]=d;print(name,d)
line('all',ids)
for t in sorted({r['type'] for r in rows}): line(t,[i for i in ids if N[i]['type']==t and not N[i]['not_answerable']])
line('not answerable',[i for i in ids if N[i]['not_answerable']])
line('answerable',[i for i in ids if not N[i]['not_answerable']])
for name,D in (('current',C),('networks',N)):
    a=[D[i]['answer_sessions'] for i in ids if D[i]['answer_sessions'] is not None and not D[i]['not_answerable']];t=[D[i]['answer_turns'] for i in ids if D[i]['answer_turns'] is not None]
    d=dict(sessions=round(100*np.mean(a),1),turns=round(100*np.mean(t),1),all_sessions=round(100*np.mean([x==1 for x in a]),1),not_mentioned=round(100*np.mean(['not mentioned' in D[i]['pred'].lower() for i in ids]),1),nm_on_answerable=round(100*np.mean(['not mentioned' in D[i]['pred'].lower() for i in ids if not D[i]['not_answerable']]),1),tokens=round(np.mean([D[i]['tokens'] for i in ids])),n_turns=len(t))
    out['ev_'+name]=d;print(name,d)
# reached but wrong
idx=[i for i in ids if not N[i]['not_answerable'] and N[i]['answer_sessions']==1]
print('networks reached all answer sessions',len(idx),'right',sum(N[i]['m'] for i in idx),'wrong',sum(1-N[i]['m'] for i in idx))
idx2=[i for i in ids if not C[i]['not_answerable'] and C[i]['answer_sessions']==1]
print('current reached all',len(idx2),'right',sum(C[i]['m'] for i in idx2))
# marker sensitivity: use second marks where they exist
for r in rows:
    k=keys["%s\t%s"%(r['id'],r['pred'].strip())];r['m2']=aud.get(k,r['m'])
d=np.array([N[i]['m']-C[i]['m'] for i in ids]);
N2={r['id']:r['m2'] for r in rows if r['system']=='no_facts'};C2={r['id']:r['m2'] for r in rows if r['system']=='current'}
print('with second marks: current',round(100*np.mean([C2[i] for i in ids]),1),'networks',round(100*np.mean([N2[i] for i in ids]),1))
# strict recheck on steps: identical answers
print('identical answers',sum(N[i]['pred'].strip()==C[i]['pred'].strip() for i in ids))
json.dump(dict(out=out,labels=lab),open("SCRATCH/lme_score/result.json","w"))
