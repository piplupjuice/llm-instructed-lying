import csv,json,glob,os,re,statistics as st,collections
import numpy as np
B="C:/Users/nehad/Desktop/llm lies/Model Outputs"
CS=['C1','C2','C3','C4']
def med(v): return round(st.median(v),3) if v else None
def mean(v): return round(sum(v)/len(v),3) if v else None
out={}
for lp in sorted([p for p in glob.glob(f"{B}/**/labels.csv", recursive=True) if "paper" not in p.replace(os.sep, "/").split("/")[-3:-1]]):
    d=os.path.dirname(lp); run=os.path.relpath(d,B).replace(os.sep,'/')
    if 'mist' in run.lower(): continue
    R={}
    L={r['qid']:r for r in csv.DictReader(open(lp,encoding='utf-8-sig'))}
    Q={}
    for line in open(f"{d}/generations.jsonl",encoding='utf-8'):
        g=json.loads(line)
        if g['sample']==-1: Q.setdefault(g['qid'],{})[g['cond']]=g['text']
    beh={r['qid']:r['question'] for r in csv.DictReader(open(f"{d}/behavioural.csv",encoding='utf-8-sig'))}
    for c in CS:
        ids=[q for q,r in L.items() if r['case']==c]
        if not ids: R[c]={'n':0}; continue
        x={'n':len(ids)}
        x['q_len']=med([len(beh.get(q,'')) for q in ids])
        x['truth_len']=med([len(Q[q]['truth']) for q in ids if q in Q])
        x['lie_len']=med([len(Q[q]['lie']) for q in ids if q in Q])
        lies=[Q[q]['lie'] for q in ids if q in Q]
        x['lie_meta%']=round(100*mean([bool(re.search(r'incorrect|wrong|mistake|error',t,re.I)) for t in lies]))
        x['lie_truncwarn']=None
        x['gold_in_lie%']=round(100*mean([L[q]['gold'].replace(',','').strip() in set(re.findall(r'-?\d+(?:\.\d+)?',Q[q]['lie'].replace(',',''))) for q in ids if q in Q]))
        x['luck']=mean([int(L[q]['luck_correct'])/int(L[q]['luck_n']) for q in ids if L[q]['luck_n'] not in ('','0')])
        # how far is lie pred from gold (relative) for wrong lies
        rel=[]
        for q in ids:
            try:
                gld=float(L[q]['gold'].replace(',','')); lp_=float(L[q]['lie_pred'].replace(',',''))
                if gld!=0: rel.append(abs(lp_-gld)/abs(gld))
            except: pass
        x['lie_rel_err']=med(rel)
        R[c]=x
    # softmax
    sm=collections.defaultdict(lambda: collections.defaultdict(list))
    if os.path.exists(f"{d}/softmax.jsonl"):
        for line in open(f"{d}/softmax.jsonl",encoding='utf-8'):
            s=json.loads(line)
            if not s.get('valid'): continue
            k=(s['case'],s['cond'])
            sm[k]['pg'].append(s['p_gold_first']); sm[k]['ent'].append(s['entropy']); sm[k]['top1'].append(s['top10'][0][1]); sm[k]['lp'].append(s['logp_gold_seq'])
    for c in CS:
        if R[c]['n']==0: continue
        for cond in ('truth','lie'):
            v=sm.get((c,cond))
            if v: R[c][f'{cond}_pgold']=med(v['pg']); R[c][f'{cond}_ent']=med(v['ent']); R[c][f'{cond}_top1']=med(v['top1']); R[c][f'{cond}_logp']=med(v['lp'])
    # logit lens: first depth where gold rank==0 ; final rank
    try:
        rank=np.load(f"{d}/lens_gold_rank.npy"); idx=list(csv.DictReader(open(f"{d}/extract_index.csv")))
        nl=rank.shape[2]
        for c in CS:
            rows=[int(r['row']) for r in idx if r['case']==c]
            if not rows: continue
            for ci,cond in enumerate(('truth','lie')):
                fr=[];fin=[]
                for rw in rows:
                    rr=rank[rw,ci]; hit=np.where(rr==0)[0]
                    fr.append(hit[0]/(nl-1) if len(hit) else 1.5); fin.append(float(rr[-1]))
                R[c][f'{cond}_lens_first_top1_depth']=med(fr); R[c][f'{cond}_lens_final_rank']=med(fin)
                R[c][f'{cond}_lens_top1_any%']=round(100*mean([x<=1.0 for x in fr]))
    except Exception as e: R['lens_err']=str(e)
    # geometry stats at P2: mid layer and last layer
    try:
        G=list(csv.DictReader(open(f"{d}/geometry_stats.csv")))
        layers=sorted({int(g['layer']) for g in G}); mid=layers[len(layers)//2]; last=layers[-1]
        for g in G:
            if g['position']=='P2' and int(g['layer']) in (mid,last) and g['case'] in R and R[g['case']].get('n'):
                R[g['case']][f"P2_{g['metric']}_{'mid' if int(g['layer'])==mid else 'last'}"]=round(float(g['mean']),3)
            if g['position']=='P1' and int(g['layer'])==mid and g['case'] in R and R[g['case']].get('n'):
                R[g['case']][f"P1_{g['metric']}_mid"]=round(float(g['mean']),3)
    except Exception as e: R['geo_err']=str(e)
    # attention mass from P2 (lie) to regions
    try:
        acc=collections.defaultdict(lambda: collections.defaultdict(list))
        for jf in glob.glob(f"{d}/attn/*_lie.json"):
            meta=json.load(open(jf)); z=np.load(jf[:-5]+'.npz')['att']  # L,H,2,T
            for pi,pos in enumerate(('P1','P2')):
                a=z[:,:,pi,:].astype(np.float32).mean(axis=(0,1))
                for reg,ix in meta['regions'].items():
                    ix=[i for i in ix if i<len(a)]
                    acc[(meta['case'],pos)][reg].append(float(a[ix].sum()) if ix else 0.0)
        for (c,pos),regs in acc.items():
            for reg,v in regs.items(): R[c][f'attn_{pos}_{reg}']=round(mean(v),4)
            R[c][f'attn_n']=len(next(iter(regs.values())))
    except Exception as e: R['attn_err']=str(e)
    out[run]=R
json.dump(out,open("C:/Users/nehad/AppData/Local/Temp/claude/c--Users-nehad-Desktop-llm-lies/8b51814a-26e4-456b-8e8e-f4cedd1ccc95/scratchpad/percase.json","w"),indent=1)
print("done")
