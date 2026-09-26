"""Sampled macro F0.5 audit of the current exact name/address matching rule."""
import argparse
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parent.parent
CACHE=ROOT/"cache"; GT=ROOT/"datasets"/"train"/"train_ground_truth.tsv"
S1=CACHE/"train_s1_norm.parquet"; TARGETS=[CACHE/"train_s2_norm.parquet",CACHE/"train_s3_norm.parquet"]
CHUNK=50_000
COLS=["entity_id","business_name_normalized","business_address_normalized","country_normalized"]

def norm_frame(df):
    name=df.business_name_normalized.fillna("").astype(str)
    addr=df.business_address_normalized.fillna("").astype(str)
    country=df.country_normalized.fillna("").astype(str)
    return pd.DataFrame({"entity_id":df.entity_id.astype(str),"nk":country+"|"+name,"ak":country+"|"+addr})

def batches(path):
    for b in pq.ParquetFile(path).iter_batches(batch_size=CHUNK,columns=COLS):
        yield norm_frame(b.to_pandas())

def sample_s1(n,seed):
    df=pq.read_table(S1,columns=COLS).to_pandas()
    return norm_frame(df.sample(n=min(n,len(df)),random_state=seed).reset_index(drop=True))

def truth_for(ids):
    truth={x:set() for x in ids}
    for c in pd.read_csv(GT,sep="\t",usecols=["source1_entity_id","matched_entity_ids"],dtype=str,keep_default_na=False,chunksize=CHUNK):
        c=c[c.source1_entity_id.isin(ids)]
        for sid,txt in c.itertuples(index=False,name=None):
            if txt: truth[sid]={x.strip() for x in txt.split(",") if x.strip()}
    return truth

def f05(pred,true,tp):
    if pred==0 and true==0:return 1.0
    if pred==0 or true==0:return 0.0
    p=tp/pred;r=tp/true
    return 1.25*p*r/(.25*p+r) if p+r else 0.0

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--sample-size",type=int,default=2000);ap.add_argument("--seed",type=int,default=20260926);args=ap.parse_args()
    for p in [S1,*TARGETS,GT]:
        if not p.exists():raise FileNotFoundError(p)
    print(f"Sampling {args.sample_size:,} training Source 1 rows...",flush=True)
    sample=sample_s1(args.sample_size,args.seed); ids=set(sample.entity_id); truth=truth_for(ids)
    # Include blank normalized keys to reproduce the current implementation's `country|` join.
    name_keys=set(sample.nk); addr_keys=set(sample.ak)
    pair_keys=set(zip(sample.nk,sample.ak))
    by_name={k:set(g.entity_id) for k,g in sample.groupby("nk") if k in name_keys}
    by_addr={k:set(g.entity_id) for k,g in sample.groupby("ak") if k in addr_keys}
    by_pair={k:set(g.entity_id) for k,g in sample[sample.nk.isin(name_keys)&sample.ak.isin(addr_keys)].groupby(["nk","ak"])}
    true_to_s1={}
    for sid,ts in truth.items():
        for tid in ts:true_to_s1.setdefault(tid,[]).append(sid)
    counts_n={};counts_a={};counts_b={};flags={sid:[] for sid in ids}; lookup=sample.set_index("entity_id")
    print("Scanning target caches in bounded batches...",flush=True)
    for path in TARGETS:
        for f in batches(path):
            n=f[f.nk.isin(name_keys)].nk.value_counts()
            for k,v in n.items():counts_n[k]=counts_n.get(k,0)+int(v)
            a=f[f.ak.isin(addr_keys)].ak.value_counts()
            for k,v in a.items():counts_a[k]=counts_a.get(k,0)+int(v)
            b=f[f.nk.isin(name_keys)&f.ak.isin(addr_keys)].groupby(["nk","ak"]).size()
            for k,v in b.items():
                if k in pair_keys:counts_b[k]=counts_b.get(k,0)+int(v)
            tr=f[f.entity_id.isin(true_to_s1)]
            for tid,nk,ak in tr.itertuples(index=False,name=None):
                for sid in true_to_s1[tid]:
                    s=lookup.loc[sid];flags[sid].append((nk==s.nk and s.nk in name_keys,ak==s.ak and s.ak in addr_keys))
    total_true=sum(map(len,truth.values()));ceiling=sum(sum(n or a for n,a in flags[s]) for s in ids)
    print(f"Exact name/address candidate ceiling: {ceiling/max(1,total_true):.4f} ({ceiling:,}/{total_true:,} true pairs)")
    results=[]
    for nc in (*range(1, 31), 10**12):
        for ac in (*range(0, 16), 20, 50, 10**12):
            vals=[]
            for r in sample.itertuples(index=False):
                nonempty_n=r.nk.split("|",1)[1]!=""; nonempty_a=r.ak.split("|",1)[1]!=""
                n=counts_n.get(r.nk,0); a=counts_a.get(r.ak,0); b=counts_b.get((r.nk,r.ak),0)
                use_n=nonempty_n and n<=nc
                use_a=nonempty_a and a<=ac
                pred=(n if use_n else 0)+(a if use_a else 0)-(b if use_n and use_a else 0)
                tp=sum((x and use_n) or (y and use_a) for x,y in flags[r.entity_id])
                vals.append(f05(pred,len(truth[r.entity_id]),tp))
            results.append((sum(vals)/len(vals),nc,ac))
    current=next(x for x in results if x[1]==10 and x[2]==4)
    address5=next(x for x in results if x[1]==10 and x[2]==5)
    prior=next(x for x in results if x[1]==10 and x[2]==10**12)
    safe_ceiling=sum(sum((n and r.nk.split("|",1)[1]!="") or (a and r.ak.split("|",1)[1]!="") for n,a in flags[r.entity_id]) for r in sample.itertuples(index=False))
    print(f"Nonblank exact-name/address candidate ceiling: {safe_ceiling/max(1,total_true):.4f} ({safe_ceiling:,}/{total_true:,})")
    print(f"Current matcher rule (nonblank name <=10 OR address <=4): macro F0.5={current[0]:.4f}")
    print(f"Address <=5 comparison: macro F0.5={address5[0]:.4f}")
    print(f"Address-unlimited comparison (nonblank name <=10 OR any address): macro F0.5={prior[0]:.4f}")
    print("Best blank-safe frequency cutoffs:")
    for score,nc,ac in sorted(results,reverse=True)[:8]:
        print(f"  {score:.4f}: name <= {nc if nc<10**12 else 'unlimited'}, address <= {ac if ac<10**12 else 'unlimited'}")
    print("Seeded training Source 1 sample; hidden leaderboard score may differ.")
if __name__=="__main__":main()
