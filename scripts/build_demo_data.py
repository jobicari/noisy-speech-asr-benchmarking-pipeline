#!/usr/bin/env python3
"""Write docs/demo/data.json: transcripts of the demo clips and the WER heatmap.

Run from the repo root after downloading the v2 sweep folders (see analysis/v2/README.md):

    python scripts/build_demo_data.py

The clips are listed in docs/demo/clips.json. They were picked automatically: among
recordings of 7-13 s with 18-35 words, for each corruption the clip whose per-model WER
(capped at 100%) is closest to that model's capped corpus WER on that corruption; for
BL_500_1000, only clips where whisper-tiny loops.
"""
import collections
import json
import pickle
import zlib

P = "results/lambda_bench_runs/"
REPO = ""
r=pickle.load(open(P+'fullrun2000b/SAMPLES_PER_CALL_1_r1/fullRun__hf__986ea79c__2000samples__fullrun2000b_SAMPLES_PER_CALL_1_r1.pkl','rb'))
fb=pickle.load(open(P+'fullrun2000b_loops/WHISPER_FALLBACK_True_r1/fullRun__hf__a9498565__959samples__fullrun2000b_loops.pkl','rb'))
clips=json.load(open(REPO+'docs/demo/clips.json'))
M={'openai_whisper-tiny':'tiny','openai_whisper-base':'base','openai_whisper-small':'small','openai_whisper-medium':'medium','openai_whisper-large-v3-turbo':'turbo','openai_whisper-large-v3':'large-v3','facebook_wav2vec2-large-960h-lv60-self':'wav2vec2'}
ORDER=['tiny','base','small','medium','turbo','large-v3','wav2vec2']
def loops(h): b=h.encode(); return len(b)>0 and len(b)/len(zlib.compress(b))>2.4
agg=collections.defaultdict(lambda:[0,0,0]); rows={}
for z in r:
    m=M[z['model']]; e=z['w_subs']+z['w_dels']+z['w_ins']
    a=agg[(m,z['corr_type'])]; a[0]+=e; a[1]+=min(e,z['num_words']); a[2]+=z['num_words']
    rows[(z['idx'],z['corr_type'],m)]=z
fbr={(z['idx'],z['corr_type'],M[z['model']]):z for z in fb}
corrs=sorted({k[1] for k in agg}, key=lambda c: agg[('large-v3',c)][1]/agg[('large-v3',c)][2])
heat={'models':ORDER,'corruptions':corrs,
      'raw':[[round(100*agg[(m,c)][0]/agg[(m,c)][2],1) for m in ORDER] for c in corrs],
      'capped':[[round(100*agg[(m,c)][1]/agg[(m,c)][2],1) for m in ORDER] for c in corrs]}
out=[]
for c in clips:
    i,k=c['id'],c['corr_type']; ex={'id':i,'corr_type':k,'ref':rows[(i,k,'tiny')]['ref'],
      'duration_s':round(rows[(i,'none','tiny')]['duration_s'],1),'models':[]}
    for m in ORDER:
        z=rows[(i,k,m)]; d={'model':m,'hyp':z['hyp'],'wer':round(100*z['wer'],1),'loop':loops(z['hyp']),
           'corpus_capped':round(100*agg[(m,k)][1]/agg[(m,k)][2],1)}
        if (i,k,m) in fbr and d['loop']:
            f=fbr[(i,k,m)]; d['fallback']={'hyp':f['hyp'],'wer':round(100*f['wer'],1)}
        ex['models'].append(d)
    out.append(ex)
json.dump({'heatmap':heat,'clips':out},open(REPO+'docs/demo/data.json','w'),separators=(',',':'))
print(f"Wrote {len(out)} clips to docs/demo/data.json")
