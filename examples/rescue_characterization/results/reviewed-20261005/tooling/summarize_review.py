"""Transparent coverage and disagreement register for the fixed qualitative sample."""
import argparse,json,csv
from collections import defaultdict,Counter
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--review',type=Path,required=True);a=p.parse_args()
 rows=json.loads((a.review/'label-table.json').read_text());links={r['cohort']+'/'+r['clip_id']:r for r in json.loads((a.review/'evidence-links.json').read_text())}
 groups=defaultdict(list);disagreements=[];agreement={};coverage={}
 for r in rows:groups[(r['cohort'],r['study'],r['benchmark'],r['selection_class'])].append(r)
 distribution=[]
 for keys,rs in sorted(groups.items()):
  distribution.append(dict(zip(['cohort','study','benchmark','selection_class'],keys))|dict(clips=len(rs),cases=len({r['group_id'] for r in rs}),families=len({(r['suite'],r['family']) for r in rs}),conditions=len({r['condition_id'] for r in rs}),distinct_initial_state_hashes=len({r['initial_state_sha256'] for r in rs}),baseline_stages=dict(Counter(r['baseline']['label']['primary_stage'] for r in rs)),baseline_confidence=dict(Counter(r['baseline']['label']['confidence'] for r in rs)),baseline_observed_outcomes=dict(Counter(r['baseline']['label']['observed_outcome'] for r in rs))))
 for r in rows:
  b=r['baseline']['label'];o=(r.get('independent_overlap') or {}).get('label')
  if o:
   diffs={f:[b[f],o[f]] for f in ['observed_outcome','primary_stage','confidence'] if b[f]!=o[f]}
   if diffs:
    substantive=any(f in diffs for f in ['observed_outcome','primary_stage'])
    disagreements.append(dict(cohort=r['cohort'],clip_id=r['clip_id'],differences=diffs,substantive=substantive,resolution='coordinator_unclear_adjudication' if r.get('adjudication') else 'retained_unresolved' if substantive else 'confidence_difference_retained',baseline=r['baseline'],independent_overlap=r['independent_overlap'],adjudication=r.get('adjudication'),raw_link=links[r['cohort']+'/'+r['clip_id']]['immutable_raw_url']))
 for cohort in ['new','historical']:
  rs=[r for r in rows if r['cohort']==cohort];ovs=[r for r in rs if r.get('independent_overlap')]
  agreement[cohort]={f:dict(n=len(ovs),agree=sum(r['baseline']['label'][f]==r['independent_overlap']['label'][f] for r in ovs)) for f in ['observed_outcome','primary_stage','confidence']}
 for kind in ['baseline','independent_overlap']:
  labs=[r[kind]['label'] for r in rows if r.get(kind)]
  coverage[kind]=dict(clips=len(labs),sheet_frame_views=sum(len(l['viewing_coverage'].get('contact_sheet_frames',[])) for l in labs),additional_frame_views=sum(len(l['viewing_coverage'].get('additional_video_frames',[])) for l in labs),numeric_trace_explicit_true=sum(l['viewing_coverage'].get('numeric_trace') is True for l in labs),numeric_trace_narrative_records=sum(isinstance(l['viewing_coverage'].get('numeric_trace'),str) and bool(l['viewing_coverage']['numeric_trace'].strip()) for l in labs),continuous_full_video=sum(l['full_video_viewed'] for l in labs))
 result=dict(distributions=distribution,agreement_by_cohort=agreement,viewing_coverage=coverage,disagreements=len(disagreements),substantive_disagreements=sum(r['substantive'] for r in disagreements),coordinator_unclear_adjudications=sum(bool(r.get('adjudication')) for r in rows),method='Frame views may overlap between sheet/additional views and reviewers. Overlap oversamples ambiguous clips; agreement is descriptive for this selected review sample, not population annotation reliability. All differing labels are retained; unresolved differences cannot support clear class claims.')
 for name,d in [('coverage-details.json',result),('disagreement-register.json',disagreements)]:
  with (a.review/name).open('x') as f:json.dump(d,f,indent=2);f.write('\n')
 fields=['cohort','study','benchmark','suite','family','condition_id','category','severity','seed','init_index','initial_state_sha256','group_id','clip_id','arm','raw_success','selection_class','original_selection_class','paired_claim_eligible']
 extra=['baseline_outcome','baseline_stage','baseline_confidence','overlap_outcome','overlap_stage','overlap_confidence','adjudicated_outcome','immutable_raw_url','immutable_replay_url']
 with (a.review/'labels.csv').open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields+extra);w.writeheader()
  for r in rows:
   d={k:r[k] for k in fields};b=r['baseline']['label'];o=(r.get('independent_overlap') or {}).get('label',{});ad=(r.get('adjudication') or {}).get('label',{})
   d.update(baseline_outcome=b['observed_outcome'],baseline_stage=b['primary_stage'],baseline_confidence=b['confidence'],overlap_outcome=o.get('observed_outcome'),overlap_stage=o.get('primary_stage'),overlap_confidence=o.get('confidence'),adjudicated_outcome=ad.get('observed_outcome'))
   ref=links[r['cohort']+'/'+r['clip_id']];d.update({k:ref[k] for k in ['immutable_raw_url','immutable_replay_url']});w.writerow(d)
 print(json.dumps({k:result[k] for k in ['agreement_by_cohort','viewing_coverage','substantive_disagreements','coordinator_unclear_adjudications']},indent=2))
if __name__=='__main__':main()
