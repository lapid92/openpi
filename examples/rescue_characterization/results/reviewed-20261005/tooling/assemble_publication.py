"""Explicit small-evidence package; all simulation media remain on Volt."""
import argparse,json,hashlib
from pathlib import Path
R=Path('/volt/artifacts/rescue-characterization');REPO=Path('/volt/code/frozen-flow-study')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);a=p.parse_args()
 reports=[];names=set()
 def add(path,name):
  path=Path(path);assert path.is_file(),path;assert name not in names,name;names.add(name)
  reports.append(dict(path=str(path),name=name,sha256=sha(path)))
 def directory(path,prefix,suffixes={'.json','.jsonl','.py','.md','.log','.csv','.sh','.txt'}):
  for f in sorted(Path(path).iterdir()):
   if f.is_file() and f.suffix in suffixes:add(f,prefix+'/'+f.name)
 add(R/'final-report.md','REPORT.md')
 for source,prefix in [('final-independent-counts','audits/counts'),('final-independent-replay','audits/replay'),('final-reviewer','audits/final-reviewer'),('review-final-v3','review'),('review-control','review-control'),('review-tooling','tooling'),('simulator-diagnostics','simulator-diagnostics')]:
  directory(R/source,prefix)
 for name in ['review-recovery.json','review-postpublication.json','independent-final-audit.json','independent-smoke-audit.json','review-preflight.json','review-pipeline.json']:
  add(R/name,'audits/'+name)
 for name in ['state.json']:
  add(R/'recovery-20261005'/name,'recovery/'+name)
 for f in sorted((R/'coordinator-review').iterdir()):
  if f.suffix in {'.json','.py'}:add(f,'coordinator/'+f.name)
 for cohort in ['new','historical']:
  root=R/f'visual-review-{cohort}'
  for f in sorted((root/'blind').iterdir()):
   if f.suffix in {'.json','.jsonl','.md'}:add(f,'blind/'+cohort+'/'+f.name)
  for f in sorted((root/'private').iterdir()):
   if f.name in {'unblinding.json','selection.json','coverage.json'}:add(f,'mapping/'+cohort+'/'+f.name)
 for name in ['PROTOCOL.md','BLIND_REVIEW.md','build_historical_blind_packets.py']:
  add(REPO/'examples/rescue_characterization'/name,'protocol/'+name)
 # Illustrations intentionally include rescues, a regression, matched control, instability and historical evidence.
 clips=[
 ('new','C090dc827d4cc4a9e19a5','Bottle acquisition failure in F/S/S/F case'),
 ('new','C1bec2653e5f2bd8a1fea','Same bottle case two-step rescue'),
 ('new','C840d5738ae5cf6a31e98','Same bottle case four-step rescue'),
 ('new','C5b01e854d990fdcb590e','Same bottle case ten-step failure; non-monotonic counterexample'),
 ('new','C9fe34110bc86af164805','Butter placement/retention ambiguity at basket rim'),
 ('new','C09a0e9a744cec9b09a6b','Paired butter placement completion'),
 ('new','C6404f5a08cf6bbde17d5','Drawer case one-step failure'),
 ('new','Cb872c7c957a004c61816','Paired drawer two-step rescue'),
 ('new','C4ad67fde7962ff97214a','Matched same-condition control one-step failure'),
 ('new','Cf422e63768b1475f0494','Matched control remains numeric failure despite drawer opening'),
 ('new','Ca38d9e6720868f30da6a','Preserved simulator-affected book case, not policy-only attribution'),
 ('new','C0ace75dd78c305554e51','One-step bowl placement success before higher-count regression'),
 ('new','C954dcd4956421e55f879','Ten-step bowl placement regression with limited external visibility'),
 ('historical','C8c700c642d9c969d06ec','Exactly reconstructed earlier bottle-retention failure'),
 ('historical','C13025af0b16d7eec66b4','Exactly reconstructed earlier bottle/rack rescue')]
 selected=[]
 for cohort,clip,reason in clips:
  f=R/f'visual-review-{cohort}/blind/{clip}.mp4'
  selected.append(dict(path=str(f),clip_id=clip,cohort=cohort,reason=reason,sha256=sha(f)))
 for index in [0,1,3,6,7,9,10,12,13]:
  cohort,clip,reason=clips[index];f=R/f'visual-review-{cohort}/blind/{clip}.png'
  selected.append(dict(path=str(f),clip_id=clip,cohort=cohort,reason=reason+'; twelve-frame overview only',sha256=sha(f)))
 assert len(selected)==24
 summary=R/'review-final-v3/review-summary.json';lock=R/'review-control/baseline-lock.json'
 spec=dict(baseline_locked=True,baseline_lock=str(lock),baseline_lock_sha256=sha(lock),review_summary=str(summary),review_summary_sha256=sha(summary),reports=reports,selected_media=selected,scope='All raw/complete media stay on Volt; selected post-hoc illustrations are not a statistical sample.',numerical_base_commit='f631acc42573a5de5217cd57de4bf7c9b4bf8891',branch='codex/pi05-rescue-characterization')
 with a.output.open('x') as f:json.dump(spec,f,indent=2);f.write('\n')
 print(json.dumps(dict(reports=len(reports),report_bytes=sum(Path(x['path']).stat().st_size for x in reports),selected_media=len(selected),media_bytes=sum(Path(x['path']).stat().st_size for x in selected),manifest=str(a.output)),indent=2))
if __name__=='__main__':main()
