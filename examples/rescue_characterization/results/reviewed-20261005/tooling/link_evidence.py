"""Build immutable raw links without changing any source or label."""
import argparse,json,re,hashlib
from pathlib import Path
ROOT=Path('/volt/code/frozen-flow-study')
PREFIX='https://github.com/lapid92/openpi/blob/f631acc42573a5de5217cd57de4bf7c9b4bf8891/'
def main():
 p=argparse.ArgumentParser();p.add_argument('--joined',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
 rows=json.loads(a.joined.read_text())
 index=json.loads((ROOT/'examples/rescue_characterization/results/e7537584d348/raw-shards/index.json').read_text())
 files={x['source_path']:x for x in index['files']}
 old={}
 text=(ROOT/'examples/rescue_characterization/results/existing/CASES.md').read_text()
 for url,study,name,line in re.findall(r'(https://github.com/lapid92/openpi/blob/[a-f0-9]{40}/examples/(frozen_flow|selective_flow)/results/[a-f0-9]+/([^/#]+)#L([0-9]+))',text):
  old[(study,name,int(line))]=url
 result=[]
 for r in rows:
  ref=r['source'];line=ref['line']
  if r['cohort']=='new':
   f=files[ref['path']];assert f['original_sha256']==ref['file_sha256']
   part=next(x for x in f['parts'] if x['first_line']<=line<=x['last_line'])
   path='examples/rescue_characterization/results/e7537584d348/raw-shards/'+f['parts_directory']+'/'+part['name']
   link=PREFIX+path+'#L'+str(line-part['first_line']+1)
  else:
   study='selective_flow' if '/selective-flow-study/' in ref['path'] else 'frozen_flow'
   link=old[(study,Path(ref['path']).name,line)]
  rr=r.get('replay_source')
  replay_link=PREFIX+'examples/rescue_characterization/results/historical-replay/'+Path(rr['path']).name+'#L'+str(rr['line']) if rr else None
  result.append(dict(cohort=r['cohort'],clip_id=r['clip_id'],group_id=r['group_id'],arm=r['arm'],source=ref,immutable_raw_url=link,replay_source=rr,immutable_replay_url=replay_link,label_source=r['baseline']['label_source'],independent_label_source=(r.get('independent_overlap') or {}).get('label_source')))
 with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print('linked',len(result),'records')
if __name__=='__main__':main()
