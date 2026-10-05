"""Publish only reviewed small reports; never commit/push Git or upload all rollout media."""
import argparse,hashlib,json,shutil
from pathlib import Path
from review_evidence import require,sha,read,write_new
ROOT=Path("/volt/code/frozen-flow-study")
MAX_FILE=40_000_000

def validated_package(spec):
 require(spec.get("baseline_locked") is True,"Publication requires locked baseline")
 summary_path=Path(spec["review_summary"])
 require(sha(summary_path)==spec["review_summary_sha256"],"Review summary changed")
 summary=read(summary_path)
 lock_path=Path(spec["baseline_lock"])
 require(sha(lock_path)==spec["baseline_lock_sha256"]==summary["baseline_lock_sha256"],"Baseline lock provenance differs")
 lock=read(lock_path)
 require(lock["status"]=="baseline_locked" and lock["cohort_counts"]=={"new":272,"historical":116},"Baseline lock coverage differs")
 for item in lock["files"]:require(sha(item["path"])==item["sha256"],"Locked baseline labels changed")
 for item in summary["overlap_source_files"]:require(sha(item["path"])==item["sha256"],"Independent overlap labels changed")
 require(summary["status"]=="complete" and summary["baseline_clips"]==388,"Final publication requires complete baseline/cross-review")
 require(summary["completed_overlap"]==summary["planned_overlap"] and not summary["missing_overlap"],"Independent overlap incomplete")
 require(summary["low_or_unclear_clips"]==summary["low_or_unclear_cross_reviewed"],"Low/unclear review incomplete")
 reports=[]
 names=set()
 for item in spec["reports"]:
  source=Path(item["path"]);name=Path(item["name"])
  require(not name.is_absolute() and ".." not in name.parts and str(name) not in names,"Unsafe/duplicate publication name")
  names.add(str(name))
  require(source.is_file() and source.stat().st_size<MAX_FILE and sha(source)==item["sha256"],"Report size/hash differs")
  require(source.suffix.lower() in (".json",".jsonl",".csv",".tsv",".md",".txt",".py",".log",".sh"),"Reports must be small auditable artifacts")
  require("-worker-" not in source.name,"Never copy unsharded worker raw records")
  reports.append((source,name))
 require(reports,"No reports declared")
 selected=spec.get("selected_media",[])
 require(len(selected)<=24,"Select at most24review examples; full media remain on pod")
 total=0
 for item in selected:
  path=Path(item["path"])
  require(path.suffix.lower() in (".png",".mp4",".npz") and path.is_file(),"Unsupported selected media")
  require(path.stat().st_size<MAX_FILE and sha(path)==item["sha256"],"Selected media differs")
  require(item.get("clip_id") and item.get("reason"),"Selected media needs clip identity and selection rationale")
  total+=path.stat().st_size
 require(total<=256_000_000,"Selected media budget exceeds256MB")
 return reports,selected

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--manifest",type=Path,required=True)
 p.add_argument("--copy-git",type=Path)
 p.add_argument("--execute-wandb",action="store_true")
 p.add_argument("--output",type=Path,required=True)
 args=p.parse_args()
 require(args.manifest.stat().st_size<MAX_FILE,"Publication manifest itself exceeds size limit")
 spec=read(args.manifest);reports,selected=validated_package(spec)
 require(not args.output.exists(),"Refuse publication receipt overwrite")
 receipt={"status":"validated","manifest":str(args.manifest),"manifest_sha256":sha(args.manifest),
          "reports":len(reports),"selected_media":len(selected),"git_commit_or_push_performed":False}
 if args.copy_git:
  dest=args.copy_git.resolve();allowed=(ROOT/"examples/rescue_characterization/results").resolve()
  require(allowed in dest.parents and not dest.exists(),"Git target must be a new results subdirectory")
  dest.mkdir(parents=True,exist_ok=False)
  for source,name in reports:
   target=dest/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
   require(sha(target)==sha(source),"Copy mismatch")
  shutil.copy2(args.manifest,dest/"publication-manifest.json")
  receipt["git_staging_directory"]=str(dest)
 if args.execute_wandb:
  import wandb
  run=wandb.init(project="pi05-rescue-characterization",job_type="final-review-publication",
                 config={"publication_manifest_sha256":sha(args.manifest),"baseline_locked":True,"population_rates_from_visual_sample":False})
  exit_code=1
  try:
   require(bool(run.url),"Online W&B publication unavailable")
   artifact=wandb.Artifact("pi05-reviewed-evidence",type="review-report")
   for source,name in reports:artifact.add_file(str(source),name=str(name))
   artifact.add_file(str(args.manifest),name="publication-manifest.json")
   logged=run.log_artifact(artifact);logged.wait()
   receipt["wandb_run_url"]=run.url;receipt["report_artifact"]=logged.qualified_name
   if selected:
    media=wandb.Artifact("pi05-selected-review-examples",type="review-media")
    for i,item in enumerate(selected):media.add_file(item["path"],name=str(i)+"-"+item["clip_id"]+Path(item["path"]).suffix)
    linked=run.log_artifact(media);linked.wait()
    receipt["selected_media_artifact"]=linked.qualified_name
   receipt["status"]="published_to_wandb"
   run.summary.update(receipt)
   exit_code=0
  finally:run.finish(exit_code=exit_code)
 write_new(args.output,receipt)
 print(json.dumps(receipt,indent=2))
if __name__=="__main__":main()

