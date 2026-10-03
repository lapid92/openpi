"""Build deterministic blinded review packets from complete audited four-arm records.
No simulator/model execution. Keep private/ inaccessible to the blind reviewer.
"""
import argparse
from collections import defaultdict, Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw

ARMS = (1, 2, 4, 10)
def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
def filehash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""): h.update(b)
    return h.hexdigest()
def rank(seed, value): return digest([seed, value])
def select_balanced(cases, limit, seed):
    """Round robin across families and conditions, independent of within-class magnitudes."""
    families = defaultdict(lambda: defaultdict(list))
    for case in cases:
        r = case["rows"][1]
        families[(r["suite"], r["family"])][r["condition_id"]].append(case)
    queues = []
    for family in sorted(families, key=lambda x: rank(seed, x)):
        conditions = families[family]
        slots = [sorted(v, key=lambda c: rank(seed, c["key"])) for _, v in
                 sorted(conditions.items(), key=lambda kv: rank(seed, kv[0]))]
        queue = []
        while any(slots):
            for slot in slots:
                if slot: queue.append(slot.pop(0))
        queues.append(queue)
    selected = []
    while any(queues) and len(selected) < limit:
        for queue in queues:
            if queue and len(selected) < limit: selected.append(queue.pop(0))
    return selected
def sheet(video, trace, clip_id, output, frame_count, samples=12):
    with np.load(trace, allow_pickle=False) as a:
        actions = a["actions"].copy()
        states = a["eef_gripper_states"].copy()
    require(len(actions) == frame_count - 1, "Action/frame count mismatch")
    indices = sorted(set(np.linspace(0, frame_count - 1, min(samples, frame_count)).round().astype(int).tolist()))
    select = "+".join("eq(n\\," + str(i) + ")" for i in indices)
    proc = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-vf",
                           "select=" + select, "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                          check=True, stdout=subprocess.PIPE)
    # Input dimensions are independently probed, not guessed from raw bytes.
    meta = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "json", str(video)], text=True))["streams"][0]
    w,h = int(meta["width"]),int(meta["height"])
    frames = np.frombuffer(proc.stdout, dtype=np.uint8).reshape(-1,h,w,3)
    require(len(frames) == len(indices), "Selected-frame mismatch")
    cell_w,cell_h,cols = 512,290,3
    canvas = Image.new("RGB",(cols*cell_w,44+cell_h*((len(indices)+cols-1)//cols)), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((10,10), clip_id + " | two camera views | frame 0 = stabilized start", fill="black")
    for k,(index,frame) in enumerate(zip(indices,frames)):
        x,y=(k%cols)*cell_w,44+(k//cols)*cell_h
        im=Image.fromarray(frame);im.thumbnail((512,256));canvas.paste(im,(x,y))
        draw.text((x+4,y+258), f"frame/action {index} | t={index/20:.2f}s", fill="black")
    canvas.save(output)
    # Indices and action/pose measurements only: never labels, outcomes, or step arm.
    return {"sampled_frame_indices":indices,"simulator_actions":len(actions),
            "action_mean_abs":np.abs(actions).mean(axis=0).tolist(),
            "action_max_abs":np.abs(actions).max(axis=0).tolist(),
            "gripper_action_min_max":[float(actions[:,-1].min()),float(actions[:,-1].max())],
            "eef_start_xyz":states[0,:3].tolist(),"eef_last_pre_action_xyz":states[-1,:3].tolist(),
            "eef_path_length_pre_action":float(np.linalg.norm(np.diff(states[:,:3],axis=0),axis=1).sum()),
            "trace_note":"EEF/gripper states precede each action; final post-action pose is absent."}
def require(test,message):
    if not test: raise ValueError(message)
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--records",nargs="+",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--per-class",type=int,default=12)
    p.add_argument("--selection-seed",type=int,default=20261003)
    p.add_argument("--blind-key",required=True,help="Private opaque-ID key; never share with reviewer")
    p.add_argument("--manifest",type=Path,required=True)
    a=p.parse_args()
    require(a.per_class>0,"Positive per-class count required")
    require(not a.output.exists(),"Refuse overwrite of review packet")
    manifest=json.loads(a.manifest.read_text())
    mh=filehash(a.manifest)
    groups=defaultdict(dict); error_refs=[]; sources=[]
    for path in a.records:
        fh=filehash(path);sources.append({"path":str(path),"sha256":fh})
        for lineno,line in enumerate(path.read_text().splitlines(),1):
            if not line.strip():continue
            row=json.loads(line)
            require(row["manifest_sha256"]==mh,"Mixed manifest records")
            require(row["phase"]=="main","Only scored main cases eligible; smoke must stay separate")
            ref={"path":str(path),"line":lineno,"file_sha256":fh,"record_sha256":hashlib.sha256(line.encode()).hexdigest()}
            if row["status"]=="error":
                error_refs.append(ref);continue
            require(row["status"]=="ok" and type(row["success"]) is bool,"Invalid outcome")
            key=(row["benchmark"],row["condition_id"],row["seed"],row["init_index"])
            require(row["flow_steps"] not in groups[key],"Duplicate arm")
            groups[key][row["flow_steps"]]={"row":row,"source":ref}
    expected={(bench,c["condition_id"],x["seed"],x["init_index"]) for bench,spec in manifest["benchmarks"].items()
              for c in spec["conditions"] for x in c["cases"]}
    require(set(groups)==expected,"Require complete declared population before outcome-stratified sampling")
    strata=defaultdict(list)
    for key, entries in groups.items():
        require(set(entries)==set(ARMS),"Incomplete matched four-arm case")
        rows={arm:entries[arm]["row"] for arm in ARMS}
        vector=[rows[n]["success"] for n in ARMS]
        kind="rescue" if not vector[0] and any(vector[1:]) else "regression" if vector[0] and not all(vector[1:]) else "never_rescued" if not any(vector) else None
        if kind:strata[(key[0],kind)].append({"key":key,"rows":rows,"entries":entries,"kind":kind})
    selected=[]
    coverage={}
    for key,cases in sorted(strata.items(), key=lambda kv:(kv[0][1]=="never_rescued",kv[0])):
        chosen=select_balanced(cases,a.per_class,a.selection_seed)
        if key[1]=="never_rescued":
            targets={(c["rows"][1]["suite"],c["rows"][1]["family"],c["rows"][1]["category"],c["rows"][1]["severity"])
                     for c in selected if c["key"][0]==key[0]}
            matched=[c for c in cases if (c["rows"][1]["suite"],c["rows"][1]["family"],c["rows"][1]["category"],c["rows"][1]["severity"]) in targets]
            chosen=select_balanced(matched,a.per_class,a.selection_seed)
            remaining=[c for c in cases if c not in chosen]
            chosen+=select_balanced(remaining,a.per_class-len(chosen),a.selection_seed)

        selected.extend(chosen)
        coverage["/".join(key)]={"available":len(cases),"selected":len(chosen),
            "available_families":len({(c["rows"][1]["suite"],c["rows"][1]["family"]) for c in cases}),
            "selected_families":len({(c["rows"][1]["suite"],c["rows"][1]["family"]) for c in chosen}),
            "selected_conditions":len({c["rows"][1]["condition_id"] for c in chosen}),
            "matched_never_available":len(matched) if key[1]=="never_rescued" else None,
            "matched_never_selected":sum(c in matched for c in chosen) if key[1]=="never_rescued" else None}
    blind=a.output/"blind";private=a.output/"private";blind.mkdir(parents=True);private.mkdir(mode=0o700)
    packets=[];unblind=[]
    for case in sorted(selected,key=lambda c:rank(a.blind_key,c["key"])):
        group_id="G"+rank(a.blind_key,case["key"])[:16]
        packet={"group_id":group_id,"task_instruction":case["rows"][1]["prompt"],"clips":[]}
        for arm in sorted(ARMS,key=lambda n:rank(a.blind_key,[case["key"],n])):
            row=case["rows"][arm];capture=row["recording"]
            clip_id="C"+rank(a.blind_key,[case["key"],arm])[:20]
            for kind in ("video","trace"):
                require(filehash(capture[kind+"_path"])==capture[kind+"_sha256"],"Artifact changed")
            # Hard link gives an opaque video name without copying or revealing source filename.
            os.link(capture["video_path"],blind/(clip_id+".mp4"))
            image_path=blind/(clip_id+".png")
            metrics=sheet(capture["video_path"],capture["trace_path"],clip_id,image_path,capture["video_frames"])
            # Copy numeric trace minus terminal success to an opaque artifact for detailed time-series review.
            with np.load(capture["trace_path"],allow_pickle=False) as z:
                np.savez_compressed(blind/(clip_id+".npz"),actions=z["actions"],eef_gripper_states=z["eef_gripper_states"])
            packet["clips"].append({"clip_id":clip_id,"contact_sheet":image_path.name,
                                   "video":clip_id+".mp4","numeric_trace":clip_id+".npz","metrics":metrics})
            unblind.append({"group_id":group_id,"clip_id":clip_id,"arm":arm,"case_key":case["key"],
                            "selection_class":case["kind"],"success":row["success"],"source":case["entries"][arm]["source"],
                            "recording":capture,"condition_id":row["condition_id"],"family":row["family"],
                            "category":row["category"],"severity":row["severity"]})
        packets.append(packet)
    (blind/"packets.json").write_text(json.dumps(packets,indent=2)+"\n")
    (private/"unblinding.json").write_text(json.dumps(unblind,indent=2)+"\n")
    (private/"selection.json").write_text(json.dumps({"seed":a.selection_seed,"per_class":a.per_class,
        "coverage":coverage,"source_files":sources,"manifest_sha256":mh,"error_attempts":error_refs,
        "selection":"Equal target per benchmark/class; round robin families then conditions; hash order within each level.",
        "scope":"Outcome-stratified qualitative sample; never estimate population rates from selected clips."},indent=2)+"\n")
    print(json.dumps({"output":str(a.output),"groups":len(packets),"clips":len(unblind),"coverage":coverage},indent=2))
if __name__=="__main__":main()

