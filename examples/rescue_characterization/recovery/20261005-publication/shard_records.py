"""Lossless, line-preserving publication shards; originals remain on the pod."""
import argparse,hashlib,json
from pathlib import Path

LIMIT=39_000_000

def digest(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for block in iter(lambda:f.read(8*1024*1024),b""):h.update(block)
 return h.hexdigest()

def shard(source,folder,limit=LIMIT):
 source=Path(source);folder=Path(folder)
 if limit<=0:raise ValueError("Positive byte limit required")
 folder.mkdir(parents=True,exist_ok=False)
 parts=[];stream=None;size=0;lines=0;first=1;whole=hashlib.sha256();total=0
 def close():
  nonlocal stream,size,lines,first
  if stream:
   name=Path(stream.name).name;stream.flush();stream.close()
   parts.append(dict(name=name,bytes=size,lines=lines,first_line=first,last_line=first+lines-1,sha256=digest(folder/name)))
   first+=lines;stream=None;size=lines=0
 try:
  with source.open("rb") as f:
   for raw in f:
    if len(raw)>=limit:raise ValueError("Single line reaches shard limit")
    if stream and size+len(raw)>=limit:close()
    if stream is None:stream=(folder/f"part-{len(parts)+1:04d}.jsonl").open("wb")
    stream.write(raw);size+=len(raw);lines+=1;whole.update(raw);total+=len(raw)
  close()
 finally:
  if stream and not stream.closed:stream.close()
 record=dict(original_name=source.name,original_bytes=total,original_lines=first-1,
             original_sha256=whole.hexdigest(),parts=parts,limit_exclusive_bytes=limit)
 verify(record,folder)
 if digest(source)!=whole.hexdigest():raise ValueError("Source changed while sharding")
 return record

def verify(record,folder,output=None):
 folder=Path(folder);h=hashlib.sha256();total=lines=0;expected=1
 stream=Path(output).open("xb") if output else None
 try:
  for part in record["parts"]:
   path=folder/part["name"]
   if part["first_line"]!=expected or path.stat().st_size!=part["bytes"] or digest(path)!=part["sha256"]:
    raise ValueError("Shard identity/range mismatch")
   if part["bytes"]>=record["limit_exclusive_bytes"]:raise ValueError("Oversized shard")
   n=0
   with path.open("rb") as f:
    for raw in f:
     h.update(raw);total+=len(raw);n+=1
     if stream:stream.write(raw)
   if n!=part["lines"] or part["last_line"]!=expected+n-1:raise ValueError("Shard line count mismatch")
   expected+=n;lines+=n
  if total!=record["original_bytes"] or lines!=record["original_lines"] or h.hexdigest()!=record["original_sha256"]:
   raise ValueError("Reconstruction differs from original bytes")
 finally:
  if stream:stream.close()
 return True

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--index",type=Path,required=True)
 p.add_argument("--output-dir",type=Path)
 a=p.parse_args();index=json.loads(a.index.read_text())
 if a.output_dir:a.output_dir.mkdir(parents=True,exist_ok=True)
 for record in index["files"]:
  target=a.output_dir/record["original_name"] if a.output_dir else None
  verify(record,a.index.parent/record["parts_directory"],target)
 print(json.dumps({"status":"verified","files":len(index["files"]),"reconstructed":a.output_dir is not None}))
if __name__=="__main__":main()

