import argparse,base64,json,io,hashlib,datetime
from pathlib import Path
import imageio.v2 as imageio
from PIL import Image,ImageDraw
p=argparse.ArgumentParser();p.add_argument('cohort');p.add_argument('clip');p.add_argument('frames',nargs='+',type=int);a=p.parse_args()
root=Path('/volt/artifacts/rescue-characterization')
video=root/f'visual-review-{a.cohort}/blind/{a.clip}.mp4'
reader=imageio.get_reader(str(video)); ims=[]
for f in a.frames:
 im=Image.fromarray(reader.get_data(f)); canvas=Image.new('RGB',(im.width,im.height+20),'white');canvas.paste(im,(0,20));ImageDraw.Draw(canvas).text((4,3),f'{a.clip} frame {f}, last action {f-1}, t={f/20:.2f}s',fill='black');ims.append(canvas)
reader.close()
out=Image.new('RGB',(ims[0].width,sum(x.height for x in ims)),'white');y=0
for im in ims:out.paste(im,(0,y));y+=im.height
b=io.BytesIO();out.save(b,format='JPEG',quality=35)
base=root/'coordinator-review'/f'{a.cohort}-{a.clip}-{"-".join(map(str,a.frames))}'
base.with_suffix('.jpg').write_bytes(b.getvalue())
base.with_suffix('.json').write_text(json.dumps(dict(clip_id=a.clip,cohort=a.cohort,frames=a.frames,video=str(video),image_sha256=hashlib.sha256(b.getvalue()).hexdigest(),generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2))
print(base64.b64encode(b.getvalue()).decode())
