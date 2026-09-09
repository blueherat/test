"""Recover the complete B/16 member from an interrupted stored ZIP, checking its CRC."""
import hashlib,json,struct,zlib
from pathlib import Path
source=Path('/data/users/zhoushunyu/research_repos/JiT/checkpoints/jit.zip')
target=Path('/home/zhoushunyu/data/eqvae/models/JiT/jit-b-16/checkpoint-last.pth')
header=5510063481;following=7086120956
with source.open('rb') as f:
 f.seek(header);h=struct.unpack('<IHHHHHIIIHH',f.read(30));assert h[0]==0x04034b50 and h[3]==0
 name=f.read(h[-2]).decode();f.read(h[-1]);start=f.tell();assert name=='jit-b-16/checkpoint-last.pth'
 f.seek(following-16);sig,expected_crc,csize,usize=struct.unpack('<IIII',f.read(16))
 assert sig==0x08074b50 and csize==usize and start+csize==following-16
 target.parent.mkdir(parents=True,exist_ok=True);temp=target.with_suffix('.partial')
 crc=0;digest=hashlib.sha256();remaining=csize;f.seek(start)
 with temp.open('wb') as o:
  while remaining:
   b=f.read(min(16*1024*1024,remaining));assert b;o.write(b);crc=zlib.crc32(b,crc);digest.update(b);remaining-=len(b)
 assert crc==expected_crc,(crc,expected_crc)
 if target.exists():
  assert hashlib.sha256(target.read_bytes()).hexdigest()==digest.hexdigest();temp.unlink()
 else:temp.replace(target)
result=dict(source=str(source),member=name,offset=start,size=csize,crc32=crc,crc_verified=True,sha256=digest.hexdigest(),target=str(target),note='Complete member recovered; original truncated archive preserved. CRC is integrity, not external authenticity verification.')
Path('experiments/results/terminal_defect_20260908/jit_transfer/checkpoint_recovery.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
