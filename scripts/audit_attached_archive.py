from __future__ import annotations
import argparse, csv, hashlib, json, zipfile
from pathlib import Path, PurePosixPath
parser = argparse.ArgumentParser(description='Inventory a user-supplied archive without extracting outside the project folder.')
parser.add_argument('archive', type=Path, help='path to the archive to inspect')
parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[1])
args = parser.parse_args()
src = args.archive.expanduser().resolve()
root = args.project_root.expanduser().resolve()
if not src.is_file():
    parser.error(f'archive not found: {src}')
dst = (root / 'references' / 'archive_review').resolve()
dst.mkdir(parents=True, exist_ok=True)
def decode_name(info):
    n = info.filename
    if info.flag_bits & 0x800:
        return n
    try:
        return n.encode('cp437').decode('cp932')
    except Exception:
        return n
rows=[]
with zipfile.ZipFile(src) as z:
    for i, info in enumerate(z.infolist()):
        name=decode_name(info)
        # Normalize both slash styles before containment check.
        rel=PurePosixPath(name.replace('\\','/'))
        target=(dst.joinpath(*rel.parts)).resolve()
        if not target.is_relative_to(dst):
            rows.append({'index':i,'name':name,'size':info.file_size,'sha256':'','extracted':'BLOCKED_PATH'})
            continue
        if info.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            rows.append({'index':i,'name':name,'size':0,'sha256':'','extracted':'directory'})
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        payload=z.read(info)
        digest=hashlib.sha256(payload).hexdigest()
        target.write_bytes(payload)
        rows.append({'index':i,'name':name,'size':len(payload),'sha256':digest,'extracted':str(target.relative_to(root))})
with (root/'reports'/'attached_zip_inventory.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=['index','name','size','sha256','extracted'])
    w.writeheader(); w.writerows(rows)
summary={}
for row in rows:
    ext=Path(row['name']).suffix.lower() or '[no extension]'
    summary[ext]=summary.get(ext,0)+1
(root/'reports'/'attached_zip_inventory_summary.json').write_text(json.dumps({'archive':str(src),'entries':len(rows),'extensions':summary},ensure_ascii=False,indent=2),encoding='utf-8')
print('entries',len(rows),'extensions',json.dumps(summary,ensure_ascii=False))
for row in rows:
    ext=Path(row['name']).suffix.lower()
    if ext in {'.csv','.xls','.xlsx','.pdf','.docx','.txt'}:
        print(f"{row['size']:>10} {row['name']}")
