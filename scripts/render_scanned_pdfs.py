from pathlib import Path
import fitz
root=Path('YATA_MPC/references/archive_review')
out=Path('YATA_MPC/reports')
targets=[]
for p in root.rglob('*.pdf'):
 if '理由' in p.name or 'ハッチ有無' in p.name:
  targets.append(p)
for idx,p in enumerate(targets,1):
 doc=fitz.open(p); pix=doc[0].get_pixmap(matrix=fitz.Matrix(1.7,1.7),alpha=False)
 dest=out/f'archive_pdf_visual_{idx}.png'; pix.save(dest)
 print(idx,str(p.relative_to(root)).encode('unicode_escape').decode(),len(doc),'pages ->',dest)
