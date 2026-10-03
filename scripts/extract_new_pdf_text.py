from pathlib import Path
import fitz
root=Path('YATA_MPC').resolve()
for p in [root/'references/2024_NOM_E66Me3_SunPower_datasheet.pdf',root/'references/M2096C_SCM_manual.pdf']:
    doc=fitz.open(p)
    text=''.join(f'\n===== PAGE {i+1} =====\n{page.get_text("text")}' for i,page in enumerate(doc))
    p.with_suffix('.txt').write_text(text,encoding='utf-8')
    print(p.name,'pages',len(doc),'chars',len(text))
