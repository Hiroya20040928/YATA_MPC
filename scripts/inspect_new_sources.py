from pathlib import Path
from openpyxl import load_workbook
import json,re
root=Path('YATA_MPC').resolve()
# ASCII-only report avoids console codepage loss; the underlying extracted texts are UTF-8.
out=[]
for fn in ['2024_NOM_E66Me3_SunPower_datasheet.txt','M2096C_SCM_manual.txt']:
    p=root/'references'/fn
    txt=p.read_text(encoding='utf-8')
    out.append(f'===== {fn} =====')
    if fn.startswith('2024_'):
        out.append(txt.encode('unicode_escape').decode('ascii'))
    else:
        for i,line in enumerate(txt.splitlines()):
            if re.search(r'PWM|ECO|efficien|\u6e29\u5ea6|\u30c8\u30eb\u30af|\u96fb\u6d41|\u8eca\u901f|\u30e2\u30fc\u30bf|\u6d88\u8cbb',line,re.I):
                out.append(f'L{i+1}: '+line.encode('unicode_escape').decode('ascii'))
# Workbook key sheets: every populated row for each motor mode and both charge records.
wbpath=root/'data/input/design/M2096_characterization_user_supplied.xlsx'
wb=load_workbook(wbpath,data_only=True,read_only=True)
out.append('===== motor workbook =====')
for ws in wb.worksheets:
    out.append(f'-- {ws.title} {ws.max_row}x{ws.max_column} --')
    for row in ws.iter_rows(values_only=True):
        cells=[f'{i+1}={v!r}' for i,v in enumerate(row) if v is not None]
        if cells: out.append(' ; '.join(cells))
# Charge record values.
for p in (root/'references/archive_review').rglob('*.xlsx'):
    if 'Day5,6' in p.name:
        w=load_workbook(p,data_only=True,read_only=True)
        out.append('===== '+p.name+' =====')
        for ws in w.worksheets:
            out.append(f'-- {ws.title} --')
            for row in ws.iter_rows(values_only=True):
                vals=[str(v) for v in row if v is not None]
                if vals: out.append(' | '.join(vals))
(root/'reports'/'new_source_key_values_ascii.txt').write_text('\n'.join(out),encoding='utf-8')
print('saved lines',len(out),'bytes',(root/'reports'/'new_source_key_values_ascii.txt').stat().st_size)
