"""Reproduce all statistical tables from the packaged frozen evidence.

Requires NumPy and pandas; Matplotlib for figures. No GPU or network access.
"""
import argparse
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=root/'sources');p.add_argument('--skip-plots',action='store_true')
a=p.parse_args()
(root/'logs').mkdir(exist_ok=True)
def run(command,name):
    log=root/'logs'/(name+'.log')
    with log.open('w',encoding='utf-8') as handle:
        result=subprocess.run(command,stdout=handle,stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'{name} failed; inspect {log}')
for script,output in [('audit_and_overlap.py','audit'),('effects/analyze_effects.py','effects'),('groups/analyze_groups.py','groups'),('correlations/analyze_correlations.py','correlations')]:
    print('Running',script,flush=True)
    run([sys.executable,str(root/script),'--source',str(a.source.resolve()),'--output',str(root/output)],output)
if not a.skip_plots:
    run([sys.executable,str(root/'build_synthesis.py'),'--source',str(a.source.resolve())],'figures')
print('All requested CPU analyses completed.')
