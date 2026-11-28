"""Verify saved Point 3 evidence; optionally reproduce analyses in a temporary copy."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verify_snapshot():
    manifest = load(ROOT/'provenance/immutable_sha256.json')
    for name, expected in manifest.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected, name
    evidence = ROOT/'report/evidence'
    for name, expected in load(evidence/'evidence_sha256.json').items():
        assert hashlib.sha256((evidence/name).read_bytes()).hexdigest() == expected, name
    seed = ROOT/'selection/tooling/seeds/original_40_landmarks.json'
    assert hashlib.sha256(seed.read_bytes()).hexdigest() == '4f8e8885f1cf3bc0d48e92f662555c76293cf4ebf683941f27c8e232b6c36d2c'
    original = load(seed)
    assert isinstance(original, list) and len(original) == len(set(original)) == 40
    selection = load(ROOT/'selection/evidence/point3_selection_summary.json')
    chosen = set(selection['selected_landmarks'])
    additions = {r['scene'] for r in selection['additions']}
    assert len(chosen) == 54 and len(additions) == 14
    assert chosen == set(original) | additions and not set(original) & additions
    notebook = load(ROOT/'selection/original40_run.ipynb')
    outputs = '\n'.join(''.join(o.get('text', [])) for c in notebook['cells'] for o in c.get('outputs', []))
    assert 'run_3181fc0dfdf53b02' in outputs and 'heuristic_coverage_elbow' in outputs
    assert '"selected_landmarks": 40' in outputs
    plans = [load(evidence/name)['plan'] for name in ['h1_run_study.json','h2_run_study.json','h3_study.json']]
    identities = [{r['scene']: r['split'] for r in p['rows']} for p in plans]
    assert identities[0] == identities[1] == identities[2]
    assert set(identities[0]) == chosen
    cases = load(ROOT/'qualitative/cases.json')
    summary = load(ROOT/'qualitative/verified_figure_summary.json')
    with (evidence/'paired_hypotheses.csv').open(encoding='utf-8', newline='') as handle:
        paired = {r['scene']: r for r in csv.DictReader(handle)}
    for h, case in cases.items():
        geometry = ROOT/'qualitative/geometry'/h
        results = load(geometry/'results.json')
        assert results['identity']['scene'] == case['scene']
        for condition, column in [('no_intervention','baseline'),(case['scalar'],'scalar'),(case['condition'],h)]:
            check = load(geometry/(condition+'_verification.json'))
            assert check['status'] == 'passed' and check['prediction_hashes_exact'] and check['metrics_match']
            assert math.isclose(results['conditions'][condition]['CD_norm'],float(paired[case['scene']][column]),rel_tol=0,abs_tol=1e-10)
        figure = ROOT/'qualitative/figures'/h/'before_after.png'
        assert hashlib.sha256(figure.read_bytes()).hexdigest() == summary[h]['figure_sha256']
    from pypdf import PdfReader
    reader = PdfReader(ROOT/'report/main.pdf')
    assert len(reader.pages) == 19
    assert all(':' not in p.extract_text() and ';' not in p.extract_text() for p in reader.pages)
    for name in ['04_baseline_evaluation','05_pass1_signals','06_adaptive_selection','08_discussion','09_conclusion']:
        lines = (ROOT/'report/sections'/(name+'.tex')).read_text(encoding='utf-8').splitlines()
        assert not any(line.strip() and not line.lstrip().startswith('%') for line in lines), name
    print(f'Passed immutable hashes for {len(manifest)} files, selection lineage, figure records, and 19-page PDF.', flush=True)


def same_csv(expected, actual):
    with expected.open(encoding='utf-8-sig',newline='') as f:
        first = list(csv.reader(f))
    with actual.open(encoding='utf-8-sig',newline='') as f:
        second = list(csv.reader(f))
    assert len(first) == len(second), expected
    for i,(left,right) in enumerate(zip(first,second)):
        assert len(left) == len(right), (expected,i)
        for a,b in zip(left,right):
            if a == b:
                continue
            try:
                x,y = float(a),float(b)
            except ValueError:
                raise AssertionError((expected,i,a,b)) from None
            assert (math.isnan(x) and math.isnan(y)) or math.isclose(x,y,rel_tol=1e-10,abs_tol=1e-10), (expected,i,a,b)


def reproduce():
    with tempfile.TemporaryDirectory(prefix='point3-verify-') as temp:
        work = Path(temp)/'package'
        shutil.copytree(ROOT,work,ignore=shutil.ignore_patterns('__pycache__','*.log','*.aux','*.bbl','*.out','*.blg'))
        def run(script,*args):
            print('Reproducing',script,flush=True)
            process = subprocess.run([sys.executable,'-X','utf8',str(work/script),*args],cwd=work,
                                     stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8')
            if process.returncode:
                raise RuntimeError(f'{script} failed\n{process.stdout[-10000:]}')
        run('report/reproduce_results.py')
        run('report/anchor_analysis/analyze_by_anchor.py')
        run('analysis/run_all.py')
        run('findings/analyze_saved_findings.py')
        run('findings/outlier_sensitivity.py')
        run('selection_audit/analyze_selection.py')
        run('selection_audit/coverage_and_similarity.py')
        checked = 0
        folders = ['report/anchor_analysis','analysis/audit','analysis/effects','analysis/groups',
                   'analysis/correlations','findings','selection_audit']
        for folder in folders:
            for original in (ROOT/folder).glob('*.csv'):
                same_csv(original,work/original.relative_to(ROOT))
                checked += 1
        same_csv(ROOT/'analysis/all_exploratory_primary_tests.csv',work/'analysis/all_exploratory_primary_tests.csv')
        assert load(work/'analysis/global_multiplicity.json')['global_bh_discoveries_at_0_05'] == 0
        assert load(work/'report/evidence/verified_statistics.json') == load(ROOT/'report/evidence/verified_statistics.json')
        assert load(work/'findings/outlier_sensitivity.json') == load(ROOT/'findings/outlier_sensitivity.json')
        print(f'Passed regeneration and numeric comparisons for {checked+1} CSV tables plus core JSON summaries.',flush=True)
    verify_snapshot()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full',action='store_true',help='Reproduce all CPU analyses in a temporary copy.')
    args = parser.parse_args()
    verify_snapshot()
    if args.full:
        reproduce()
