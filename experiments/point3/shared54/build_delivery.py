"""Build a verified source bundle, one shared GPU notebook and three report notebooks."""
import ast
import hashlib
import json
from pathlib import Path
import textwrap
import zipfile

from notebook_setup import SETUP
from protocol import HERE, SOURCE, make_plan, save, job_budget


def markdown(value):
    return {'cell_type':'markdown','metadata':{},'source':textwrap.dedent(value).strip()+'\n'}


def code(value):
    value=textwrap.dedent(value).strip()+'\n'
    ast.parse(value)
    return {'cell_type':'code','metadata':{},'source':value,'execution_count':None,'outputs':[]}


def notebook(path,cells,gpu=False):
    for i,cell in enumerate(cells):cell['id']=f'point3-{i:03d}'
    meta={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
          'language_info':{'name':'python'},'colab':{'name':path.name,'provenance':[]}}
    if gpu:meta['accelerator']='GPU'
    save(path,{'nbformat':4,'nbformat_minor':5,'metadata':meta,'cells':cells})


def build():
    dist=HERE/'dist';dist.mkdir(exist_ok=True)
    files=[]
    for folder in ('eval','intervention'):
        files.extend(p for p in (SOURCE/folder).rglob('*') if p.suffix in {'.py','.json'} and '__pycache__' not in p.parts)
    files.extend(SOURCE/name for name in ('requirements.txt','.project-root','LICENSE','AGENTS.md','README.md') if (SOURCE/name).exists())
    files.extend(p for p in (SOURCE/'experiments/point3').glob('*.py'))
    files.extend(p for p in HERE.rglob('*') if p.is_file() and p.suffix in {'.py','.json','.csv','.md'}
                 and not set(p.parts)&{'dist','validation','__pycache__','snapshots'})
    bundle=dist/'source.tmp.zip'
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(set(files)):
            # Deterministic bundle identity, independent of filesystem timestamps.
            info=zipfile.ZipInfo(p.relative_to(SOURCE).as_posix(),date_time=(2026,11,15,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,p.read_bytes())
    sha=hashlib.sha256(bundle.read_bytes()).hexdigest()
    target=dist/f'point3_shared54_source_{sha[:12]}.zip';bundle.replace(target)
    restore=f'''
        import hashlib, json, os, subprocess, sys, time, zipfile
        from pathlib import Path
        _previous_source = globals().get('SOURCE')
        from google.colab import drive
        drive.mount('/content/drive')

        SHARED = Path('/content/drive/MyDrive/Point3_Shared54')
        SOURCE_SHA256 = {sha!r}
        BUNDLE = SHARED / {target.name!r}
        CALIBRATION_SIZE = 18  # 18 calibration + 36 confirmation; 54 sweeps are optional.
        RUN_LABEL = 'v1'       # Keep unchanged to resume; use a new label for a new study.
        SESSION_HOURS = 8      # Session limit, NOT a predicted completion time.
        DEVICES = '0'          # Use '0,1' only when two matching GPUs actually exist.
        OUTPUT = SHARED / 'results' / f'shared54_{{SOURCE_SHA256[:12]}}_cal{{CALIBRATION_SIZE}}_{{RUN_LABEL}}'
        OUTPUT.mkdir(parents=True, exist_ok=True)
        SOURCE = Path('/content') / ('point3_shared54_source_' + SOURCE_SHA256[:12])
        if not BUNDLE.exists():
            raise FileNotFoundError(f'Keep the source ZIP in {{SHARED}} alongside this notebook: {{BUNDLE.name}}')
        if hashlib.sha256(BUNDLE.read_bytes()).hexdigest() != SOURCE_SHA256:
            raise ValueError('Source bundle checksum mismatch')
        with zipfile.ZipFile(BUNDLE) as archive:
            if archive.testzip() is not None: raise ValueError('Damaged source bundle')
            for entry in archive.infolist():
                dest = (SOURCE / entry.filename).resolve()
                if not dest.is_relative_to(SOURCE.resolve()): raise ValueError('Unsafe bundle path')
                payload = archive.read(entry)
                if dest.exists() and dest.read_bytes() != payload:
                    raise ValueError(f'Local source changed: {{dest}}; use a separate version directory')
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists(): dest.write_bytes(payload)
        STUDY = SOURCE / 'experiments/point3/shared54'
        if _previous_source is not None and Path(_previous_source).resolve() != SOURCE.resolve():
            for _name, _module in list(sys.modules.items()):
                _file = getattr(_module, '__file__', None)
                if _file and Path(_file).resolve().is_relative_to(Path(_previous_source).resolve()):
                    sys.modules.pop(_name, None)
        sys.path[:0] = [str(STUDY), str(STUDY.parent), str(SOURCE)]
        WORK = Path('/content/adaptiveattend_point3')
        EVAL_PY = WORK / 'envs/vggt/bin/python'
        SAM_PY = WORK / 'envs/sam3/bin/python'
        SCRATCH = Path('/content/point3_shared54_work')
        DEADLINE = time.monotonic() + SESSION_HOURS*3600
        print('Verified source:', SOURCE_SHA256[:12])
        print('Persistent results:', OUTPUT)
        print('54 landmarks; up to 50 eligible distinct photographs per landmark; actual counts recorded.')
    '''
    cells=[markdown('''
        # Point 3 — shared 54-landmark experiment

        **Run this GPU notebook once for H1 (per layer), H2 (per view), and H3 (per anchor).**
        The three companion notebooks read its outputs without rerunning models.

        The faster design uses **18 calibration + 36 confirmation landmarks**, each with **up to 50 eligible distinct photographs**.
        Seven calibration conditions share their masks, baseline, ground truth and loaded VGGT model.
        Freeze choices on calibration, then run only chosen conditions on confirmation. This is at most
        **306 scene/condition jobs plus reference checks**, compared with 702 independent jobs.
        The 79 final holdout landmarks remain untouched. This is preliminary evidence for point 3,
        not a completed benchmark or a guarantee of finishing within an hour.

        1. Keep this notebook and its source ZIP in `MyDrive/Point3_Shared54`.
        2. Select a GPU runtime. **50-view memory requirements must pass the pilot**; a T4 has not been verified.
        3. In Colab Secrets, create `HF_TOKEN`, enable notebook access, and use a token with approved SAM 3 access.
        4. Run cells in order. The pilot must pass before the main run. Resume with the same settings/output path.

        Forth Bridge uses the user-authorized manual anchor **bridge**. Washington Monument uses the
        authorized **obelisk** correction after its supplied **altar** anchor produced 50 empty masks.
        The original 18/36 split is preserved; obelisk uses H3's global-scalar fallback.
        Every selected-view SAM output is checked;
        saved overlays require visual review before architectural conclusions. Rare anchor types, including
        dome and minaret, use the calibration-selected global scalar fallback.
    '''),code(restore),code('''
        from protocol import make_plan, job_budget
        PLAN = make_plan(CALIBRATION_SIZE)
        print(json.dumps(job_budget(PLAN), indent=2))
        print('Anchor types with sufficient calibration support:', PLAN['supported_anchor_types'])
        print('Global-scalar fallback types:', PLAN['fallback_anchor_types'])
        print('Manual override:', PLAN['anchor_overrides'])
        print('Calibration IDs:', [r['scene'] for r in PLAN['rows'] if r['split']=='calibration'])
    '''),markdown('''
        ## Prepare environments

        Reuses compatible installed environments; otherwise installs isolated, pinned VGGT and SAM environments.
        This can take several minutes. Credentials are read from Secrets only when launching a run.
    '''),code(SETUP),code('''
        def run_phase(phase):
            from google.colab import userdata
            from hypothesis_study import bounded_run
            remaining = (DEADLINE-time.monotonic())/3600
            if remaining <= 0:
                raise RuntimeError('Session budget expired. Rerun the settings cell to start a new session and resume.')
            environment = os.environ.copy()
            try:
                try:
                    environment['HF_TOKEN'] = userdata.get('HF_TOKEN')
                except Exception:
                    raise RuntimeError('Add HF_TOKEN in Colab Secrets and enable notebook access; SAM 3 access must be approved.') from None
                environment['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '0'
                command = [EVAL_PY, '-u', STUDY/'run.py', '--output', OUTPUT, '--work', SCRATCH,
                    '--sam-python', SAM_PY, '--phase', phase, '--hours', str(remaining),
                    '--calibration-size', str(CALIBRATION_SIZE), '--devices', DEVICES]
                bounded_run(command, OUTPUT/'notebook.log', SOURCE, environment, DEADLINE, echo=True)
            finally:
                environment.pop('HF_TOKEN', None)
    '''),markdown('''
        ## Pilot: memory, numerical equivalence and timing

        Runs all seven calibration conditions on one landmark. A fresh reference run is compared against
        the cached run with a reused model. Predictions must match exactly and metrics within the declared
        tolerance. If memory, masks or equivalence fail, the bulk run stays blocked. No automatic reduction
        in image count for memory occurs. Smaller source scenes retain all eligible views up to 50. Inspect logs and change the runtime if needed.
    '''),code('''
        PILOT_PASSED = False
        run_phase('pilot')
        checks = [json.loads(p.read_text()) for p in OUTPUT.glob('scenes/*/reference_check.json')]
        if not any(c.get('status') == 'passed' for c in checks):
            raise RuntimeError('No successful reference check found. Inspect scene logs.')
        PILOT_PASSED = True
        for path in OUTPUT.glob('scenes/*/scene_timing.json'):
            print(path.parent.name, json.loads(path.read_text()))
        print('Pilot passed. These measured scene times are not a dataset-wide runtime guarantee.')
    '''),markdown('''
        ## Complete or resume the shared study

        Runs remaining calibration jobs, freezes scalar/layer/view/anchor choices, and then evaluates
        the 36 confirmation landmarks. Validated checkpoints are reused. An incomplete calibration never
        unlocks confirmation. A deadline or failed scene is recorded as partial, not successful completion.
        Results and SAM masks persist in Drive. Local scene inputs are released after all conditions finish.
    '''),code('''
        if not globals().get('PILOT_PASSED', False):
            raise RuntimeError('Run and pass the pilot cell first.')
        run_phase('all')
    '''),markdown('''
        ## Inspect results and masks

        Open the H1/H2/H3 companion report notebooks after completion, or to inspect partial progress.
        Negative structured-minus-scalar normalized Chamfer distance favors the structured rule.
        Look at failures and no-intervention comparisons as well. Numerical mask checks do not establish
        that SAM segmented the intended architectural object; review overlays, particularly Forth Bridge.
    '''),code('''
        from IPython.display import display, Image, Markdown
        summary = OUTPUT/'summary.json'
        if summary.exists(): print(json.dumps(json.loads(summary.read_text()), indent=2))
        else: print('No report yet; inspect notebook.log and scene failure logs.')
        bridge = hashlib.sha256('Forth_Bridge'.encode()).hexdigest()[:16]
        preview = OUTPUT/'scenes'/bridge/'mask_preview.jpg'
        if preview.exists():
            display(Markdown('### Forth Bridge — manual “bridge” anchor'))
            display(Image(filename=str(preview)))
            print(json.loads((preview.parent/'mask_validation.json').read_text()))
        else: print('Forth Bridge masks have not been generated yet.')
        print('Reports:', OUTPUT/'reports')
    ''')]
    notebook(dist/'Point3_Shared54_Run.ipynb',cells,True)
    descriptions={
        'H1':('Per-layer intervention strength',
              'Test a relative activation/gradient norm rule that varies strength across layers. Calibration selects alpha from 0.01 and 0.02.'),
        'H2':('Per-view intervention strength',
              'Test a fixed mask-visibility rule that varies strength across the selected views. Calibration selects the common eta from 100 and 1000; exponent 0.5 and clipping 0.5–2 stay fixed.'),
        'H3':('Per-anchor intervention strength',
              'Test an anchor-to-eta lookup selected on calibration only. Types with at least two calibration landmarks choose eta 100 or 1000; unsupported types use the global scalar. Dome and minaret are fallback cases, not independently estimated effects.')}
    for h,(title,description) in descriptions.items():
        report_cells=[markdown(f'''
            # {h} — {title}

            {description}

            **CPU report notebook.** Run `Point3_Shared54_Run.ipynb` for the experiments, then read its shared
            results here. This notebook never trains a model, reruns a reconstruction, or selects settings
            from confirmation scores. Default: 18 calibration and 36 confirmation landmarks, up to 50 eligible distinct photographs each.

            Primary comparison: paired landmark difference in normalized Chamfer distance (structured − global
            scalar); lower is better. Report the mean/median, wins/ties/losses and no-intervention comparison.
            These 36 landmarks provide internal confirmation. The final 79-landmark holdout remains unrun.
        '''),code(f'''
            import json
            from pathlib import Path
            import pandas as pd
            from google.colab import drive
            from IPython.display import display, Markdown, Image
            drive.mount('/content/drive')
            CALIBRATION_SIZE = 18
            RUN_LABEL = 'v1'
            OUTPUT = Path('/content/drive/MyDrive/Point3_Shared54/results') / f'shared54_{sha[:12]}_cal{{CALIBRATION_SIZE}}_{{RUN_LABEL}}'
            HYPOTHESIS = {h!r}
            REPORT = OUTPUT/'reports'/HYPOTHESIS
            if not (REPORT/'summary.json').exists():
                raise FileNotFoundError('Run the shared experiment notebook first; use its exact output path.')
            display(Markdown((REPORT/'REPORT.md').read_text()))
            print(json.dumps(json.loads((REPORT/'summary.json').read_text()), indent=2))
        '''),code('''
            paired = pd.read_csv(REPORT/'paired_results.csv')
            display(paired)
            plot = REPORT/'paired_differences.png'
            if plot.exists() and len(paired): display(Image(filename=str(plot)))
            if len(paired):
                print('Mean structured − no-intervention:', paired['structured_minus_baseline'].mean())
                if HYPOTHESIS == 'H3':
                    display(paired.groupby(['anchor','anchor_fallback'], dropna=False)['structured_minus_scalar'].agg(['count','mean','median']))
                    print('Identical H3/scalar predictions:', int(paired['same_prediction_as_scalar'].sum()))
        '''),code('''
            display(pd.read_csv(OUTPUT/'failures.csv'))
            diagnostics = pd.read_csv(OUTPUT/'strength_diagnostics.csv')
            display(diagnostics)
            selection = OUTPUT/'selection.json'
            if selection.exists(): print(json.dumps(json.loads(selection.read_text()), indent=2))
            else: print('Calibration incomplete; no settings frozen yet.')
            print('Raw metrics, prediction fingerprints and traces are in scenes/<scene hash>/<condition>.zip.')
        '''),markdown('''
            ## Interpretation and milestone use

            Claim a benefit only for the evaluated cohort, report all failures, and distinguish a small paired
            difference from robust evidence. A partial cohort is descriptive. A negative result is still useful:
            it tests whether added structure earns its complexity. Review mask overlays and accepted-step traces.
            The two-point candidate grids and one seed are deliberately limited preliminary experiments.
            A larger frozen holdout evaluation and repeated seeds remain follow-up work.
        ''')]
        notebook(dist/f'Point3_{h}_Report.ipynb',report_cells)
    plan=make_plan();save(dist/'planned_study.json',plan)
    save(dist/'delivery.json',{'source_zip':target.name,'source_sha256':sha,'budget':job_budget(plan),
         'notebooks':['Point3_Shared54_Run.ipynb']+[f'Point3_{h}_Report.ipynb' for h in descriptions]})
    print(json.dumps({'source_zip':str(target),'sha256':sha,'notebooks':4},indent=2))


if __name__=='__main__':build()
