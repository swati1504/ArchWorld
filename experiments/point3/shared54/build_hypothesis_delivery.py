"""Package three independent GPU notebooks: one executable experiment per hypothesis."""
import copy
import json

from build_delivery import build,markdown,code,notebook
from build_mask_delivery import build_mask_notebook
from gpu_setup import setup_for
from protocol import HERE,load,save,candidate_names,job_budget


METHODS={
    'H1':('Layer-dependent intervention',
        'Does adapting the coefficient to each layer improve reconstruction over one global scalar?',
        'Use eta_l = min(100000, alpha * norm(T_l[M]) / max(norm(g_l[M]), 1e-8)). '
        'Tune alpha in {0.01, 0.02}; scalar controls tune eta in {100, 1000}. '
        'Keep the reference objective, all-token update and accept/revert rule.',198),
    'H2':('View-dependent intervention',
        'Does varying the coefficient with anchor visibility in each view improve reconstruction?',
        'Use eta_v = eta * q_v / RMS(q), with q_v = clip(((n_v+1)/(mean(n)+1))^0.5, 0.5, 2). '
        'Tune eta in {100, 1000} for the view rule and for the scalar control separately. '
        'Exponent 0.5, clipping 0.5–2 and the first-hook visibility counts remain fixed across layers.',198),
    'H3':('Anchor-dependent intervention',
        'Does choosing a scalar by architectural anchor type improve over one scalar shared by all anchors?',
        'Tune eta in {100, 1000} independently for each type with at least two calibration landmarks. '
        'Other types use the calibration-selected global scalar, including the authorized Washington Monument obelisk correction. Supported types are arch, column, facade, '
        'statue, tower and window. Dome/minaret have insufficient support and use the fallback; no spire is present.',162)
}


def replace_once(text,old,new):
    if text.count(old)!=1:raise ValueError('Notebook template changed; expected exactly one replacement: '+old)
    return text.replace(old,new)


def build_three():
    build()
    dist=HERE/'dist';delivery=load(dist/'delivery.json')
    source=load(dist/'Point3_Shared54_Run.ipynb');base_plan=load(dist/'planned_study.json')
    books=[build_mask_notebook(dist,source)];budgets={}
    for h,(title,question,method,budget) in METHODS.items():
        settings=source['cells'][1]['source'].replace('Point3_Shared54','Point3_Hypothesis54')
        settings=replace_once(settings,
            "OUTPUT = SHARED / 'results' / f'shared54_{SOURCE_SHA256[:12]}_cal{CALIBRATION_SIZE}_{RUN_LABEL}'",
            f"HYPOTHESIS = {h!r}\nSTUDY_ROOT = SHARED / 'results' / f'hypotheses54_{{SOURCE_SHA256[:12]}}_cal{{CALIBRATION_SIZE}}_{{RUN_LABEL}}'\nOUTPUT = STUDY_ROOT / HYPOTHESIS")
        settings=replace_once(settings,"SCRATCH = Path('/content/point3_shared54_work')",
                              "SCRATCH = Path('/content/point3_hypothesis54_work') / HYPOTHESIS")
        settings=replace_once(settings,"print('54 landmarks; up to 50 eligible distinct photographs per landmark; actual counts recorded.')",
                              "print(HYPOTHESIS, ': 54 landmarks; up to 50 eligible distinct photographs per landmark; actual counts recorded.')")
        settings+='''
COMMON_MASKS = SHARED / 'common_masks'
MASK_ROLE = 'read'
print('Common masks:', COMMON_MASKS)
print('Mask role:', MASK_ROLE, '(all hypotheses reuse the preparation notebook outputs)')
'''
        plan_cell=replace_once(source['cells'][2]['source'],'make_plan(CALIBRATION_SIZE)','make_plan(CALIBRATION_SIZE, HYPOTHESIS)')
        plan_cell+='''
from mask_cache import require_ready
READY_MASKS = require_ready(COMMON_MASKS, PLAN)
print('Shared masks ready:', READY_MASKS['landmarks'], 'landmarks /', READY_MASKS['images'], 'masks')
'''
        setup=setup_for(source['cells'][4]['source'],'evaluation')
        launcher='''
def run_phase(phase):
    from google.colab import userdata
    from hypothesis_study import bounded_run
    remaining = (DEADLINE-time.monotonic())/3600
    if remaining <= 0:
        raise RuntimeError('Session budget expired. Rerun settings to start a new session and resume.')
    environment = os.environ.copy()
    try:
        try:
            environment['HF_TOKEN'] = userdata.get('HF_TOKEN')
        except Exception:
            environment.pop('HF_TOKEN', None)
            print('HF_TOKEN unavailable; using public VGGT downloads and shared masks.')
        environment['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '0'
        command = [EVAL_PY, '-u', STUDY/'run.py', '--output', OUTPUT, '--work', SCRATCH,
            '--sam-python', SAM_PY, '--phase', phase, '--hours', str(remaining),
            '--calibration-size', str(CALIBRATION_SIZE), '--devices', DEVICES,
            '--hypothesis', HYPOTHESIS, '--mask-cache', COMMON_MASKS, '--mask-role', MASK_ROLE]
        bounded_run(command, OUTPUT/'notebook.log', SOURCE, environment, DEADLINE, echo=True)
    finally:
        environment.pop('HF_TOKEN', None)
'''
        pilot=replace_once(source['cells'][7]['source'],"print('Pilot passed. These measured scene times are not a dataset-wide runtime guarantee.')",
                           "print(HYPOTHESIS, 'pilot passed. Scene timing is not a dataset-wide runtime guarantee.')")
        cells=[markdown(f'''
            # {h} — {title}

            **This notebook runs the complete {h} experiment, including its own controls, tuning and report.**
            It can run concurrently with the other two notebooks when each has a separate GPU runtime.
            All three use the same **54 landmarks**, with **18 calibration + 36 confirmation** and **up to 50
            eligible distinct photographs per landmark**. Smaller landmarks retain all eligible views; none are
            replaced or padded. All methods use identical images for each landmark, and tables report actual counts.
            Each has a separate output folder, checkpoints and logs. The 79 final holdout landmarks are untouched.

            **Question:** {question}

            **Method and decision variables:** {method}

            Pick settings using mean calibration normalized Chamfer distance only; freeze before confirmation.
            Primary outcome: the paired confirmation-landmark difference, structured − global scalar (negative is better).
            Also report the no-intervention comparison, completeness, failures and effective-strength traces.

            **Budget:** at most {budget} scene/condition jobs plus reference checks. An eight-hour session limit
            permits resume; it is not a measured completion estimate. Each GPU must pass the 50-image memory pilot.
            The three notebooks repeat controls and some preparation; total GPU work
            is higher than the single shared-runner design. Parallel speedup has not been measured.

            **Shared masks:** first finish `Point3_00_Prepare_Masks_UpTo50.ipynb`. It imports compatible old masks,
            generates only missing ones, and publishes the complete cache. H1, H2 and H3 only read that cache;
            they do not install or run SAM. Missing masks stop this notebook with instructions to finish preparation.

            **Start:** keep all four notebooks and their source ZIP in `MyDrive/Point3_Hypothesis54`.
            Select a separate GPU runtime for each, using matching GPU models/software for a combined comparison.
            Only mask preparation needs `HF_TOKEN` with SAM 3 access. All three experiments can use a token for
            VGGT downloads, but do not require SAM access. G4 uses a separate CUDA 12.8 environment and must pass
            the same numerical reference gate; GPU speedups are unmeasured.
            Run cells in order. Keep the same run label when resuming. Do not run two copies of this same hypothesis
            into the same output folder. After mask preparation, hypotheses can run in any order.

            Forth Bridge uses the documented, user-authorized **bridge** anchor. Numeric mask checks and an all-view
            overlay are saved; visual mask review is still necessary before architectural claims.
        '''),code(settings),code(plan_cell),markdown('''
            ## Prepare environments

            Each hypothesis prepares VGGT only and reads the shared masks.
            Compatible environments are reused. Credentials are read from Secrets only when launching work.
        '''),code(setup),code(launcher),
        markdown(f'''
            ## Pilot for {h}

            Runs one calibration landmark with this notebook's {len(candidate_names(h))} conditions.
            A fresh scalar reference must match the accelerated run, including a reused model and cached inputs.
            Failure blocks the bulk run. The pilot prefers a calibration scene with the largest selected view count.
            There is no image-count reduction to fit GPU memory. Smaller datasets use their recorded available count.
            Each hypothesis performs
            its own pilot; compatible validated checkpoints are reused on resume.
        '''),code(pilot),markdown(f'''
            ## Run or resume {h}

            Finish {h} calibration, freeze its choices, then evaluate all 36 confirmation landmarks.
            This cell does not start the other hypotheses or wait for their reconstruction results.
            The preparation notebook must have completed the shared masks before this run starts.
            Validated checkpoints are reused after a timeout or disconnection. Failed landmarks stay visible.
        '''),copy.deepcopy(source['cells'][9]),markdown(f'''
            ## {h} results and mask review

            This notebook creates its own paired table, plot, summary and strength diagnostics.
            Partial results are descriptive; they do not establish a benefit across the full confirmation cohort.
            The two-point grids and one seed are preliminary experimental choices.
        '''),code('''
            import pandas as pd
            from IPython.display import display, Markdown, Image
            report_folder = OUTPUT/'reports'/HYPOTHESIS
            if (report_folder/'summary.json').exists():
                display(Markdown((report_folder/'REPORT.md').read_text()))
                print(json.dumps(json.loads((report_folder/'summary.json').read_text()), indent=2))
                paired = pd.read_csv(report_folder/'paired_results.csv')
                display(paired)
                if len(paired):
                    print('Mean structured − no-intervention:', paired['structured_minus_baseline'].mean())
                    plot = report_folder/'paired_differences.png'
                    if plot.exists(): display(Image(filename=str(plot)))
                    if HYPOTHESIS == 'H3':
                        display(paired.groupby(['anchor','anchor_fallback'])['structured_minus_scalar'].agg(['count','mean','median']))
                display(pd.read_csv(OUTPUT/'failures.csv'))
                display(pd.read_csv(OUTPUT/'strength_diagnostics.csv'))
            else: print('No report yet. Inspect notebook.log and scene failure logs.')
            bridge = hashlib.sha256('Forth_Bridge'.encode()).hexdigest()[:16]
            preview = OUTPUT/'scenes'/bridge/'mask_preview.jpg'
            if preview.exists():
                display(Markdown('### Forth Bridge — manual bridge anchor; review every selected-view overlay'))
                display(Image(filename=str(preview)))
                print(json.loads((preview.parent/'mask_validation.json').read_text()))
            else: print('Forth Bridge has not yet run here; review its masks in the preparation notebook.')
            print('Saved results:', OUTPUT)
            reuse = [json.loads(p.read_text()) for p in OUTPUT.glob('scenes/*/mask_reuse.json')]
            if reuse:
                print('Mask totals:', {key: sum(r.get(key, 0) for r in reuse) for key in ['reused','imported','generated']})
            print('Common reusable mask cache:', COMMON_MASKS)
        ''')]
        if h=='H3':
            cells.extend([markdown('''
                ## Optional combined comparison after all three notebooks finish

                Rerun the following cell once H1, H2 and H3 are complete. It performs no model inference.
                It checks that landmark splits, ordered images, masks, software/GPU profiles and shared control
                predictions/metrics agree. It also verifies that all selections still match calibration results.
                A mismatch blocks a combined comparison; it is never silently averaged away.
                This cell is only in H3, so concurrent notebooks do not overwrite the combined report.
            '''),code('''
                pending = []
                for hypothesis in ['H1','H2','H3']:
                    path = STUDY_ROOT/hypothesis/'summary.json'
                    if not path.exists() or json.loads(path.read_text()).get('status') != 'complete':
                        pending.append(hypothesis)
                if pending:
                    print('Combined report awaits:', ', '.join(pending), '. Rerun this cell after they finish.')
                else:
                    subprocess.run([str(EVAL_PY), str(STUDY/'compare_hypotheses.py'), '--root', str(STUDY_ROOT)],
                                   cwd=str(SOURCE), check=True)
                    display(pd.read_csv(STUDY_ROOT/'comparison/paired_hypotheses.csv'))
                    print(json.dumps(json.loads((STUDY_ROOT/'comparison/summary.json').read_text()), indent=2))
            ''')])
        name=f'Point3_{h}_Run_54.ipynb';notebook(dist/name,cells,True);books.append(name)
        scoped={**base_plan,'hypothesis':h,'candidate_definitions':{k:base_plan['candidate_definitions'][k] for k in candidate_names(h)}}
        budgets[h]=job_budget(scoped)
    save(dist/'hypothesis_delivery.json',{**delivery,'notebooks':books,'budget':budgets,
          'folder_name':'Point3_Hypothesis54','independent_jobs_upper_bound':sum(b['total_upper_bound'] for b in budgets.values())})
    print(json.dumps({'notebooks':books,'budgets':budgets,'source_zip':delivery['source_zip']},indent=2))


if __name__=='__main__':build_three()
