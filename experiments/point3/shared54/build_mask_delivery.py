"""Dedicated shared-mask notebook; no VGGT inference or hypothesis selection."""
import textwrap
from build_delivery import code,markdown,notebook
from gpu_setup import setup_for


def build_mask_notebook(dist,source):
    settings=source['cells'][1]['source'].replace('Point3_Shared54','Point3_Hypothesis54')
    settings=settings.replace("OUTPUT = SHARED / 'results' / f'shared54_{SOURCE_SHA256[:12]}_cal{CALIBRATION_SIZE}_{RUN_LABEL}'",
                              "OUTPUT = SHARED / 'mask_preparation' / f'masks_{SOURCE_SHA256[:12]}_{RUN_LABEL}'")
    settings=settings.replace("SCRATCH = Path('/content/point3_shared54_work')", "SCRATCH = Path('/content/point3_mask_preparation_work')")
    settings+='''
COMMON_MASKS = SHARED/'common_masks'
MASK_SEARCH_ROOTS = [
    SHARED/'mask_preparation', SHARED/'results',
    Path('/content/drive/MyDrive/Point3_Shared54/results'),
    Path('/content/drive/MyDrive/AdaptiveAttend_Point3'),
    WORK/'results', WORK/'masks',
]
print('Shared masks for all hypotheses:', COMMON_MASKS)
'''
    setup=setup_for(source['cells'][4]['source'],'sam')
    setup="if MASKS_READY:\n    print('All masks are already verified; no model installation needed.')\nelse:\n"+textwrap.indent(setup,'    ')
    cells=[markdown('''
        # 00 — Prepare shared SAM masks once

        Run this notebook before H1, H2 and H3. It keeps the same frozen **54 landmarks**, using
        **all eligible distinct photographs up to 50 per landmark** and its assigned architectural anchor.
        Landmarks with fewer eligible views keep their actual count; none are replaced or padded.
        It imports compatible saved masks and generates only missing ones. It runs no reconstruction
        experiments and selects no intervention strengths. The 79 final holdout landmarks stay untouched.

        1. Keep this notebook and its matching source ZIP in `MyDrive/Point3_Hypothesis54`.
        2. Select a GPU runtime. G4 uses a separate CUDA 12.8 environment; A100/L4/T4 use the existing profile.
        3. Add `HF_TOKEN` in Colab Secrets, enable access for this notebook, and use approved SAM 3 access.
        4. Run cells in order. Wait for **READY: 54 landmarks / N masks**, and inspect the overlays.
        5. Start H1, H2 and H3 on separate GPU sessions, or sequentially on one GPU. All three only read masks.

        Run only one copy of mask preparation against this cache at a time. Progress persists in Drive;
        rerun with the same settings after interruption. The eight-hour session limit is not a runtime estimate.
        Masks are checkpointed after each ten newly processed images. Existing files are searched read-only.
        Earlier masks require matching RGB content, anchor and SAM settings; filenames alone are insufficient.
        Unknown-provenance files appear in the discovery report, without silently becoming experimental inputs.

        Fixed parameters: seed **1729**, SAM confidence **0.5**, pinned SAM revision and deterministic
        selection capped at 50. The up-to-50 rule was explicitly authorized on 2026-11-15 after encountering
        smaller scenes. Actual counts are saved in `view_counts.csv` and all hypothesis results.
        Changing settings can require new masks. Forth Bridge uses the authorized manual **bridge**
        anchor. Washington Monument uses the user-authorized **obelisk** correction (2026-11-15):
        the supplied **altar** anchor produced 0 nonempty masks out of 50. The original 18/36 split
        stays fixed, and obelisk uses H3's declared global-scalar fallback. The other 53 compatible
        completed preparations are verified and reused without downloading their inputs again.
        Numeric checks cannot establish semantic correctness; review every selected-view overlay.
        Zero depth/validity maps are preserved as views without evaluation support, as in the reference evaluator.
        Existing compatible masks, including completed scenes from the earlier notebook, are reused.
    '''),code(settings),code('''
        from protocol import make_plan, save
        from mask_cache import discover, ready_path, require_ready
        PLAN = make_plan()
        MASKS_READY = False
        if ready_path(COMMON_MASKS, PLAN).exists():
            READY = require_ready(COMMON_MASKS, PLAN, deep=True)
            MASKS_READY = True
            print('READY:', READY['landmarks'], 'landmarks /', READY['images'], 'masks')
            print('Original preparation and previews:', READY['preparation_output'])
        else:
            found = discover(MASK_SEARCH_ROOTS)
            save(OUTPUT/'mask_discovery_preflight.json', found)
            print('Documented archives:', len(found['sources']))
            print('Unverified locations:', len(found['unverified']))
            if found['unverified']: print(json.dumps(found['unverified'][:10], indent=2))
            print('Coverage is established by matching actual image contents during preparation.')
    '''),markdown('''
        ## Install or reuse the SAM environment

        Only SAM is installed. The token is read privately from Secrets when work starts.
        A complete, verified cache skips setup and generation entirely.
    '''),code(setup),markdown('''
        ## Import, generate missing masks, and resume

        Downloads each landmark archive as needed, selects up to 50 eligible distinct images,
        imports matching masks, generates missing ones, and saves validation and previews.
        Completed landmarks are skipped on resume; partial generation continues from saved masks.
        Any failure remains explicit and blocks the ready marker until all 54 landmarks succeed.
    '''),code('''
        if not MASKS_READY:
            from google.colab import userdata
            from hypothesis_study import bounded_run
            remaining = (DEADLINE-time.monotonic())/3600
            if remaining <= 0: raise RuntimeError('Session budget expired; rerun settings to resume.')
            environment = os.environ.copy()
            try:
                try:
                    environment['HF_TOKEN'] = userdata.get('HF_TOKEN')
                except Exception:
                    raise RuntimeError('Add HF_TOKEN in Colab Secrets, enable notebook access, and obtain SAM 3 approval.') from None
                environment['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '0'
                command = [SAM_PY, '-u', STUDY/'prepare_masks.py', '--output', OUTPUT, '--work', SCRATCH,
                           '--sam-python', SAM_PY, '--mask-cache', COMMON_MASKS, '--hours', str(remaining)]
                for root in MASK_SEARCH_ROOTS: command.extend(['--mask-search-root', root])
                bounded_run(command, OUTPUT/'notebook.log', SOURCE, environment, DEADLINE, echo=True)
            finally:
                environment.pop('HF_TOKEN', None)
        READY = require_ready(COMMON_MASKS, PLAN, deep=True)
        MASKS_READY = True
        print('READY:', READY['landmarks'], 'landmarks /', READY['images'], 'masks')
        print('H1, H2 and H3 can now reuse this cache:', COMMON_MASKS)
    '''),markdown('''
        ## Review coverage and masks

        Each scene has a mask archive, input identities, numeric validation, reuse counts and preview.
        Manual-anchor scenes (Forth Bridge and Washington Monument) show all selected views;
        other scenes show up to six previews. Empty SAM outputs are retained
        as valid no-detection records, but at least two views per scene must contain a detected anchor.
        Review other scene previews in the printed preparation folder as well.
    '''),code('''
        from IPython.display import display, Image, Markdown
        review_root = Path(READY['preparation_output']) if MASKS_READY else OUTPUT
        summary = review_root/'summary.json'
        info = {}
        if summary.exists():
            info = json.loads(summary.read_text())
            print({k:v for k,v in info.items() if k not in ('entries','cohort')})
        reuse = [json.loads(p.read_text()) for p in review_root.glob('scenes/*/mask_reuse.json')]
        print('Saved per-scene mask counts (latest attempt):',
              {k:sum(r.get(k,0) for r in reuse) for k in ['reused','imported','generated']})
        entries = READY['entries'] if MASKS_READY else info.get('entries', [])
        counts = [{'scene':e['identity']['scene'], 'num_views':len(e['identity']['images'])} for e in entries]
        display(counts)
        print('Landmarks below the 50-view cap:', [r for r in counts if r['num_views'] < 50])
        for scene, override in PLAN['anchor_overrides'].items():
            scene_id = hashlib.sha256(scene.encode()).hexdigest()[:16]
            preview = review_root/'scenes'/scene_id/'mask_preview.jpg'
            if preview.exists():
                display(Markdown(f"### {scene} — manual {override['anchor']} anchor; inspect all selected-view overlays"))
                display(Image(filename=str(preview)))
                print(json.loads((preview.parent/'mask_validation.json').read_text()))
        print('All previews, scene logs and provenance:', review_root)
        print('Shared reusable mask cache:', COMMON_MASKS)
    ''')]
    name='Point3_00_Prepare_Masks_UpTo50.ipynb';notebook(dist/name,cells,True)
    return name
