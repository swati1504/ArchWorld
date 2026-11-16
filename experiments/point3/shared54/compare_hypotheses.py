"""Verify comparability and join three independently executed hypothesis studies."""
import argparse
import json
from pathlib import Path
import statistics
import zipfile

from protocol import load,save,fingerprint,conditions,select_settings
from run import scene_dir,valid_saved
from prepare import file_hash
from fast_eval import compare_results
from full_dataset import METRICS
from report import write_csv


def cohort(plan):
    return {k:v for k,v in plan.items() if k not in ('hypothesis','candidate_definitions')}


def verify_study(output,hypothesis):
    study=load(output/'study.json');plan=study['plan']
    if plan.get('hypothesis')!=hypothesis:raise ValueError('Wrong hypothesis output folder')
    chosen=load(output/'selection.json')
    scores={};results={}
    for row in plan['rows']:
        if not valid_saved(row,output,study,conditions(row,chosen,hypothesis)):
            raise ValueError(f"{hypothesis} incomplete: {row['scene']}")
        result=load(scene_dir(output,row)/'results.json')['conditions']
        results[row['scene']]=result
        if row['split']=='calibration':scores[row['scene']]={k:v['CD_norm'] for k,v in result.items() if k!='no_intervention'}
    if select_settings(plan,scores)!=chosen:raise ValueError('Frozen selection no longer matches calibration checkpoints')
    return study,chosen,results


def verify_controls(reference,other,names):
    for name in names:
        with zipfile.ZipFile(reference/(name+'.zip')) as a,zipfile.ZipFile(other/(name+'.zip')) as b:
            read=lambda z,n:json.loads(z.read(n))
            for arm in ('base','inter'):
                compare_results(read(a,METRICS),read(b,METRICS),read(a,arm+'_evaluation.json'),read(b,arm+'_evaluation.json'))


def verify_inputs(reference,other):
    a,b=load(reference/'input_manifest.json'),load(other/'input_manifest.json')
    for key in ('input_key','views','archive_sha256','anchor','anchor_override','view_rule','files'):
        if a.get(key)!=b.get(key):raise ValueError(f'Inputs differ across hypotheses: {key}')
    masks=[]
    for folder in (reference,other):
        record=load(folder/'mask_validation.json')
        if record.get('numeric_checks')!='passed' or record['input_key']!=a['input_key']:
            raise ValueError('Missing matching mask validation')
        if file_hash(folder/'masks.zip')!=record.get('mask_archive_sha256'):
            raise ValueError('SAM mask archive changed or lacks a checksum')
        hashes=record.get('mask_sha256_by_file')
        if not hashes or len(hashes)!=len(a['views']):raise ValueError('Expected one audited SAM mask file per selected view')
        masks.append(hashes)
    if masks[0]!=masks[1]:raise ValueError('SAM masks differ across hypotheses; do not combine as a controlled comparison')


def compare(root):
    root=Path(root);loaded={h:verify_study(root/h,h) for h in ('H1','H2','H3')}
    reference,chosen,_=loaded['H1']
    plan=reference['plan'];common=fingerprint(cohort(plan))
    profile=load(root/'H1/execution_environment.json')
    if not any(r['split']=='confirmation' for r in plan['rows']):raise ValueError('No confirmation cohort')
    for h,(study,selection,results) in loaded.items():
        if fingerprint(cohort(study['plan']))!=common or study['source_sha256']!=reference['source_sha256']:
            raise ValueError('Source or cohort differs across hypotheses')
        if selection['global_scalar']!=chosen['global_scalar']:raise ValueError('Global scalar choices differ across hypotheses')
        if load(root/h/'execution_environment.json')!=profile:raise ValueError('GPU/software profiles differ across hypotheses')
    rows=[]
    for row in plan['rows']:
        original=scene_dir(root/'H1',row)
        names=['no_intervention','scalar_eta100','scalar_eta1000'] if row['split']=='calibration' else ['no_intervention',chosen['global_scalar']]
        for h in ('H2','H3'):
            other=scene_dir(root/h,row);verify_inputs(original,other);verify_controls(original,other,names)
        if row['split']!='confirmation':continue
        scene=row['scene'];values=loaded['H1'][2][scene]
        record={'scene':scene,'anchor':row['anchor'],'num_views':len(load(original/'input_manifest.json')['views']),
            'baseline':values['no_intervention']['CD_norm'],
            'scalar':values[chosen['global_scalar']]['CD_norm']}
        for h,(_,selection,results) in loaded.items():
            candidate=selection[h][row['anchor']] if h=='H3' else selection[h]
            record[h]=results[scene][candidate]['CD_norm']
            record[h+'_minus_scalar']=record[h]-record['scalar']
        record['H3_fallback']=row['anchor'] in loaded['H3'][1]['fallback_types']
        rows.append(record)
    folder=root/'comparison'
    write_csv(folder/'paired_hypotheses.csv',rows,['scene','anchor','num_views','baseline','scalar','H1','H1_minus_scalar',
              'H2','H2_minus_scalar','H3','H3_minus_scalar','H3_fallback'])
    summary={'status':'complete_and_comparable','confirmation_landmarks':len(rows),'cohort_sha256':common,
        'source_sha256':reference['source_sha256'],'global_scalar':chosen['global_scalar'],
        'checks':'same cohort, inputs, masks, hardware/software, control predictions and metrics',
        'mask_semantic_review':'still required; numerical agreement does not establish correct segmentation',
        'final_holdout_status':'not_run','hypotheses':{}}
    for h in ('H1','H2','H3'):
        differences=[r[h+'_minus_scalar'] for r in rows]
        summary['hypotheses'][h]={'mean_difference':statistics.mean(differences),
            'median_difference':statistics.median(differences),'wins':sum(d<0 for d in differences),
            'ties':sum(d==0 for d in differences),'losses':sum(d>0 for d in differences)}
    save(folder/'summary.json',summary)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);args=parser.parse_args()
    try:
        result=compare(args.root)
    except Exception as exc:
        save(args.root/'comparison/status.json',{'status':'not_comparable_or_incomplete','reason':str(exc)})
        raise
    save(args.root/'comparison/status.json',result)
    print(json.dumps(result,indent=2))
