"""Three hypothesis reports from one shared result store; never reruns a model."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import zipfile

from protocol import load,save,conditions,policies,fingerprint


def write_csv(path,rows,fields):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def report(output,plots=True):
    from run import scene_dir,valid_saved
    output=Path(output);study=load(output/'study.json');plan=study['plan']
    selected=load(output/'selection.json') if (output/'selection.json').exists() else None
    scope=plan.get('hypothesis')
    if selected and selected['plan_sha256']!=fingerprint(plan):raise ValueError('Selection belongs to another study')
    records=[];failures=[];completed=[];diagnostics=[]
    for row in plan['rows']:
        saved=scene_dir(output,row)
        needed=conditions(row,selected,scope) if row['split']=='calibration' or selected else []
        ok=bool(needed) and valid_saved(row,output,study,needed)
        state='complete' if ok else 'failed' if (saved/'failure.json').exists() else 'pending'
        if not ok:
            failures.append({'scene':row['scene'],'split':row['split'],'status':state,
                'reason':load(saved/'failure.json')['reason'] if (saved/'failure.json').exists() else 'Not completed'})
            continue
        result=load(saved/'results.json')
        for name,values in result['conditions'].items():
            records.append({'scene':row['scene'],'split':row['split'],'anchor':row['anchor'],
                            'num_views':result['identity']['num_views'],'condition':name,**values})
            with zipfile.ZipFile(saved/(name+'.zip')) as z:
                trace=json.loads(z.read('strength_trace.json'))['layers']
            eta=[v for layer in trace for v in layer['eta_by_view']]
            diagnostics.append({'scene':row['scene'],'condition':name,'accepted_layers':sum(x['accepted'] for x in trace),
                'coefficient_min':min(eta),'coefficient_max':max(eta),
                'clipped_view_steps':sum(x['clipped_views'] for x in trace),
                'mean_applied_relative_update':statistics.mean(x['applied_masked_relative_update'] for x in trace)})
        if row['split']=='confirmation':completed.append((row,result))
    fields=['scene','split','anchor','num_views','condition','CD_norm_base','CD_norm','seconds']
    write_csv(output/'results.csv',records,fields)
    write_csv(output/'failures.csv',failures,['scene','split','status','reason'])
    write_csv(output/'strength_diagnostics.csv',diagnostics,['scene','condition','accepted_layers','coefficient_min',
        'coefficient_max','clipped_view_steps','mean_applied_relative_update'])
    calibration_total=sum(r['split']=='calibration' for r in plan['rows'])
    confirmation_total=sum(r['split']=='confirmation' for r in plan['rows'])
    status=('complete' if not failures and confirmation_total else
            'calibrated_only' if selected and not failures else 'partial')
    summary={'status':status,'calibration_landmarks':calibration_total,'confirmation_landmarks':confirmation_total,
        'confirmation_complete':len(completed),'incomplete_landmarks':len(failures),
        'final_holdout_landmarks':len(plan['heldout_ids']),'final_holdout_status':'not_run',
        'numeric_conditions_complete':len(records),'mask_review':'Review saved mask_preview.jpg files before architectural claims',
        'view_count_policy':plan.get('view_count_policy','exact'),
        'view_counts':{r['scene']:r['num_views'] for r in records},'hypotheses':{}}
    for h in ((scope,) if scope else ('H1','H2','H3')):
        rows=[]
        for meta,result in completed:
            mapping=policies(meta,selected);c=result['conditions']
            b,s,a=[c[mapping[k]]['CD_norm'] for k in ('no_intervention','global_scalar',h)]
            rows.append({'scene':meta['scene'],'anchor':meta['anchor'],'num_views':result['identity']['num_views'],
                'baseline':b,'scalar':s,'structured':a,
                'structured_minus_scalar':a-s,'structured_minus_baseline':a-b,
                'candidate':mapping[h],'scalar_candidate':mapping['global_scalar'],
                'anchor_fallback':h=='H3' and meta['anchor'] in selected['fallback_types'],
                'same_prediction_as_scalar':mapping[h]==mapping['global_scalar']})
        folder=output/'reports'/h
        write_csv(folder/'paired_results.csv',rows,['scene','anchor','num_views','baseline','scalar','structured',
            'structured_minus_scalar','structured_minus_baseline','candidate','scalar_candidate','anchor_fallback','same_prediction_as_scalar'])
        stats={'paired_landmarks':len(rows),'expected_landmarks':confirmation_total,'complete_cohort':status=='complete'}
        if rows:
            d=[r['structured_minus_scalar'] for r in rows]
            stats.update(mean_difference=statistics.mean(d),median_difference=statistics.median(d),
                wins=sum(x<0 for x in d),ties=sum(x==0 for x in d),losses=sum(x>0 for x in d))
            if h=='H3':
                learned=[r['structured_minus_scalar'] for r in rows if not r['anchor_fallback']]
                stats.update(learned_anchor_landmarks=len(learned),fallback_landmarks=len(rows)-len(learned),
                             mean_difference_supported_anchors=statistics.mean(learned) if learned else None)
        summary['hypotheses'][h]=stats
        save(folder/'summary.json',stats)
        text=f'# {h}: shared 54-landmark study\n\nStatus: **{status}**. Paired confirmation landmarks: {len(rows)}/{confirmation_total}.\n\n'
        text+='Lower normalized Chamfer distance is better. Negative structured-minus-scalar favors the structured rule.\n\n'
        text+='These are internal confirmation results; the 79 frozen final-evaluation IDs remain untouched. Partial cohorts are descriptive only.\n\n'
        text+='Selection uses calibration scores only. H3 falls back to the global scalar for anchors with fewer than two calibration landmarks; dome and minaret are fallback types in this subset.\n\n'
        text+='Each landmark uses all eligible distinct images up to 50; actual counts are in the tables. Comparisons are paired on identical images per landmark, and aggregate metrics give each landmark equal weight.\n\n'
        text+='Review masks, accepted-step traces, failures, and comparison against no intervention before interpreting differences.\n'
        (folder/'REPORT.md').write_text(text,encoding='utf-8')
        if plots and rows:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig,ax=plt.subplots(figsize=(9,max(3,len(rows)*.22)))
            ax.barh([r['scene'] for r in rows],[r['structured_minus_scalar'] for r in rows],color='#237e96')
            ax.axvline(0,color='black',linewidth=.8);ax.invert_yaxis()
            ax.set_xlabel('Structured − scalar normalized Chamfer distance; lower is better')
            ax.set_title(f'{h}: {len(rows)}/{confirmation_total} confirmation landmarks ({status})')
            fig.tight_layout();fig.savefig(folder/'paired_differences.png',dpi=180);plt.close(fig)
    save(output/'summary.json',summary)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    print(json.dumps(report(p.parse_args().output),indent=2))
