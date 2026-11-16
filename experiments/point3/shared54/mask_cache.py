"""Content-verified SAM cache: preparation publishes; all hypotheses only read."""
import hashlib
import io
import json
import os
from pathlib import Path
import time
import uuid
import zipfile

from protocol import HERE,fingerprint,load,save


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def profile(plan):
    return {'sam_revision':plan['sam_revision'],'confidence':plan['sam_confidence']}


def identity(plan,manifest):
    # Image contents and SAM settings determine reuse, independently of notebook,
    # source-bundle hash, aliases, view order or intervention coefficients.
    return {'schema':'point3-sam-cache-v1','scene':manifest['scene'],'anchor':manifest['anchor'],
            'profile':profile(plan),'images':sorted(v['rgb_sha256'] for v in manifest['views'])}


def cohort_identity(plan):
    archives=load(HERE/'inputs/archive_sources.json')
    return {'schema':'point3-mask-cohort-v1','dataset_revision':plan['dataset_revision'],
            'num_views':plan['num_views'],'seed':plan['seed'],'view_rule':plan['view_rule'],
            'view_count_policy':plan.get('view_count_policy','exact'),
            'profile':profile(plan),'scenes':[
                {'scene':r['scene'],'anchor':r['anchor'],'archive_sha256':archives[r['scene']]['lfs']['oid']}
                for r in sorted(plan['rows'],key=lambda r:r['scene'])]}


def ready_path(cache,plan):
    return Path(cache)/'ready'/(fingerprint(cohort_identity(plan))+'.json')


def check_entry(cache,plan,entry,deep=False):
    definition=entry['identity'];rows={r['scene']:r for r in plan['rows']}
    row=rows.get(definition['scene'])
    if not row or definition['anchor']!=row['anchor'] or definition['profile']!=profile(plan):
        raise ValueError('Shared mask scene/anchor/SAM settings do not match the plan')
    count=len(definition['images'])
    minimum=plan.get('min_views',2) if plan.get('view_count_policy')=='up_to_available' else plan['num_views']
    if not minimum<=count<=plan['num_views'] or len(set(definition['images']))!=count:
        raise ValueError('Shared mask set does not cover the selected distinct images')
    archive=Path(cache)/'sets'/fingerprint(definition)/'masks.zip'
    with zipfile.ZipFile(archive) as z:
        record=json.loads(z.read('record.json'))
        if (record['identity']!=definition or not record['complete'] or record['nonempty_views']<2 or
                set(record['hashes'])!=set(definition['images'])):
            raise ValueError('Shared mask set is incomplete or has a different identity')
        if deep:
            if sha(archive)!=entry['archive_sha256'] or z.testzip() is not None:
                raise ValueError('Shared mask archive checksum mismatch')
            for key,expected in record['hashes'].items():
                if hashlib.sha256(z.read(key+'.npy')).hexdigest()!=expected:
                    raise ValueError('Shared mask payload checksum mismatch')
    return record


def require_ready(cache,plan,deep=False):
    path=ready_path(cache,plan)
    if not path.exists():
        raise FileNotFoundError('Run Point3_00_Prepare_Masks_UpTo50.ipynb to completion before H1, H2 or H3.')
    ready=load(path)
    if ready.get('cohort')!=cohort_identity(plan) or ready.get('status')!='ready':
        raise ValueError('Mask preparation belongs to a different cohort or is incomplete')
    entries=ready['entries']
    if len(entries)!=len(plan['rows']) or {e['identity']['scene'] for e in entries}!={r['scene'] for r in plan['rows']}:
        raise ValueError('Mask preparation does not cover every landmark')
    for entry in entries:check_entry(cache,plan,entry,deep)
    return ready


def publish_ready(cache,plan,entries,output):
    if len(entries)!=len(plan['rows']) or {e['identity']['scene'] for e in entries}!={r['scene'] for r in plan['rows']}:
        raise ValueError('Cannot publish an incomplete mask cohort')
    for entry in entries:check_entry(cache,plan,entry,deep=True)
    result={'status':'ready','cohort':cohort_identity(plan),'entries':entries,
            'landmarks':len(entries),'images':sum(len(e['identity']['images']) for e in entries),
            'view_counts':{e['identity']['scene']:len(e['identity']['images']) for e in entries},
            'preparation_output':str(output),'visual_review':'required; numeric checks are not semantic review'}
    save(ready_path(cache,plan),result)
    return result


def checked_array(payload,image):
    import numpy as np
    from PIL import Image
    arr=np.load(io.BytesIO(payload),allow_pickle=False)
    if not arr.size:
        if arr.ndim!=1:raise ValueError('Malformed empty SAM mask')
        return False
    if arr.ndim<2 or not np.isfinite(arr).all() or not np.isin(arr,[0,1]).all():
        raise ValueError('Invalid binary SAM mask')
    with Image.open(image) as im:
        if arr.shape[-2:]!=(im.height,im.width):raise ValueError('SAM mask/image dimensions differ')
    return bool(arr.any())


def image_path(folder,manifest,view):
    return folder/'data/archworld'/manifest['scene']/'images'/view['alias']


def discover(roots):
    sources=[];loose=[];missing=[];seen=set();visited=0
    excluded={'envs','third_party','.git','__pycache__','common_masks','download','node_modules'}
    for root in roots:
        root=Path(root)
        if not root.exists():missing.append(str(root));continue
        for current,dirs,files in os.walk(root,followlinks=False):
            dirs[:]=[d for d in dirs if d not in excluded]
            current=Path(current);resolved=str(current.resolve())
            if resolved in seen:dirs[:]=[];continue
            seen.add(resolved);visited+=1
            if visited>20000:raise ValueError('Mask search too broad; narrow MASK_SEARCH_ROOTS')
            if 'masks.zip' in files:
                if 'input_manifest.json' not in files or 'mask_validation.json' not in files:
                    loose.append({'path':str(current/'masks.zip'),'reason':'Missing image-content/SAM provenance; not auto-imported'})
                    continue
                try:
                    manifest=load(current/'input_manifest.json');settings=manifest.get('sam_profile')
                    if settings is None:
                        for parent in [current,*list(current.parents)[:5]]:
                            if (parent/'study.json').exists():
                                settings=profile(load(parent/'study.json')['plan']);break
                    if not settings:raise ValueError('No recorded SAM settings')
                    sources.append({'folder':str(current),'scene':manifest['scene'],'anchor':manifest['anchor'],
                                    'profile':settings})
                except (ValueError,KeyError,OSError) as exc:
                    loose.append({'path':str(current),'reason':str(exc)})
            if current.name=='mask' and any(n.endswith('.npy') for n in files):
                loose.append({'path':str(current),'files':sum(n.endswith('.npy') for n in files),
                              'reason':'Loose masks found; image hashes and SAM settings need provenance before import'})
    return {'sources':sorted(sources,key=lambda s:s['folder']),'unverified':loose,'missing_roots':missing,
            'note':'Discovery is read-only; original masks are never changed'}


def import_sources(sources,plan,manifest,folder,paths,origins):
    rejected=[];count=0;wanted={v['rgb_sha256']:(v,p) for v,p in zip(manifest['views'],paths)}
    for source in sources:
        if source['scene']!=manifest['scene']:continue
        if source['anchor']!=manifest['anchor'] or source['profile']!=profile(plan):
            rejected.append({'source':source['folder'],'reason':'Different anchor or SAM settings'});continue
        source_dir=Path(source['folder'])
        try:
            old=load(source_dir/'input_manifest.json');validation=load(source_dir/'mask_validation.json')
            if old['scene']!=manifest['scene'] or old['anchor']!=manifest['anchor']:raise ValueError('Source identity changed')
            if validation.get('numeric_checks')!='passed' or validation['input_key']!=old['input_key']:
                raise ValueError('Source masks lack matching successful validation')
            archive=source_dir/'masks.zip'
            if validation.get('mask_archive_sha256') and sha(archive)!=validation['mask_archive_sha256']:
                raise ValueError('Source archive checksum mismatch')
            with zipfile.ZipFile(archive) as z:
                if z.testzip() is not None:raise ValueError('Corrupt source masks')
                if json.loads(z.read('identity.json'))!={'input_key':old['input_key']}:
                    raise ValueError('Source archive and input manifest differ')
                for old_view in old['views']:
                    match=wanted.get(old_view['rgb_sha256'])
                    if not match:continue
                    view,target=match
                    if target.exists():continue
                    name=f"{old['scene']}/mask/{Path(old_view['alias']).stem}_{old['anchor']}.npy"
                    payload=z.read(name)
                    expected=validation.get('mask_sha256_by_file',{}).get(name)
                    if expected and hashlib.sha256(payload).hexdigest()!=expected:raise ValueError('Source mask checksum mismatch')
                    checked_array(payload,image_path(folder,manifest,view))
                    target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(payload)
                    origins[view['rgb_sha256']]={'kind':'imported','source':str(archive),'source_image':old_view['original']}
                    count+=1
        except (KeyError,ValueError,OSError,zipfile.BadZipFile) as exc:
            rejected.append({'source':str(source_dir),'reason':str(exc)})
    return count,rejected


def restore(archive,definition,manifest,folder,paths,complete):
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise ValueError('Corrupt shared mask cache')
        record=json.loads(z.read('record.json'))
        if record['identity']!=definition:raise ValueError('Shared masks belong to different inputs/SAM settings')
        if complete and (not record['complete'] or set(record['hashes'])!=set(definition['images'])):
            raise ValueError('Shared mask set is incomplete')
        for view,target in zip(manifest['views'],paths):
            key=view['rgb_sha256']
            if key not in record['hashes']:continue
            payload=z.read(key+'.npy')
            if hashlib.sha256(payload).hexdigest()!=record['hashes'][key]:raise ValueError('Shared mask checksum mismatch')
            checked_array(payload,image_path(folder,manifest,view))
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(payload)
    return record


def publish(archive,definition,manifest,folder,paths,origins,complete=False):
    archive.parent.mkdir(parents=True,exist_ok=True)
    temporary=archive.with_name(archive.name+'.'+uuid.uuid4().hex+'.tmp')
    hashes={};nonempty=0
    try:
        with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as z:
            for view,path in zip(manifest['views'],paths):
                if not path.exists():continue
                payload=path.read_bytes();nonempty+=checked_array(payload,image_path(folder,manifest,view))
                key=view['rgb_sha256'];hashes[key]=hashlib.sha256(payload).hexdigest();z.writestr(key+'.npy',payload)
            if complete and (len(hashes)!=len(paths) or nonempty<2):
                raise ValueError(f"Cannot publish incomplete/unusable masks: {manifest['scene']} / anchor={manifest['anchor']!r}; "
                                 f"{len(hashes)}/{len(paths)} files, {nonempty} nonempty views (at least 2 required). "
                                 'A resume reuses these cached masks; inspect the anchor and masks before retrying.')
            record={'identity':definition,'hashes':hashes,'origins':origins,'complete':complete,'nonempty_views':nonempty}
            z.writestr('record.json',json.dumps(record))
        with zipfile.ZipFile(temporary) as z:
            if z.testzip() is not None:raise ValueError('Mask publication checksum failed')
        temporary.replace(archive)
    finally:temporary.unlink(missing_ok=True)


def ensure(request,manifest,paths,generate):
    folder=Path(request['prepared']);definition=identity(request['plan'],manifest)
    cache=Path(request['mask_cache'])/'sets'/fingerprint(definition)
    complete=cache/'masks.zip';partial=cache/'partial.zip'
    role=request['mask_role'];origins={};generated=0;imported=0;rejected=[]
    if role not in ('build','read'):raise ValueError('Unknown shared mask role')
    if role=='read' and not complete.exists():
        raise FileNotFoundError(f"Missing masks for {manifest['scene']}; finish Point3_00_Prepare_Masks_UpTo50.ipynb first")
    if complete.exists():
        record=restore(complete,definition,manifest,folder,paths,True)
        return {'mode':'shared_cache','reused':len(paths),'generated':0,'imported':0,
                'cache':str(complete),'origins':record['origins']}
    if partial.exists():origins.update(restore(partial,definition,manifest,folder,paths,False)['origins'])
    # Local partial files must also be validated before they count as reusable.
    for view,path in zip(manifest['views'],paths):
        if not path.exists():continue
        try:checked_array(path.read_bytes(),image_path(folder,manifest,view))
        except (ValueError,EOFError,OSError):path.unlink();continue
        origins.setdefault(view['rgb_sha256'],{'kind':'resumed_local'})
    imported,rejected=import_sources(request.get('mask_sources',[]),request['plan'],manifest,folder,paths,origins)
    missing=[view for view,path in zip(manifest['views'],paths) if not path.exists()]
    def checkpoint():publish(partial,definition,manifest,folder,paths,origins)
    checkpoint()
    if missing:
        for view in missing:origins[view['rgb_sha256']]={'kind':'generated','profile':profile(request['plan'])}
        print(f"{manifest['scene']}: reusing {len(paths)-len(missing)}/{len(paths)} masks; generating {len(missing)} missing masks",flush=True)
        generate(missing,checkpoint)
        generated=len(missing)
    if not all(p.exists() for p in paths):raise ValueError('SAM did not produce every missing mask; partial cache is saved')
    publish(complete,definition,manifest,folder,paths,origins,True)
    return {'mode':'published_shared','reused':len(paths)-generated,'generated':generated,'imported':imported,
            'cache':str(complete),'origins':origins,'rejected_sources':rejected}
