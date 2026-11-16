"""SHA-verified scene preparation with explicit, collision-safe image identities."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import random
import shutil
import struct
import sys
import tarfile
import zipfile

from protocol import HERE, SOURCE, fingerprint, load, save


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def safe_parts(name):
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '\\' in name or any(':' in x for x in p.parts):
        raise ValueError('Unsafe archive path')
    return p.parts


def derived_name(image, available, basename_counts):
    p = PurePosixPath(image)
    exact = [image+'.npy', str(p.with_suffix(''))+'.npy']
    found = sorted(set(exact) & set(available))
    if len(p.parts)==1 and basename_counts[p.name]>1:
        return None # Flat depth/mask identity is ambiguous across different archive paths.
    if len(found)==1: return found[0]
    if len(found)>1: raise ValueError(f'Ambiguous derived file for {image}')
    if basename_counts[p.name] == 1:
        possible = {p.name+'.npy', p.stem+'.npy'}
        found = sorted(name for name in available if name in possible)
        if len(found)==1: return found[0]
    return None


def choose_views(eligible, seed, scene, count=50, allow_fewer=False):
    # Duplicate contents are audited; only the reconstruction input selection is
    # deduplicated. The original 54-landmark collection remains unchanged.
    by_hash = {}
    for row in sorted(eligible,key=lambda r:r['original']):
        by_hash.setdefault(row['rgb_sha256'],row)
    pool = list(by_hash.values())
    if len(pool)<2:raise ValueError(f'{scene}: only {len(pool)} eligible distinct photographs; need at least two')
    if len(pool)<count and not allow_fewer:
        raise ValueError(f'{scene}: only {len(pool)} eligible distinct photographs; need exactly {count}')
    derived_seed = int(hashlib.sha256(f'{seed}:{scene}'.encode()).hexdigest()[:16],16)
    selected = random.Random(derived_seed).sample(pool,min(count,len(pool)))
    return sorted(selected,key=lambda r:r['original'])


def input_key(plan,row,archive):
    return fingerprint({'schema':'explicit50-v1','scene':row['scene'],'anchor':row['anchor'],
        'archive_sha256':archive['lfs']['oid'],'revision':plan['dataset_revision'],
        'views':plan['num_views'],'seed':plan['seed'],'view_rule':plan['view_rule'],
        'prepare_sha256':file_hash(Path(__file__)),'sam_revision':plan['sam_revision'],
        'confidence':plan['sam_confidence']})


def validate_prepared(folder):
    manifest=load(folder/'prepared.json')
    for name, expected in manifest['files'].items():
        if file_hash(folder/name)!=expected: raise ValueError(f'Prepared input changed: {name}')
    count=len(manifest['views'])
    if not 2<=count<=manifest.get('max_num_views',50) or len({r['rgb_sha256'] for r in manifest['views']})!=count:
        raise ValueError('Expected 2 to 50 distinct original photograph contents')
    if manifest.get('num_views',count)!=count:raise ValueError('Recorded view count differs from manifest')
    return manifest


def validate_evaluation_array(path,kind):
    import numpy as np
    arr=np.load(path,allow_pickle=False)
    if arr.ndim!=2 or min(arr.shape)<1 or not np.isfinite(arr).all():
        raise ValueError(f'Invalid {kind} array: {path.name}')
    # The reference evaluator discards pixels with zero depth or a false validity
    # mask. An entirely zero map is a valid no-support view, not a corrupt file.
    return {'file':path.name,'shape':list(arr.shape),'nonzero_pixels':int(np.count_nonzero(arr)),
            'positive_pixels':int(np.count_nonzero(arr>1e-4)) if kind=='depth' else None}


def write_images_binary_utf8(images,path):
    # COLMAP stores UTF-8 bytes followed by NUL. The upstream helper packs each
    # Unicode character as a single byte, which fails on unselected accented names.
    # Keep every record/pose/track, including unselected images, without changing
    # the reference evaluator's reader or metrics.
    with Path(path).open('wb') as handle:
        handle.write(struct.pack('<Q',len(images)))
        for im in images.values():
            if '\x00' in im.name:raise ValueError('Invalid NUL in COLMAP image name')
            if len(im.xys)!=len(im.point3D_ids):raise ValueError('COLMAP observation lengths differ')
            handle.write(struct.pack('<i4d3di',im.id,*im.qvec,*im.tvec,im.camera_id))
            handle.write(im.name.encode('utf-8')+b'\x00')
            handle.write(struct.pack('<Q',len(im.point3D_ids)))
            for xy,point_id in zip(im.xys,im.point3D_ids):handle.write(struct.pack('<ddq',*xy,point_id))


def prepare(request):
    from huggingface_hub import hf_hub_download
    import numpy as np
    from PIL import Image
    plan,row=request['plan'],request['row']
    folder=Path(request['prepared']);folder.mkdir(parents=True,exist_ok=True)
    if (folder/'prepared.json').exists(): return validate_prepared(folder)
    archive_info=load(HERE/'inputs/archive_sources.json')[row['scene']]
    if shutil.disk_usage(folder).free < max(16*2**30,archive_info['size']*2):
        raise OSError('Insufficient local scratch space for this scene archive')
    archive=hf_hub_download('ArchWorld/ArchWorld',archive_info['path'],repo_type='dataset',
        revision=plan['dataset_revision'],cache_dir=folder/'download',token=os.environ.get('HF_TOKEN') or None)
    if file_hash(archive)!=archive_info['lfs']['oid']: raise ValueError('Dataset archive SHA256 mismatch')
    scene=row['scene']
    maps={k:{} for k in ('images','depth','validity')}
    hashes={};sparse_members=set()
    sparse=folder/'data/archworld'/scene/'sparse';sparse.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive,'r:gz') as handle:
        for member in handle:
            parts=safe_parts(member.name)
            if not member.isfile():
                if member.isdir(): continue
                raise ValueError('Scene archives must not contain links or special files')
            if len(parts)>=4 and parts[:2]==('archworld',scene):
                kind=parts[2];rel='/'.join(parts[3:])
                if kind in ('images','depth'):
                    if rel in maps[kind]: raise ValueError('Repeated archive member')
                    maps[kind][rel]=member.name
                    if kind=='images':
                        h=hashlib.sha256()
                        with handle.extractfile(member) as f:
                            for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
                        hashes[rel]=h.hexdigest()
                elif kind=='sparse' and parts[-1] in ('images.bin','cameras.bin','points3D.bin'):
                    if parts[-1] in sparse_members:
                        raise ValueError('Ambiguous or repeated sparse reconstruction in archive')
                    sparse_members.add(parts[-1])
                    target=sparse/parts[-1]
                    if target.exists(): target.unlink() # Only files in this owned, incomplete preparation.
                    with handle.extractfile(member) as f,target.open('wb') as out:shutil.copyfileobj(f,out)
            elif len(parts)>=3 and parts[:2]==('manual_filtration_masks',scene):
                rel='/'.join(parts[2:])
                if rel in maps['validity']:raise ValueError('Repeated validity-mask archive member')
                maps['validity'][rel]=member.name
    spec=importlib.util.spec_from_file_location('point3_colmap_reader',SOURCE/'eval/read_write_model.py')
    reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)
    images=reader.read_images_binary(sparse/'images.bin')
    counts=Counter(PurePosixPath(n).name for n in maps['images'])
    eligible=[]
    for im in images.values():
        name=im.name
        if name not in maps['images']:continue
        depth=derived_name(name,maps['depth'],counts)
        validity=derived_name(name,maps['validity'],counts)
        if depth and validity:
            eligible.append({'original':name,'rgb_sha256':hashes[name], 'colmap_id':int(im.id),
                             'depth_source':depth,'validity_source':validity})
    selected=choose_views(eligible,plan['seed'],scene,plan['num_views'],
                          allow_fewer=plan.get('view_count_policy')=='up_to_available')
    print(f"{scene}: selected {len(selected)} views from {len({v['rgb_sha256'] for v in eligible})} eligible distinct images (cap {plan['num_views']})",flush=True)
    to_extract={}
    data=folder/'data';flat=data/'archworld'/scene
    for view in selected:
        ext=PurePosixPath(view['original']).suffix.lower()
        alias='p3_'+hashlib.sha256(view['original'].encode()).hexdigest()[:20]+ext
        view['alias']=alias
        for kind,original,target in (
            ('images',view['original'],flat/'images'/alias),
            ('depth',view['depth_source'],flat/'depth'/(alias+'.npy')),
            ('validity',view['validity_source'],data/'manual_filtration_masks'/scene/(alias+'.npy'))):
            to_extract.setdefault(maps[kind][original],[]).append(target)
    with tarfile.open(archive,'r:gz') as handle:
        for member in handle:
            if member.name not in to_extract:continue
            payload=handle.extractfile(member).read()
            for target in to_extract[member.name]:
                target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(payload)
    aliases={v['original']:v['alias'] for v in selected}
    renamed={key:im._replace(name=aliases.get(im.name,im.name)) for key,im in images.items()}
    write_images_binary_utf8(renamed,sparse/'images.bin')
    evaluation_arrays={'depth':[],'validity':[]}
    for view in selected:
        path=flat/'images'/view['alias']
        with Image.open(path) as im: im.convert('RGB').load()
        for kind,path in [('depth',flat/'depth'/(view['alias']+'.npy')),
                          ('validity',data/'manual_filtration_masks'/scene/(view['alias']+'.npy'))]:
            evaluation_arrays[kind].append(validate_evaluation_array(path,kind))
    shutil.copy2(HERE/'inputs/ArchWorld_metadata.csv',data/'ArchWorld_metadata.csv')
    manifest={'input_key':request['input_key'],'scene':scene,'anchor':row['anchor'],
        'anchor_override':plan['anchor_overrides'].get(scene),'views':selected,
        'sam_profile':{'sam_revision':plan['sam_revision'],'confidence':plan['sam_confidence']},
        'archive_sha256':archive_info['lfs']['oid'], 'original_rgb_count':len(maps['images']),
        'registered_eligible_files':len(eligible),'eligible_distinct_contents':len({v['rgb_sha256'] for v in eligible}),
        'num_views':len(selected),'max_num_views':plan['num_views'],
        'view_count_policy':plan.get('view_count_policy','exact'),
        'evaluation_arrays':evaluation_arrays,
        'zero_depth_views':sum(r['positive_pixels']==0 for r in evaluation_arrays['depth']),
        'zero_validity_views':sum(r['nonzero_pixels']==0 for r in evaluation_arrays['validity']),
        'view_rule':plan['view_rule'],'files':{p.relative_to(folder).as_posix():file_hash(p) for p in data.rglob('*') if p.is_file()}}
    save(folder/'prepared.json',manifest)
    # Reclaim only the download cache owned by this preparation; canonical inputs stay.
    owned=(folder/'download').resolve()
    if owned.parent!=folder.resolve():raise ValueError('Invalid owned download cache')
    shutil.rmtree(owned)
    return manifest


def mask_paths(folder,manifest):
    return [folder/'masks'/manifest['scene']/'mask'/(Path(v['alias']).stem+'_'+manifest['anchor']+'.npy') for v in manifest['views']]


def masks(request):
    import numpy as np
    from PIL import Image,ImageDraw
    folder=Path(request['prepared']);manifest=validate_prepared(folder)
    saved=Path(request['saved']);saved.mkdir(parents=True,exist_ok=True)
    paths=mask_paths(folder,manifest)
    if not request.get('mask_cache') and (saved/'masks.zip').exists():
        with zipfile.ZipFile(saved/'masks.zip') as z:
            if z.testzip() is not None:raise ValueError('Corrupt mask cache')
            if json.loads(z.read('identity.json'))!={'input_key':request['input_key']}:
                raise ValueError('Mask cache belongs to different images/anchor/SAM settings')
            for path in paths:
                name=path.relative_to(folder/'masks').as_posix()
                path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(z.read(name))
    for path in paths:
        if path.exists():
            try:np.load(path,allow_pickle=False)
            except (ValueError,EOFError,OSError):path.unlink()
    def generate_missing(views,checkpoint):
        import torch
        random.seed(request['plan']['seed']);np.random.seed(request['plan']['seed'])
        torch.manual_seed(request['plan']['seed']);torch.cuda.manual_seed_all(request['plan']['seed'])
        sys.path.insert(0,str(SOURCE))
        from intervention import make_masks as m
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        m.elements_for=lambda scene,all_elements=False:[manifest['anchor']]
        processor=Sam3Processor(build_sam3_image_model(device='cuda'),confidence_threshold=request['plan']['sam_confidence'])
        for start in range(0,len(views),10):
            chunk=views[start:start+10]
            m.views_for=lambda scene:[v['alias'] for v in chunk]
            m.segment_scene(processor,folder/'data',manifest['scene'],folder/'masks',False,True)
            checkpoint() # Persist validated partial masks for interruption/resume.
    if request.get('mask_cache'):
        from mask_cache import ensure
        reuse=ensure(request,manifest,paths,generate_missing)
        save(saved/'mask_reuse.json',reuse)
        print(f"Masks ready: {reuse['reused']} reused, {reuse['generated']} generated, {reuse['imported']} imported",flush=True)
    elif not all(p.exists() for p in paths):
        generate_missing([v for v,p in zip(manifest['views'],paths) if not p.exists()],lambda:None)
    count=len(manifest['views'])
    preview_count=count if manifest.get('anchor_override') else min(6,count)
    columns=5 if preview_count>6 else 3
    nonempty=0;coverage=[]
    overlay=Image.new('RGB',(columns*300,((preview_count+columns-1)//columns)*240),'white');draw=ImageDraw.Draw(overlay)
    for i,(view,path) in enumerate(zip(manifest['views'],paths)):
        arr=np.load(path,allow_pickle=False)
        with Image.open(folder/'data/archworld'/manifest['scene']/'images'/view['alias']) as im:
            image=im.convert('RGB')
        if arr.size:
            if arr.ndim<2 or not np.isfinite(arr).all():raise ValueError('Invalid SAM mask')
            if not np.isin(arr,[0,1]).all():raise ValueError('Expected binary SAM masks')
            union=np.any(arr,axis=tuple(range(arr.ndim-2))) if arr.ndim>2 else arr.astype(bool)
            if union.shape!=(image.height,image.width):raise ValueError('SAM mask/image dimensions differ')
            nonempty+=bool(union.any())
        coverage.append({'view':i,'alias':view['alias'],'fraction':float(union.mean()) if arr.size else 0.})
        if i<preview_count:
            if arr.size:
                alpha=Image.fromarray((union*95).astype('uint8')).resize(image.size)
                image=Image.composite(Image.new('RGB',image.size,'red'),image,alpha)
            image.thumbnail((300,210));x,y=(i%columns)*300,(i//columns)*240
            overlay.paste(image,(x,y));draw.text((x+4,y+214),f'View {i} / {manifest["anchor"]}',fill='black')
    if nonempty<2:raise ValueError(f'Anchor present in only {nonempty}/{count} views')
    overlay.save(saved/'mask_preview.jpg')
    temporary=saved/'masks.zip.tmp'
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('identity.json',json.dumps({'input_key':request['input_key']}))
        for path in paths:z.write(path,path.relative_to(folder/'masks').as_posix())
    with zipfile.ZipFile(temporary) as z:
        if z.testzip() is not None:raise ValueError('Mask cache verification failed')
    temporary.replace(saved/'masks.zip')
    save(saved/'mask_validation.json',{'files':count,'nonempty_views':nonempty,'numeric_checks':'passed',
        'visual_review':'required before interpreting architectural effects','input_key':request['input_key'],
        'anchor':manifest['anchor'],'manual_override':manifest.get('anchor_override'),
        'preview_views':preview_count,'coverage_by_view':coverage,
        'mask_archive_sha256':file_hash(saved/'masks.zip'),
        'mask_sha256_by_file':{p.relative_to(folder/'masks').as_posix():file_hash(p) for p in paths}})
    shutil.copy2(folder/'prepared.json',saved/'input_manifest.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','masks']);parser.add_argument('request',type=Path)
    args=parser.parse_args();request=load(args.request)
    (prepare if args.action=='prepare' else masks)(request)
