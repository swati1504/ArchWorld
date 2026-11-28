"""Render measured aligned point clouds with a shared camera, framing and crop.

Scientific rasterization only: no geometry modification or image generation.
Run with --geometry DIRECTORY --output DIRECTORY --scene 'Readable name'
and --split calibration|confirmation. Requires NumPy, Pillow and ReportLab.
"""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

def font(size, bold=False):
    # Preserve Arial where available; record platform font differences when rerendering.
    from matplotlib.font_manager import FontProperties, findfont
    path = findfont(FontProperties(family=['Arial', 'DejaVu Sans'], weight='bold' if bold else 'normal'))
    return ImageFont.truetype(path, size)

def render(points, colors, basis, bounds, width, height):
    projected=np.asarray(points,dtype=np.float64)@basis
    low,high=np.asarray(bounds)
    x=np.rint((projected[:,0]-low[0])/(high[0]-low[0])*(width-1)).astype(np.int64)
    y=np.rint((high[1]-projected[:,1])/(high[1]-low[1])*(height-1)).astype(np.int64)
    inside=(x>=1)&(x<width-1)&(y>=1)&(y<height-1)
    x,y,z,rgb=x[inside],y[inside],projected[inside,2],np.asarray(colors)[inside,:3]
    order=np.argsort(z,kind='stable');x,y,z,rgb=x[order],y[order],z[order],rgb[order]
    depth=np.full(width*height,np.inf);pixels=np.full((width*height,3),255,dtype=np.uint8)
    for dx,dy in [(0,0),(-1,0),(1,0),(0,-1),(0,1)]:
        index,first=np.unique((y+dy)*width+x+dx,return_index=True)
        nearer=z[first]<depth[index];index,first=index[nearer],first[nearer]
        depth[index]=z[first];pixels[index]=rgb[first]
    return Image.fromarray(pixels.reshape(height,width,3)),float(inside.mean())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--geometry',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--scene',required=True)
    ap.add_argument('--split',choices=['calibration','confirmation'],required=True)
    ap.add_argument('--camera',type=int,default=3)
    ap.add_argument('--hypothesis',choices=['H1','H2','H3'],default='H1')
    ap.add_argument('--condition')
    ap.add_argument('--scalar',default='scalar_eta1000')
    ap.add_argument('--frame-quantile',type=float,default=.001)
    ap.add_argument('--zoom',type=float,nargs=4,metavar=('XMIN','YMIN','XMAX','YMAX'))
    args=ap.parse_args()
    g=args.geometry;out=args.output;out.mkdir(parents=True,exist_ok=True)
    selected=args.condition or {'H1':'layer_alpha0.02','H2':'view_eta100','H3':'scalar_eta100'}[args.hypothesis]
    assert selected!=args.scalar, 'Choose a case with a distinct hypothesis result.'
    conditions=['no_intervention',selected,args.scalar]
    results=json.loads((g/'results.json').read_text());scores=results['conditions']
    clouds={};provenance={'scene':results['identity']['scene'],'split':args.split,'hypothesis':args.hypothesis,
                         'condition':selected,'scalar_condition':args.scalar,'source_files':{}}
    for c in conditions:
        v=json.loads((g/(c+'_verification.json')).read_text())
        assert v['status']=='passed' and v['prediction_hashes_exact'] and v['metrics_match']
        assert v['identity']['scene']==results['identity']['scene']
        path=g/(c+'_inter.npz');clouds[c]=np.load(path,allow_pickle=False)
        provenance['source_files'][path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    assert scores[selected]['CD_norm'] < scores['no_intervention']['CD_norm']
    assert results['split']==args.split
    cameras=json.loads((g/'camera_views.json').read_text());cam=cameras[args.camera]
    c2w=np.asarray(cam['cam_to_world']);basis=np.column_stack([c2w[:3,0],-c2w[:3,1],c2w[:3,2]])
    np.testing.assert_allclose(basis.T@basis,np.eye(3),atol=1e-6)
    # Fit all conditions and the reference jointly, preserving relative scale/location.
    assert 0 <= args.frame_quantile < .05
    quantiles=[args.frame_quantile,1-args.frame_quantile]
    qs=[np.quantile(clouds[c]['points']@basis[:,:2],quantiles,axis=0) for c in conditions]
    reference_path=g/'ground_truth.npz'
    reference=np.load(reference_path,allow_pickle=False)
    qs.append(np.quantile(reference['points']@basis[:,:2],quantiles,axis=0))
    provenance['reference_sha256']=hashlib.sha256(reference_path.read_bytes()).hexdigest()
    provenance['framing']=f'Joint {quantiles[0]}/{quantiles[1]} projected quantile bounds of three conditions and reference, with 8% padding'
    provenance['frame_quantiles']=quantiles
    lo=np.min([q[0] for q in qs],axis=0);hi=np.max([q[1] for q in qs],axis=0)
    center=(lo+hi)/2;span=(hi-lo)*1.08
    # Wide full view avoids adding vertical whitespace around a long building.
    width,height=1500,700;aspect=width/height
    if span[0]/span[1]<aspect:span[0]=span[1]*aspect
    else:span[1]=span[0]/aspect
    bounds=np.array([center-span/2,center+span/2])
    # Existing camera-3 clock-tower crop, expressed in world-space projection.
    # It is the exact same rectangle for all conditions, not a per-panel fit.
    if args.zoom:
        zoom=np.asarray(args.zoom).reshape(2,2)
    else:
        # Default detail region is central; inspect and set a documented common crop if needed.
        zoom=np.array([bounds[0]+span*np.array([.45,.2]),bounds[0]+span*np.array([.85,.8])])
    x0=int((zoom[0,0]-bounds[0,0])/span[0]*width);x1=int((zoom[1,0]-bounds[0,0])/span[0]*width)
    y0=int((bounds[1,1]-zoom[1,1])/span[1]*height);y1=int((bounds[1,1]-zoom[0,1])/span[1]*height)
    assert 0<=x0<x1<=width and 0<=y0<y1<=height
    rasters={};fractions={}
    for c in conditions:
        im,f=render(clouds[c]['points'],clouds[c]['colors'],basis,bounds,width,height)
        im.save(out/(c+'.png'));rasters[c]=im;fractions[c]=f
    # Main figure follows the paper's before/after plus matched detail layout.
    W,H=2400,1840;fig=Image.new('RGB',(W,H),'white');d=ImageDraw.Draw(fig)
    dark='#182532';muted='#536274';red='#E05353';orange='#D98B16'
    d.text((65,40),args.scene,font=font(49,True),fill=dark)
    gain=100*(1-scores[selected]['CD_norm']/scores['no_intervention']['CD_norm'])
    setting={'H1':'layer strength, alpha = '+selected.removeprefix('layer_alpha'),
             'H2':'view strength, base eta = '+selected.removeprefix('view_eta'),
             'H3':'anchor setting, eta = '+selected.removeprefix('scalar_eta')}[args.hypothesis]
    d.text((65,105),f"VGGT | {args.hypothesis} {setting} | {len(results['identity']['views'])} matched images | {args.split} example",font=font(26),fill=muted)
    d.text((1670,44),f'{gain:.2f}% lower error',font=font(39,True),fill='#12756F')
    d.text((1670,98),'versus VGGT without intervention',font=font(24),fill=muted)
    main_left,main_width=230,1510;detail_left,detail_width=1880,420
    d.text((main_left,165),'Shared scene crop' if args.frame_quantile>.001 else 'Full reconstruction',font=font(25,True),fill=dark)
    if args.frame_quantile>.001:
        d.text((main_left+330,165),'Distant outliers remain in the supplied wide overview',font=font(23),fill=muted)
    d.text((detail_left,165),'Same detail region',font=font(25,True),fill=dark)
    for row,(c,title,color) in enumerate([('no_intervention','Before',red),(selected,'After '+args.hypothesis,orange)]):
        top=215+row*750
        d.text((65,top+245),title,font=font(33,True),fill=dark)
        d.text((65,top+292),'CD norm',font=font(23),fill=muted)
        d.text((65,top+324),f"{scores[c]['CD_norm']:.3f}",font=font(35,True),fill=dark)
        # Resize complete render uniformly; detail is a direct nearest-neighbor crop.
        scale=main_width/width;display_height=round(height*scale)
        fit=rasters[c].resize((main_width,display_height),Image.Resampling.LANCZOS)
        top_im=top+(720-display_height)//2
        fig.paste(fit,(main_left,top_im))
        detail=rasters[c].crop((x0,y0,x1,y1))
        detail_scale=min(detail_width/detail.width,650/detail.height)
        assert detail_scale>scale, 'Choose a smaller common detail region so the inset is enlarged.'
        detail=detail.resize((round(detail.width*detail_scale),round(detail.height*detail_scale)),Image.Resampling.NEAREST)
        dx=detail_left+(detail_width-detail.width)//2;dy=top+(720-detail.height)//2
        fig.paste(detail,(dx,dy))
        box=(main_left+round(x0*scale),top_im+round(y0*scale),main_left+round(x1*scale),top_im+round(y1*scale))
        d.rectangle(box,outline=color,width=4);d.rectangle((dx,dy,dx+detail.width,dy+detail.height),outline=color,width=4)
        d.line((box[2],box[1],dx,dy),fill=color,width=2)
        d.line((box[2],box[3],dx,dy+detail.height),fill=color,width=2)
    scalar=scores[args.scalar]['CD_norm'];hypothesis_score=scores[selected]['CD_norm']
    change=100*(hypothesis_score/scalar-1)
    d.rectangle((0,1710,W,H),fill='#F1F4F7')
    d.text((65,1727),'Same camera, framing, colors and point size. Zooms are identical crops. No geometry smoothing.',font=font(23),fill=dark)
    d.text((65,1762),f'Scalar control: {scalar:.3f}. {args.hypothesis} is {abs(change):.2f}% '+('worse' if change>0 else 'better')+' than scalar on this landmark.',font=font(23),fill=dark)
    d.text((65,1798),f'Selected {args.split} example. CD norm is measured on the whole landmark, not only the boxed detail.',font=font(21),fill=dark)
    png=out/'before_after.png';fig.save(png,dpi=(240,240))
    pdf=out/'before_after.pdf';cv=canvas.Canvas(str(pdf),pagesize=(W/3,H/3));cv.drawImage(ImageReader(fig),0,0,W/3,H/3);cv.showPage();cv.save()
    provenance.update(camera=cam,basis=basis.tolist(),shared_bounds=bounds.tolist(),zoom_world_bounds=zoom.tolist(),
        detail_magnification=detail_scale/scale,
        in_frame_fraction=fractions,point_splat='Fixed five-pixel cross with nearest-depth z-buffer',
        geometry='Saved evaluated GT-aligned points; no transformation after evaluation except shared camera projection',
        sampling='Up to 1 million points per saved cloud; private RNG seed 1729; metrics use full clouds',
        mean_error={c:scores[c]['CD_norm'] for c in conditions},gain_vs_original_percent=gain,
        hypothesis_change_vs_scalar_percent=change,paper_style_reference='https://arxiv.org/html/2610.22514v1#S5.F7')
    (out/'render_provenance.json').write_text(json.dumps(provenance,indent=2))
    print(json.dumps({'figure':str(png),'pdf':str(pdf),'gain_vs_original_percent':gain,'in_frame_fraction':fractions},indent=2))

if __name__=='__main__':main()
