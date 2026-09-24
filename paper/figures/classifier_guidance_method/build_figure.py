"""Editable method figures derived from the current classifier_guidance code.

The first figure shows SiT joint weak-head/schedule learning; the second shows
the current JiT schedule-only variant. Both preserve the AdvFD visual vocabulary.
"""
from pathlib import Path
import base64
import copy
import hashlib
import io
import json
import random
import shutil
import zipfile

import drawing as d
from PIL import Image

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[2]
d.W,d.H=1080,780
TOP_EXPANSION=40
FROZEN='#D0DDEA'
LEARN='#E1B882'
LOSS='#FAEAE2'
GRAY='#E5E7E9'
MUTED='#697A8B'
GRAD='#C45B25'
INK=d.NAVY


def rr(s,sub=False,sup=False,math=False):
    return dict(text=s,sub=sub,sup=sup)


def label(name,x,y,w,h,content,size=16,fill=INK,bold=False,math=False,align='center'):
    d.text(name,x,y,w,h,content,size,fill,bold,d.FONT_FAMILY,align)


def network_glyph(name,x,y,w,h,layers,train=False):
    """Small conceptual neural-network icon; node counts are not model widths."""
    nodes=[]
    for k,n in enumerate(layers):
        cx=x+w*k/(len(layers)-1)
        ys=[y+h/2] if n==1 else [y+h*j/(n-1) for j in range(n)]
        nodes.append([(cx,cy) for cy in ys])
    for k in range(len(nodes)-1):
        for i,a in enumerate(nodes[k]):
            for j,b in enumerate(nodes[k+1]):
                d.line(name+f'/connection {k}-{i}-{j}',[a,b],stroke='#AA875F' if train else '#91A6BB',width=.72)
    for k,layer in enumerate(nodes):
        for j,(cx,cy) in enumerate(layer):
            d.ellipse(name+f'/layer {k} node {j}',cx,cy,3.05,3.05,'#FFFEFB',stroke=INK,width=.85)


def network_module(name,x,y,w,content,layers,train=False):
    d.rect(name+'/box',x,y,w,77,LEARN if train else FROZEN,r=8)
    spread=72 if len(layers)==4 else 53
    network_glyph(name,x+(w-spread)/2,y+14,spread,31,layers,train)
    label(name+'/label',x+3,y+55,w-6,19,content,13.6)
    if train:d.flame(name+'/trainable',x+3,y+1,.58)
    else:d.snow(name+'/frozen',x+3,y+1,8)


def scale_module(name,x,y,w,train=False):
    d.rect(name+'/box',x,y,w,77,LEARN if train else FROZEN,r=8)
    d.line(name+'/time axis',[(x+17,y+37),(x+w-12,y+37)],stroke='#91A0AE',width=.65)
    d.line(name+'/value axis',[(x+17,y+10),(x+17,y+49)],stroke='#91A0AE',width=.65)
    pts=[(x+19,y+31),(x+31,y+31),(x+31,y+18),(x+43,y+18),
         (x+43,y+27),(x+55,y+27),(x+55,y+44),(x+67,y+44),(x+67,y+32),(x+80,y+32)]
    d.line(name+'/signed schedule',pts,stroke=INK,width=1.65)
    label(name+'/label',x+3,y+55,w-6,19,[rr('Scale a'),rr('α',sub=True)],13.6)
    if train:d.flame(name+'/trainable',x+3,y+1,.58)
    else:d.snow(name+'/frozen',x+3,y+1,8)


def translate(items,dy):
    for s in items:
        if s['kind']=='ellipse':s['cy']+=dy
        elif s['kind'] in ['text','image']:s['y']+=dy
        elif s['kind']=='path':
            for c in s['commands']:
                for k in range(2,len(c),2):c[k]+=dy


def module(name,x,y,w,h,content,train=False,size=15.4,icon=True):
    d.rect(name+'/box',x,y,w,h,LEARN if train else FROZEN,r=8)
    label(name+'/label',x+4,y+2,w-8,h-4,content,size)
    if icon:
        if train:d.flame(name+'/trainable',x+3,y+1,.58)
        else:d.snow(name+'/frozen',x+3,y+1,8)


def batch(name,x,y,real=True,reference=False):
    d.rect(name+'/box',x,y,134,54,'#FFFFFF',GRAY,r=8,width=1.5)
    for j in range(5):
        col='#E1706B' if real else (LEARN if j==0 else '#B7C2D3')
        d.ellipse(name+f'/sample {j}',x+19+j*24,y+17,6,6,col)
    txt=[rr('Real Data ' if real else 'Fake Batch '),rr('x',math=True),rr('r' if real else 'f',sub=True,math=True)]
    label(name+'/label',x+2,y+30,130,21,txt,14.3,'#E1706B' if real else '#CB9558')
    if reference:label(name+'/reference',x-2,y+61,138,18,'Reference only',12.7,MUTED)


def sampler(o,step,joint):
    name=f'{step}.sampler'
    d.rect(name+'/container',o+165,18,341,165+TOP_EXPANSION,'#FBFCFD','#E2E7EB',r=10,width=1.1)
    label(name+'/title',o+173,22,325,24,'Full-trajectory sampler',17.7)
    active=step=='a'
    network_module(f'{step}.strong',o+179,57,98,[rr('Strong S')],[3,4,4,3])
    network_module(f'{step}.weak',o+289,57,96,[rr('Weak W'),rr('θ',sub=True)],[2,3,2],train=active and joint)
    scale_module(f'{step}.scale',o+397,57,96,train=active)
    shift_start=len(d.SCENE)
    # All three predictions/coefficients enter the guided prediction.
    for cx in (228,337,445):d.line(f'{step}.merge/{cx}',[(o+cx,94),(o+cx,107)],width=1.15)
    d.line(f'{step}.merge/bus',[(o+228,107),(o+445,107)],width=1.15)
    d.line(f'{step}.merge/output',[(o+337,107),(o+337,115)],width=1.15,end=True)
    guided=[rr('S + a'),rr('α',sub=True),rr('(t)(S − W'),rr('θ',sub=True),rr(')')]
    label(name+'/guided prediction',o+180,115,312,26,guided,16.6,math=True)
    label(name+'/noise state',o+190,147,32,24,'z',15.3,math=True)
    label(name+'/middle state',o+314,147,42,24,[rr('x'),rr('t',sub=True)],15.3,math=True)
    label(name+'/final state',o+424,147,45,24,[rr('x'),rr('1',sub=True)],15.3,math=True)
    d.line(f'{step}.trajectory/early',[(o+225,159),(o+308,159)],end=True,width=1.3)
    d.line(f'{step}.trajectory/late',[(o+361,159),(o+419,159)],end=True,width=1.3)
    translate(d.SCENE[shift_start:],TOP_EXPANSION)
    # Deterministic noise texture: a replaceable image, not a flattened diagram.
    rng=random.Random(120)
    im=Image.new('L',(180,100));im.putdata([rng.randrange(45,225) for _ in range(18000)])
    buf=io.BytesIO();im.save(buf,format='PNG')
    d.add('image',f'{step}.noise/texture',x=o+66,y=56,w=64,h=38,data=base64.b64encode(buf.getvalue()).decode())
    label(f'{step}.noise/label',o+57,98,82,24,'Noise z',15.2)
    d.line(f'{step}.input/noise',[(o+131,76),(o+165,76)],end=True)
    shift_start=len(d.SCENE)
    label(f'{step}.input/class',o+63,144,73,25,'Class c',15.2)
    d.line(f'{step}.input/class edge',[(o+137,159),(o+165,159)],end=True)
    d.line(f'{step}.output/endpoint',[(o+447,172),(o+447,203)],end=True)
    module(f'{step}.decode',o+387,203,119,31,'VAE decode' if joint else 'RGB map',size=14.5)
    d.line(f'{step}.output/rgb',[(o+447,234),(o+447,250)],end=True)
    translate(d.SCENE[shift_start:],TOP_EXPANSION)


def feedback(o,step,joint):
    batch(f'{step}.real',o+89,250,True,reference=step=='a' and not joint)
    batch(f'{step}.fake',o+373,250,False)
    # The discriminator gets only terminal RGB features, never a noisy state.
    if step=='b':
        d.line(f'{step}.inputs/real',[(o+156,304),(o+156,321),(o+389,321)],width=1.4)
    d.line(f'{step}.inputs/fake',[(o+440,304),(o+440,321),(o+389,321),(o+389,336)],end=True,width=1.4)
    if step=='b':
        d.line(f'{step}.detach/gate',[(o+434,313),(o+446,313)],stroke=GRAD,width=2.3)
        label(f'{step}.detach/label',o+449,301,63,22,'stop grad.',11.5,GRAD)
    module(f'{step}.features',o+276,336,226,45,'',size=17.3)
    # Three overlapping feature planes visually identify the frozen encoder.
    for k in range(3):
        d.rect(f'{step}.features/feature plane {k}',o+288+8*k,343+4*k,23,23,
               ['#E8F0F7','#DCE7F1','#F8FBFE'][k],stroke='#7894B0',r=2,width=.85)
    label(f'{step}.features/name',o+343,340,151,37,[rr('Inception φ')],16.5)
    d.line(f'{step}.features/output',[(o+389,381),(o+389,404)],end=True)
    module(f'{step}.classifier',o+276,404,226,43,'',train=step=='b',size=17.8)
    network_glyph(f'{step}.classifier',o+290,414,43,22,[2,3,1],train=step=='b')
    label(f'{step}.classifier/name',o+343,408,153,35,[rr('Classifier D'),rr('ω',sub=True)],16.0)
    label(f'{step}.condition/class',o+239,403,27,21,'c',13.8,math=True)
    d.line(f'{step}.condition/edge',[(o+239,426),(o+276,426)],end=True,width=1.2)
    # A real-data probe regularizes the current weak predictor in the joint run.
    if step=='a' and joint:
        d.line(f'{step}.probe/edge',[(o+156,304),(o+156,377)],stroke=MUTED,width=1.25,end=True)
        label(f'{step}.probe/description',o+165,328,101,29,'Fresh probe',13.7,MUTED)
        d.rect(f'{step}.anchor/box',o+86,377,141,60,LOSS,r=7)
        label(f'{step}.anchor/title',o+90,381,133,24,'Scale anchor',15.6)
        label(f'{step}.anchor/formula',o+90,407,133,23,[rr('R'),rr('gap',sub=True),rr('(θ; W'),rr('0',sub=True),rr(')')],14.6,math=True)
        d.line(f'{step}.loss/anchor edge',[(o+156,437),(o+156,462),(o+289,462)],width=1.4)
    d.line(f'{step}.loss/classifier edge',[(o+389,447),(o+389,462),(o+289,462),(o+289,477)],width=1.4,end=True)
    d.rect(f'{step}.objective/box',o+77,477,428,54,LOSS,r=7)
    if step=='a':
        runs=[rr('min'),rr('θ, α' if joint else 'α',sub=True),rr('  E[softplus(−d'),rr('f',sub=True),rr(')]')]
        if joint:runs += [rr(' + λ R'),rr('gap',sub=True)]
        label(f'{step}.objective/formula',o+85,487,412,34,runs,16.2,GRAD,True,True)
        note='Learn weak head and signed schedule' if joint else 'Learn signed schedule; keep weak head fixed'
        label(f'{step}.objective/note',o+64,534,450,21,note,13.6,MUTED)
        d.line(f'{step}.gradient/return',[(o+77,502),(o+28,502),(o+28,196),(o+337,196),(o+337,183)],stroke=GRAD,width=1.8,dash=True,end=True)
        label(f'{step}.gradient/label',o-92,343,210,21,'Full-trajectory gradient',13.8,GRAD)
        d.SCENE[-1]['rotation']=-90
    else:
        runs=[rr('min'),rr('ω',sub=True),rr('  E[softplus(−d'),rr('r',sub=True),rr(')]')]
        label(f'{step}.objective/formula',o+83,477,416,27,runs,16.2,GRAD,True,True)
        runs=[rr('+ E[softplus(d'),rr('f',sub=True),rr(')] + (γ/2) R'),rr('1',sub=True)]
        label(f'{step}.objective/formula continuation',o+83,503,416,27,runs,16.2,GRAD,True,True)
        label(f'{step}.objective/note',o+74,534,440,21,[rr('R',math=True),rr('1',sub=True,math=True),rr(': gradient penalty on real features')],13.6,MUTED)


def distribution(o,step,joint):
    start=len(d.SCENE)
    if step=='a':
        d.cloud('a.before-real',o+119,608,24,17,d.RED,100,n=48)
        d.cloud('a.before-fake',o+177,578,24,19,d.BLUE,200,n=48)
        # Real cloud only translates with the panel, without changing shape.
        d.cloud('a.after-real',o+408,608,24,17,d.RED,100,n=48)
        d.cloud('a.after-fake',o+415,603,24,19,d.BLUE,200,n=48)
        label('a.change/title',o+230,564,112,27,[rr('Update '),rr('θ, α' if joint else 'α',math=True)],16.5)
        d.line('a.change/arrow',[(o+238,601),(o+332,601)],end=True,width=1.7)
        label('a.change/action',o+216,606,140,27,'Match endpoints',15.2)
        label('a.caption/title',o+45,663,490,31,'(a) Guidance-step Training',20.5,'#000000',True)
    else:
        # Paired identical clouds: D changes a decision function, not phi or data.
        for suffix,shift in [('before',0),('after',292)]:
            d.cloud(f'b.{suffix}-real',o+115+shift,584,23,17,d.RED,300,n=48)
            d.cloud(f'b.{suffix}-fake',o+161+shift,612,23,17,d.BLUE,400,n=48)
        d.line('b.before-boundary/line',[(o+87,568),(o+186,565)],stroke=INK,width=1.7,dash=True)
        d.line('b.after-boundary/line',[(o+378,633),(o+475,564)],stroke=INK,width=1.7,dash=True)
        label('b.change/title',o+232,564,112,27,[rr('Update '),rr('ω',math=True)],16.5)
        d.line('b.change/arrow',[(o+238,601),(o+333,601)],end=True,width=1.7)
        label('b.change/action',o+226,606,125,27,'Fit boundary',15.2)
        label('b.caption/title',o+26,663,510,31,'(b) Discriminator-step Training',20.5,'#000000',True)
    # Leave enough room below the two-line discriminator objective.
    for s in d.SCENE[start:]:
        if '.caption/' in s['name']:continue
        if s['kind']=='ellipse':s['cy']+=8
        elif s['kind']=='text':s['y']+=8
        elif s['kind']=='path':
            for c in s['commands']:
                for k in range(2,len(c),2):c[k]+=8


def legend():
    y=717
    d.ellipse('legend.real/dot',190,y,4.5,4.5,d.RED)
    label('legend.real/text',202,y-11,74,22,'Real',13.5,align='left')
    d.ellipse('legend.fake/dot',279,y,4.5,4.5,d.BLUE)
    label('legend.fake/text',291,y-11,64,22,'Fake',13.5,align='left')
    d.snow('legend.frozen/icon',374,y,7.5)
    label('legend.frozen/text',388,y-11,104,22,'Frozen / fixed',13.5,align='left')
    d.flame('legend.train/icon',526,y,.58)
    label('legend.train/text',541,y-11,90,22,'Trainable',13.5,align='left')
    d.line('legend.grad/line',[(645,y),(676,y)],stroke=GRAD,dash=True,width=1.6,end=True)
    label('legend.grad/text',685,y-11,89,22,'Gradient',13.5,align='left')
    label('legend.schematic/text',860,y-11,170,22,'Distributions: schematic',11.9,MUTED)


def build(joint):
    d.SCENE=[]
    for offset,step in [(0,'a'),(540,'b')]:
        sampler(offset,step,joint)
        start=len(d.SCENE)
        feedback(offset,step,joint)
        distribution(offset,step,joint)
        translate(d.SCENE[start:],TOP_EXPANSION)
    start=len(d.SCENE)
    legend()
    translate(d.SCENE[start:],TOP_EXPANSION)
    return copy.deepcopy(d.SCENE)


def audit(scene,joint):
    result={'variant':'joint' if joint else 'schedule_only','objects':len(scene)}
    overflow=[]
    for s in scene:
        if s['kind']=='text':
            w=sum(d.measure(s))
            if w>s['w']+.5:overflow.append(dict(name=s['name'],text_width=round(w,2),box_width=s['w']))
    result['text_overflow']=overflow
    assert not overflow,overflow
    fonts={r.get('font',s['font']) for s in scene if s['kind']=='text' for r in s['runs']}
    assert fonts=={d.FONT_FAMILY},fonts
    result['font_family']=d.FONT_FAMILY
    result['network_icons']='Strong: 4 schematic layers; Weak: 3; classifier: 3 (not literal architecture counts)'
    def subset(prefix):return [s for s in scene if s['name'].startswith(prefix+'/point')]
    for col in ('real','fake'):
        a,b=subset('b.before-'+col),subset('b.after-'+col)
        assert len(a)==len(b)==48
        assert all(abs(y['cx']-x['cx']-292)<1e-8 and abs(y['cy']-x['cy'])<1e-8 and x['rx']==y['rx'] for x,y in zip(a,b))
    a,b=subset('a.before-real'),subset('a.after-real')
    assert all(abs(y['cx']-x['cx']-289)<1e-8 and abs(y['cy']-x['cy'])<1e-8 for x,y in zip(a,b))
    def fill(name):return next(s['fill'] for s in scene if s['name']==name)
    for step in ('a','b'):
        assert fill(f'{step}.strong/box')==FROZEN
        assert fill(f'{step}.features/box')==FROZEN
        assert fill(f'{step}.decode/box')==FROZEN
    assert fill('a.classifier/box')==FROZEN and fill('b.classifier/box')==LEARN
    assert fill('a.weak/box')==(LEARN if joint else FROZEN)
    assert fill('a.scale/box')==LEARN
    assert fill('b.weak/box')==FROZEN and fill('b.scale/box')==FROZEN
    assert any(s['name']=='b.detach/gate' for s in scene)
    assert not any(s['name'].startswith('a.detach') for s in scene)
    result.update(trainable_states='passed',fixed_real_reference='passed',fixed_d_step_clouds='passed',gradient_gate='passed')
    return result


def main():
    notes=(
        'Visual style adapted from AdvFD Figure 3, arXiv:2608.11205v1. '
        'All pipeline content is rewritten for the local classifier_guidance implementation. '
        'Strong-model weights, feature encoder and image map are always fixed. '
        'Frozen weights retain input derivatives during the guidance update. '
        'D uses class-conditional real/fake logits on terminal RGB Inception features. '
        'softplus(s)=log(1+exp(s)); d_r=D_omega(phi(x_r),c); d_f=D_omega(phi(x_fake),c). '
        'R1=E ||grad_h D_omega(h,c)||^2 for h=phi(x_r). '
        'The actual iteration updates D first, then guidance using the updated D; panel order is expository. '
        'Scatter diagrams are conceptual; the D-step keeps both feature clouds identical and changes only its decision boundary. '
        'Arrows, modules, scatter points, icons and text are editable; only the two noise textures are raster images. '
        'All text, Greek symbols and formulas use Comic Sans MS, the lettering family of the reference. '
        'Network nodes and schedule glyphs are conceptual icons, not measured widths, exact layer counts, or learned coefficient curves. '
    )
    slides=[];audits=[]
    for joint,stem in [(True,'method_joint'),(False,'method_schedule')]:
        scene=build(joint)
        audits.append(audit(scene,joint))
        title='Joint weak-head and schedule learning' if joint else 'Schedule-only learning'
        meta=notes+('SiT joint variant: S and W are velocity predictions, VAE decoder is fixed; optimize theta and alpha with lambda=0.1 gap-energy anchor on independent real-posterior interpolants. W0 is the initial frozen weak head.' if joint else 'JiT schedule variant: S and W are clean predictions; convert the guided prediction to velocity before the fixed Heun/Euler rollout. Only alpha is optimized; W is fixed and the gap anchor is absent. RGB map is fixed rescaling/clamping.')
        xml=d.svg(scene,title)
        (ROOT/f'{stem}.svg').write_text(xml)
        d.cairosvg.svg2png(bytestring=xml.encode(),write_to=str(ROOT/f'{stem}.png'),output_width=2160,output_height=round(2160*d.H/d.W))
        d.cairosvg.svg2pdf(bytestring=xml.encode(),write_to=str(ROOT/f'{stem}.pdf'))
        (ROOT/f'{stem}.scene.json').write_text(json.dumps(scene,ensure_ascii=False,indent=2))
        for suffix in ['svg','pdf','png']:
            shutil.copyfile(ROOT/f'{stem}.{suffix}',ROOT/f'{stem}_v2.{suffix}')
        slides.append((scene,title,meta))
    d.export_deck(slides,ROOT/'classifier_guidance_method.pptx')
    shutil.copyfile(ROOT/'classifier_guidance_method.pptx',ROOT/'classifier_guidance_method_v2.pptx')
    sources=['classifier_guidance/README.md','classifier_guidance/sit_joint.py','classifier_guidance/jit_schedule.py','classifier_guidance/schedules.py','classifier_guidance/training.py','classifier_guidance/training_accumulation.py','experiments/adversarial_weak_training_20260915/binary_critic.py']
    manifest={p:(hashlib.sha256((REPO/p).read_bytes()).hexdigest() if (REPO/p).is_file() else None) for p in sources}
    (ROOT/'review_checks.json').write_text(json.dumps(dict(scene_checks=audits,source_sha256=manifest),indent=2))
    print(json.dumps(audits,indent=2))


if __name__=='__main__':main()
