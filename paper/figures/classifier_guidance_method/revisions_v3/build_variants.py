"""Three editable, consistently typeset alternatives for the joint SiT figure."""
from pathlib import Path
import copy
import hashlib
import json
import random
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
import drawing as d
import build_figure as previous
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

FONT = 'Liberation Sans'
d.FONT_FAMILY = FONT
d.FONT_FILES = {False: str(ROOT/'fonts/LiberationSans-Regular.ttf'),
                True: str(ROOT/'fonts/LiberationSans-Bold.ttf')}
W, H = 1160, 720
INK = '#263746'
MUTED = '#667783'
BORDER = '#CDD8DF'
FIXED = '#EAF0F5'
FIXED_EDGE = '#B6C8D6'
TRAIN = '#F6E8D8'
TRAIN_EDGE = '#D1AD83'
GRAD = '#B86436'
LOSS = '#FCF7F2'
d.NAVY, d.BLUE, d.RED = INK, '#5E8DB5', '#C56C60'


def r(text, sub=False):
    return dict(text=text, sub=sub)


def tx(name, x, y, w, h, text, size=16, bold=False, color=INK, align='center'):
    d.text(name, x, y, w, h, text, size, color, bold, FONT, align)


def edge(name, points, color=INK, dashed=False, arrow=True, width=1.25):
    d.line(name, points, stroke=color, width=width, dash=dashed, end=arrow)


def status(name, x, y, train=False):
    if train:
        d.flame(name+'/trainable', x, y, .51)
    else:
        d.snow(name+'/frozen', x, y, 6.8)


def bars(name, x, y, w, h, layers, train=False):
    """Connected block silhouette, inspired by DMD2; not literal layer counts."""
    n = len(layers)
    gap = w/(n+1)
    bw = min(11, gap*.68)
    heights = [h*(.75 if j in (0,n-1) else 1) for j in range(n)]
    edge(name+'/network spine', [(x,y+h/2),(x+w,y+h/2)], arrow=False, color=TRAIN_EDGE if train else FIXED_EDGE, width=.85)
    for j, bh in enumerate(heights):
        d.rect(name+f'/network block {j}', x+(j+1)*gap-bw/2, y+(h-bh)/2,
               bw, bh, '#D5AD7C' if train else '#9AB3C7', r=.8)


def network_card(name, x, y, w, h, text, train=False, strong=False):
    d.rect(name+'/box', x,y,w,h,TRAIN if train else FIXED,
           TRAIN_EDGE if train else FIXED_EDGE,r=4,width=.8)
    bars(name,x+w/2-31,y+12,62,25,[1]* (4 if strong else 3),train)
    tx(name+'/label',x+3,y+h-26,w-6,22,text,15.5)
    status(name,x+w-11,y+11,train)


def scale_card(name,x,y,w,h,train=False):
    d.rect(name+'/box',x,y,w,h,TRAIN if train else FIXED,
           TRAIN_EDGE if train else FIXED_EDGE,r=4,width=.8)
    cx=x+w/2
    edge(name+'/zero axis',[(cx-29,y+32),(cx+29,y+32)],color=FIXED_EDGE,arrow=False,width=.65)
    edge(name+'/schedule',[(cx-28,y+28),(cx-15,y+28),(cx-15,y+16),(cx-2,y+16),
                          (cx-2,y+25),(cx+9,y+25),(cx+9,y+39),(cx+20,y+39),(cx+20,y+25),(cx+29,y+25)],
         arrow=False,width=1.35)
    tx(name+'/label',x+3,y+h-26,w-6,22,[r('Scale a'),r('α',True)],15.5)
    status(name,x+w-11,y+11,train)


def sampler(name,x,y,w,h,train=False):
    d.rect(name+'/container',x,y,w,h,'#FAFCFD',BORDER,r=5,width=.9)
    tx(name+'/title',x+12,y+8,w-24,24,'Full-trajectory sampler',17,True,align='left')
    cw=(w-38)/3
    cy=y+42;ch=78 if h>=195 else 69
    network_card(name+'.strong',x+12,cy,cw,ch,'Strong S',strong=True)
    network_card(name+'.weak',x+19+cw,cy,cw,ch,[r('Weak W'),r('θ',True)],train)
    scale_card(name+'.scale',x+26+2*cw,cy,cw,ch,train)
    tx(name+'/guided field',x+12,cy+ch+7,w-24,25,
       [r('S + a'),r('α',True),r('(t)(S − W'),r('θ',True),r(')')],17)
    yy=y+h-24
    for dx,content in [(w*.17,'z'),(w*.49,[r('x'),r('t',True)]),(w*.81,[r('x'),r('1',True)])]:
        tx(name+f'/state {dx}',x+dx-18,yy-11,36,22,content,15)
    edge(name+'/trajectory first',[(x+w*.17+22,yy),(x+w*.49-25,yy)],width=1.1)
    edge(name+'/trajectory second',[(x+w*.49+25,yy),(x+w*.81-25,yy)],width=1.1)
    edge(name+'/trajectory output',[(x+w*.81+22,yy),(x+w,yy)],arrow=False,width=1.1)
    return y+h-24


def noise(name,x,y,w=48,h=33):
    rng=random.Random(17)
    for j in range(6):
        for i in range(9):
            q=rng.randrange(140,224)
            d.rect(name+f'/cell {i}-{j}',x+i*w/9,y+j*h/6,w/9+.03,h/6+.03,
                   '#'+f'{q:02x}'*3,r=0)
    d.rect(name+'/frame',x,y,w,h,None,BORDER,r=1,width=.65)
    tx(name+'/label',x-7,y+h+5,w+14,21,'Noise z',14)


def module(name,x,y,w,h,text,train=False,kind=None):
    d.rect(name+'/box',x,y,w,h,TRAIN if train else FIXED,
           TRAIN_EDGE if train else FIXED_EDGE,r=4,width=.8)
    left=7
    if kind=='features':
        for j in range(3):
            d.rect(name+f'/plane {j}',x+12+4*j,y+h/2-15+4*j,17,22,
                   ['#F5F8FA','#D8E4EE','#FFFFFF'][j],FIXED_EDGE,r=1,width=.7)
        left=45
    elif kind=='critic':
        bars(name,x+11,y+h/2-12,34,24,[1,1,1],train)
        left=48
    tx(name+'/label',x+left,y+4,w-left-10,h-8,text,16)
    status(name,x+w-8,y+4,train)


def data_box(name,x,y,w,h,text,real=False):
    d.rect(name+'/box',x,y,w,h,'#FFFFFF',BORDER,r=3,width=.85)
    d.rect(name+'/accent',x,y,2.5,h,'#C56C60' if real else '#668DAC',r=0)
    tx(name+'/label',x+6,y+3,w-12,h-6,text,15)


def g_formula():
    return [r('L'),r('G',True),r(' = E[softplus(−d'),r('f',True),r(')] + λ R'),r('gap',True)]


def d_formula(first=False,second=False):
    a=[r('L'),r('D',True),r(' = E[softplus(−d'),r('r',True),r(')]')]
    b=[r(' + E[softplus(d'),r('f',True),r(')] + (γ/2) R'),r('1',True)]
    return a if first else b if second else a+b


def key(y=686,shared=False):
    status('legend.fixed',55,y)
    tx('legend.fixed/text',69,y-10,136,21,'Frozen parameters',13.5,align='left')
    status('legend.learned',238,y,True)
    tx('legend.learned/text',253,y-10,151,21,'Trainable parameters',13.5,align='left')
    edge('legend.gradient/line',[(461,y),(500,y)],GRAD,True,width=1.4)
    tx('legend.gradient/text',509,y-10,108,21,'Gradient flow',13.5,align='left')
    note='Updates alternate; see the two objectives above.' if shared else 'Each iteration: discriminator update, then guidance update.'
    tx('legend.order/text',663,y-10,460,21,note,13,color=MUTED,align='left')


def classic():
    """Keep the reference's two-panel explanation; replace lettering and glyphs."""
    previous.INK=INK;previous.FROZEN=FIXED;previous.LEARN=TRAIN
    previous.LOSS=LOSS;previous.GRAY=BORDER;previous.MUTED=MUTED;previous.GRAD=GRAD
    previous.network_glyph=bars
    scene=previous.build(True)
    for s in scene:
        if s['kind']=='text':
            s['font']=FONT
            if '.objective/formula' in s['name']:s['bold']=False
            if s['name']=='a.caption/title':s['runs']=[r('(a) Guidance update')];s['size']=19
            if s['name']=='b.caption/title':s['runs']=[r('(b) Discriminator update')];s['size']=19
            if '.real/label' in s['name']:s['runs']=[r('Real x'),r('r',True)];s['fill']=d.RED
            if '.fake/label' in s['name']:s['runs']=[r('Fake x'),r('f',True)];s['fill']=d.BLUE
        if s['kind']=='ellipse' and '.real/sample' in s['name']:s['fill']=d.RED
        if s['kind']=='ellipse' and '.fake/sample' in s['name']:s['fill']=d.BLUE
        if s['kind']=='path' and s['name'].endswith('/box') and s['fill'] in (FIXED,TRAIN):
            s['stroke']=FIXED_EDGE if s['fill']==FIXED else TRAIN_EDGE;s['width']=.7
    # One aspect ratio for the three comparison slides, with uniform scaling.
    scale=.90; ox=94;oy=7
    for s in scene:
        if s['kind']=='ellipse':
            s['cx']=s['cx']*scale+ox;s['cy']=s['cy']*scale+oy
            s['rx']*=scale;s['ry']*=scale;s['width']*=scale
        elif s['kind']=='path':
            for c in s['commands']:
                for j in range(1,len(c),2):c[j]=c[j]*scale+ox;c[j+1]=c[j+1]*scale+oy
            s['width']*=scale
        else:
            s['x']=s['x']*scale+ox;s['y']=s['y']*scale+oy
            s['w']*=scale;s['h']*=scale
            if s['kind']=='text':s['size']*=scale
    return scene


def shared():
    d.SCENE=[]
    tx('title/main',44,17,850,32,'Guided sampling with adversarial feedback',22,True,align='left')
    edge('gradient/return',[(45,457),(22,457),(22,76),(329,76),(329,98)],GRAD,True,width=1.5)
    tx('gradient/label',77,53,350,21,'Backpropagate through the full trajectory',14,color=GRAD,align='left')
    yy=sampler('shared.sampler',155,98,355,216,True)
    noise('shared.noise',49,172,48,34)
    edge('shared.noise/edge',[(103,189),(155,189)])
    tx('shared.class/label',46,263,79,22,'Class c',15)
    edge('shared.class/edge',[(120,274),(155,274)])
    module('shared.decode',544,yy-25,105,50,'VAE decode')
    data_box('shared.fake',681,yy-19,99,38,[r('Fake x'),r('f',True)])
    module('shared.features',811,yy-30,157,60,'Inception φ',kind='features')
    module('shared.critic',999,yy-30,139,60,[r('D'),r('ω',True)],True,'critic')
    for a,b in [(510,544),(649,681),(780,811),(968,999)]:edge(f'shared.forward/{a}',[(a,yy),(b,yy)])
    data_box('shared.real',705,148,115,39,[r('Real x'),r('r',True)],True)
    edge('shared.real/edge',[(762,187),(762,224),(888,224),(888,yy-30)])
    tx('shared.conditional/label',1048,199,37,22,'c',15)
    edge('shared.conditional/edge',[(1067,222),(1067,yy-30)])
    tx('shared.detach/note',661,yy+45,307,23,'Detach generated samples for the D update',13,color=MUTED)
    edge('shared.loss/bus',[(1068,yy+30),(1068,370),(299,370)],arrow=False)
    edge('shared.loss/guidance',[(299,370),(299,406)])
    edge('shared.loss/discriminator',[(869,370),(869,406)])
    tx('shared.loss/guidance logit',311,375,69,22,[r('d'),r('f',True)],14,align='left')
    tx('shared.loss/discriminator logits',880,375,86,22,[r('d'),r('r',True),r(', d'),r('f',True)],14,align='left')
    for name,x,title in [('guidance',45,'(a) Guidance update'),('discriminator',611,'(b) Discriminator update')]:
        d.rect(name+'/objective box',x,406,508,177,LOSS,BORDER,r=4,width=.8)
        tx(name+'/heading',x+19,420,470,26,title,18,True,align='left')
    status('guidance.active',69,464,True)
    tx('guidance.parameters',84,453,210,23,[r('Update W'),r('θ',True),r(' and a'),r('α',True)],15,align='left')
    status('guidance.fixed critic',325,464)
    tx('guidance.critic state',341,453,160,23,[r('Freeze D'),r('ω',True)],15,align='left')
    tx('guidance.formula',61,490,476,27,g_formula(),18)
    tx('guidance.probe',63,533,470,22,[r('R'),r('gap',True),r(': independent real-data probe; W'),r('0',True),r(' fixed')],14,color=MUTED,align='left')
    status('discriminator.active',635,464,True)
    tx('discriminator.parameters',650,453,128,23,[r('Update D'),r('ω',True)],15,align='left')
    status('discriminator.fixed sampler',804,464)
    tx('discriminator.sampler state',820,453,277,23,'Freeze sampling parameters',15,align='left')
    tx('discriminator.formula 1',629,488,472,25,d_formula(first=True),17.5)
    tx('discriminator.formula 2',629,514,472,25,d_formula(second=True),17.5)
    tx('discriminator.regularizer',629,550,472,22,[r('R'),r('1',True),r(': gradient penalty on real features')],14,color=MUTED)
    tx('notes/logits',50,617,1060,22,[r('d'),r('r',True),r(' = D'),r('ω',True),r('(φ(x'),r('r',True),r('), c),   d'),r('f',True),r(' = D'),r('ω',True),r('(φ(x'),r('f',True),r('), c).  Icons in the forward graph show which parameters are learned across training.')],13.5,color=MUTED,align='left')
    key(shared=True)
    return copy.deepcopy(d.SCENE)


def lanes():
    d.SCENE=[]
    for i,train in enumerate([False,True]):
        o=i*337;name='g' if train else 'd'
        title='(b) Guidance update' if train else '(a) Discriminator update'
        tx(name+'.heading/title',30,15+o,690,29,title,21,True,align='left')
        tx(name+'.heading/parameters',768,21+o,365,23,
           [r('Update θ and α; freeze D'),r('ω',True)] if train else [r('Update ω; freeze the sampler')],15,color=MUTED,align='left')
        yy=sampler(name+'.sampler',111,68+o,350,179,train)
        noise(name+'.noise',32,147+o,49,34)
        edge(name+'.noise/edge',[(81,164+o),(111,164+o)])
        tx(name+'.class/label',24,210+o,62,22,'Class c',13.5)
        edge(name+'.class/edge',[(84,221+o),(111,221+o)])
        module(name+'.decode',492,yy-24,112,48,'VAE decode')
        data_box(name+'.fake',635,yy-18,90,36,[r('Fake x'),r('f',True)])
        module(name+'.features',755,yy-29,154,58,'Inception φ',kind='features')
        module(name+'.critic',940,yy-29,145,58,[r('D'),r('ω',True)],not train,'critic')
        for a,b in [(461,492),(604,635),(725,755),(909,940)]:edge(name+f'.forward/{a}',[(a,yy),(b,yy)])
        tx(name+'.condition/label',989,yy-84,42,22,'c',15)
        edge(name+'.condition/edge',[(1010,yy-60),(1010,yy-29)])
        if not train:
            data_box(name+'.real',654,97+o,112,37,[r('Real x'),r('r',True)],True)
            edge(name+'.real/edge',[(710,134+o),(710,165+o),(832,165+o),(832,yy-29)])
            edge(name+'.detach/gate',[(739,yy-7),(739,yy+7)],GRAD,arrow=False,width=2)
            tx(name+'.detach/label',708,yy+18,71,21,'detach',13,color=GRAD)
        edge(name+'.loss/edge',[(1085,yy),(1119,yy),(1119,274+o)])
        d.rect(name+'.loss/box',572,274+o,560,46,LOSS,BORDER,r=4,width=.8)
        tx(name+'.loss/formula',583,283+o,538,28,g_formula() if train else d_formula(),16.5)
        if train:
            edge(name+'.gradient/return',[(572,297+o),(536,297+o),(536,265+o),(286,265+o),(286,247+o)],GRAD,True,width=1.5)
            tx(name+'.gradient/label',134,274+o,365,24,'Full-trajectory backpropagation',14,color=GRAD)
            tx(name+'.probe/note',130,302+o,425,22,[r('R'),r('gap',True),r(': independent real-data probe; W'),r('0',True),r(' fixed')],13.5,color=MUTED,align='left')
        else:
            tx(name+'.regularizer/note',129,284+o,395,25,[r('R'),r('1',True),r(': gradient penalty on real features')],14,color=MUTED,align='left')
    edge('divider/line',[(28,345),(1132,345)],color=BORDER,arrow=False,width=.8)
    key(699)
    return copy.deepcopy(d.SCENE)


def audit(scene,tag):
    overflow=[];families=set();chars={False:set(),True:set()}
    for s in scene:
        if s['kind']!='text':continue
        families.add(s['font'])
        for run in s['runs']:
            families.add(run.get('font',s['font']))
            chars[s['bold']].update(run['text'])
        width=sum(d.measure(s))
        if width>s['w']+.6:overflow.append((s['name'],round(width,1),s['w']))
    assert families=={FONT},families
    assert not overflow,overflow
    for bold,used in chars.items():
        font=TTFont(d.FONT_FILES[bold]);missing=used-set(map(chr,font.getBestCmap()))
        assert not missing,(tag,missing)
        assert font['OS/2'].fsType==0
    names=[s['name'] for s in scene]
    assert len(names)==len(set(names)),tag
    assert any('/frozen' in x for x in names) and any('/trainable' in x for x in names)
    assert not any('strong' in x and '/trainable' in x for x in names)
    return dict(variant=tag,objects=len(scene),font_family=FONT,missing_characters=[],text_overflow=[],
                frozen_and_trainable_icons=True,strong_always_frozen=True)


NOTES=(
    'Joint SiT weak-head and signed-schedule learning. All text, Greek symbols and subscripts use '
    'Liberation Sans regular/bold. Snowflake = frozen parameters; flame = parameters learned in the indicated update. '
    'Strong S, VAE decoder and Inception φ are always frozen; their input derivatives are retained in the guidance update. '
    'The conditional discriminator operates on terminal RGB Inception features and outputs logits. '
    'Train D first with generated samples detached and real-feature R1, then freeze D and train θ and α '
    'through the entire sampler with the non-saturating loss and an independent real-data gap probe. '
    'W0 is the frozen initial weak head. Rgap is the expectation of the squared log minibatch gap-energy ratio. '
    'Small networks are schematic and do not specify exact layer counts or a separate weak backbone. '
    'The shared-process slide shows parameters learned across alternating updates; its objective panels give the per-update frozen state. '
    'Visual references: AdvFD Fig.3; DMD2 Fig.3 (2405.14867); ADD Fig.2 (2311.17042); APT Fig.1 (2501.08316). '
    'Only visual organization is borrowed; no teacher distillation branch, one-step sampler, or diffusion discriminator is implied.'
)


def main():
    d.W,d.H=W,H
    variants=[('A_classic','A · Two-panel explanation',classic()),
              ('B_shared','B · Shared forward process',shared()),
              ('C_lanes','C · Alternating training flows',lanes())]
    slides=[];checks=[]
    for tag,title,scene in variants:
        checks.append(audit(scene,tag))
        svg=d.svg(scene,title)
        (ROOT/f'{tag}.svg').write_text(svg)
        d.cairosvg.svg2png(bytestring=svg.encode(),write_to=str(ROOT/f'{tag}.png'),output_width=2320,output_height=1440)
        d.cairosvg.svg2pdf(bytestring=svg.encode(),write_to=str(ROOT/f'{tag}.pdf'))
        (ROOT/f'{tag}.scene.json').write_text(json.dumps(scene,ensure_ascii=False,indent=2))
        slides.append((scene,title,NOTES))
    d.export_deck(slides,ROOT/'concept_variants_v3.pptx')
    # Keep newly added text in the same family when the editable deck is opened.
    from zipfile import ZipFile, ZIP_DEFLATED
    from lxml import etree
    deck=ROOT/'concept_variants_v3.pptx'
    with ZipFile(deck) as z:parts={n:z.read(n) for n in z.namelist()}
    ns={'a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
    for name,data in list(parts.items()):
        if name.startswith('ppt/theme/') and name.endswith('.xml'):
            root=etree.fromstring(data)
            for node in root.xpath('//a:fontScheme//a:latin | //a:fontScheme//a:ea | //a:fontScheme//a:cs',namespaces=ns):
                node.set('typeface',FONT)
            parts[name]=etree.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True)
    with ZipFile(deck,'w',ZIP_DEFLATED) as z:
        for name,data in parts.items():z.writestr(name,data)
    # A comparison sheet uses previews only; the individual sources stay native/editable.
    sheet=Image.new('RGB',(1240,2428),'#EEF1F4');draw=ImageDraw.Draw(sheet)
    title_font=ImageFont.truetype(d.FONT_FILES[True],27)
    subtitle_font=ImageFont.truetype(d.FONT_FILES[False],18)
    titles=[('A  Two-panel explanation','Closest to the previous structure; distributions retained.'),
            ('B  Shared forward process','One sampler; two explicit adversarial objectives.'),
            ('C  Alternating training flows','Separate forward paths, shown in the actual update order.')]
    for i,((tag,_,_), (title,subtitle)) in enumerate(zip(variants,titles)):
        y=16+i*805
        draw.text((39,y),title,font=title_font,fill=INK)
        draw.text((39,y+34),subtitle,font=subtitle_font,fill=MUTED)
        im=Image.open(ROOT/f'{tag}.png').convert('RGB');im.thumbnail((1160,720),Image.Resampling.LANCZOS)
        sheet.paste(im,(40,y+68))
    sheet.save(ROOT/'comparison.png')
    report={'scene_checks':checks,'artifact_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
            for p in ROOT.iterdir() if p.suffix in {'.svg','.pdf','.png','.pptx'}}}
    sources=['classifier_guidance/sit_joint.py','classifier_guidance/schedules.py',
             'experiments/adversarial_weak_training_20260915/binary_critic.py']
    report['source_sha256']={p:(hashlib.sha256((previous.REPO/p).read_bytes()).hexdigest()
                              if (previous.REPO/p).is_file() else None) for p in sources}
    (ROOT/'review_checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
