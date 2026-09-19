"""Create the audit bundle and spreadsheet underlying the research report.

This consumes completed experiments only. It never starts or alters sampling.
"""
from pathlib import Path
import csv
import json
import re
import shutil
import numpy as np
from openpyxl import Workbook, load_workbook
from openpyxl.cell import WriteOnlyCell
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter
from experiments.guidance_pasted_20260912 import common as c

ROOT=c.EXPS/'weak_reference_20260914'
OUT=c.WORK/'docs/data/weak_reference_20260914'
READINGS=c.WORK/'readings/weak_reference_20260914'
PHASES=[('sit_screen_1k','sit'),('angular_screen_1k','sit'),
        ('moment_screen_1k','sit'),('strong_confirm_1k','sit'),
        ('strong_sg_1k','sit'),('strong_band_1k','sit'),('strong_band_confirm_1k','sit'),
        ('cross_screen_1k','jit'),('cross_screen_1k','raev2')]


def collect_images():
    rows=[];audits=[]
    for phase,model in PHASES:
        path=OUT/phase/model/'results.json';d=c.read(path)
        assert d['complete'] and all(a['passed'] for a in d['audits']),str(path)
        request=c.read(OUT/phase/model/'request.json')
        assert d['request_sha256']==c.sha(OUT/phase/model/'request.json')
        for r in d['rows']:
            rows.append(dict(phase=phase,seed=request['seed'],protocol='ADM ImageNet100' if model=='sit'
                else 'nanogen-evals ImageNet1K',**r))
        audits.extend(dict(phase=phase,model=model,**a) for a in d['audits'])
    assert len(rows)==34 and sum(r['samples'] for r in rows)==34000
    controllers=[]
    for phase in dict.fromkeys(p for p,m in PHASES):
        complete=c.read(ROOT/phase/'controller_complete.json')
        assert complete['complete'] and all(code==0 for code in complete['exit_codes']),phase
        controllers.append(dict(phase=phase,**complete))
    c.atomic(OUT/'controller_completion.json',controllers)
    # The frozen cross-model runner writes its terminal marker separately from
    # the progress snapshot; reconcile that snapshot after confirmed clean exits.
    status=c.read(ROOT/'cross_screen_1k/status.json')
    status.update(complete=True,completed=6,remaining_queue=0,verified_from_terminal_marker=True)
    c.atomic(ROOT/'cross_screen_1k/status.json',status)
    # Follow-ups reuse the exact confirmation noise/labels, not merely a seed label.
    paired=['strong_confirm_1k','strong_sg_1k','strong_band_1k']
    with np.load(ROOT/paired[0]/'inputs.npz') as f:noise=f['noise'];labels=f['labels']
    for phase in paired[1:]:
        with np.load(ROOT/phase/'inputs.npz') as f:
            assert np.array_equal(noise,f['noise']) and np.array_equal(labels,f['labels'])
    base=next(r for r in rows if r['phase']=='strong_confirm_1k' and r['arm']=='strong_e64')
    for row in rows:
        if row['phase'] in paired[1:]:
            row.update(baseline='strong_confirm_1k/strong_e64',delta_fid=row['fid']-base['fid'],
                       relative_seconds=row['seconds']/base['seconds'])
    band_base=next(r for r in rows if r['phase']=='strong_band_confirm_1k' and r['arm']=='strong_e320')
    for row in rows:
        if row['phase']=='strong_band_confirm_1k':
            row.update(baseline='strong_e320',delta_fid=row['fid']-band_base['fid'],
                       relative_seconds=row['seconds']/band_base['seconds'])
    c.atomic(OUT/'all_image_results.json',dict(rows=rows,audits=audits,
        total_generated_images=34000,independent_noise_seeds_sit=3,complete=True,
        independent_fid_extractor=False,followup_inputs_bitwise_identical=True))
    return rows,audits


def calibration_audit():
    rows=[];bins=[];all_checks=[]
    data=c.EXPS.parent/'imagenet_sit_flow/imagenet100_cmc_sdvae'
    for name in ['calibration','calibration_strong']:
        root=ROOT/name;out=OUT/name;out.mkdir(exist_ok=True)
        request=c.read(root/'request.json');result=c.read(root/'results.json');ids=c.read(root/'identities.json')
        assert result['request_sha256']==c.sha(root/'request.json')
        for filename,digest in request['hashes'].items():assert c.sha(Path(filename))==digest,filename
        for split,file_split in [('train','train'),('val','validation')]:
            selected=np.asarray(ids[split]['ids']);assert len(np.unique(selected))==1000
            mm=np.load(data/f'{file_split}_moments.npy',mmap_mode='r')
            labels=np.load(data/f'{file_split}_labels.npy')[selected].astype(np.int64)
            assert c.array_sha(mm[selected].copy())==ids[split]['moments_sha256']
            assert c.array_sha(labels)==ids[split]['labels_sha256']
            assert np.array_equal(np.bincount(labels,minlength=100),np.full(100,10))
        with np.load(root/'per_image_stats.npz') as d:
            fit=np.clip(-d['train'][:,:,1].mean(-1)/d['train'][:,:,2].mean(-1).clip(1e-20),0,3)
            assert np.array_equal(fit,d['weights'])
            for j,row in enumerate(result['rows']):
                w=fit[:,j,None];val=d['val'][:,j]
                per_image=(2*w*val[:,1]+w*w*val[:,2]).mean(0)
                np.testing.assert_allclose(fit[:,j],row['weights'],rtol=0,atol=0)
                assert abs(per_image.mean()-row['delta'])<1e-15
                assert abs(per_image.std(ddof=1)/np.sqrt(1000)-row['cluster_image_se'])<1e-15
                rows.append(dict(baseline='CFG' if name=='calibration' else 'conditional only',**row))
                for i,t in enumerate(request['time_bin_centers']):
                    bins.append(dict(baseline=name,kernel=row['kernel'],time=t,weight=float(fit[i,j]),
                        train_alignment=float(d['train'][i,j,1].mean()),
                        validation_alignment=float(d['val'][i,j,1].mean()),
                        train_energy=float(d['train'][i,j,2].mean()),
                        validation_energy=float(d['val'][i,j,2].mean())))
        gate=dict(best_kernel_among_nonbaseline_candidates=result['selected_kernel'],
            baseline_included_in_gate=True,any_validation_improvement=any(r['delta']<0 for r in result['rows']),
            deployed_DSM_fitted_image_arm=False,
            note='All validation loss differences are positive and within roughly one image-cluster standard error; retain baseline. Fixed-weight image arms are separate experiments.')
        assert not gate['any_validation_improvement']
        c.atomic(out/'acceptance_gate.json',gate)
        for filename in ['request.json','identities.json','results.json','per_image_stats.npz','configs.json']:
            shutil.copy2(root/filename,out/filename)
        all_checks.append(dict(experiment=name,passed=True,selected_moments_and_labels_verified=True,
            weights_and_heldout_deltas_recomputed=True,request_sha256=c.sha(root/'request.json')))
    c.atomic(OUT/'calibration_audit.json',all_checks)
    shutil.copy2(ROOT/'moment_stats.json',OUT/'moment_stats.json')
    for name in ['jit','raev2']:shutil.copy2(ROOT/'checks'/f'{name}.json',OUT/f'{name}_check.json')
    return rows,bins,all_checks


def source_catalog():
    rows=[]
    for path in sorted(READINGS.iterdir()):
        if path.suffix not in ('.html','.pdf'):continue
        arxiv=path.stem
        rows.append(dict(id=arxiv,url=f'https://arxiv.org/{"pdf" if path.suffix==".pdf" else "html"}/{arxiv}',
            local_path=str(path),format=path.suffix[1:],sha256=c.sha(path),accessed='2026-09-14'))
    template=Path(__file__).with_name('research_template.md').read_text()
    known={r['url'] for r in rows}
    for url in dict.fromkeys(re.findall(r'https://[^\s)]+',template)):
        if url not in known:rows.append(dict(url=url,accessed='2026-09-14',format='web citation'))
    c.atomic(OUT/'sources.json',rows)
    for filename in ['sg_pdf_sources.json','sg_v3_v4_token_changes.json','sg_v3_v4.diff',
                     'openreview_search.json','openreview_exact_title.json']:
        shutil.copy2(READINGS/filename,OUT/filename)
    return rows


def cell_value(v):
    if isinstance(v,(dict,list,tuple)):return json.dumps(v,ensure_ascii=False)
    if isinstance(v,np.generic):return v.item()
    return v


def write_sheet(wb,name,headers,values,width=22):
    ws=wb.create_sheet(name);ws.freeze_panes='A2'
    for i,h in enumerate(headers,1):ws.column_dimensions[get_column_letter(i)].width=max(width,min(48,len(h)+2))
    cells=[]
    for h in headers:
        cell=WriteOnlyCell(ws,h);cell.font=Font(bold=True,color='FFFFFF');cell.fill=PatternFill('solid',fgColor='333333');cells.append(cell)
    ws.append(cells);count=1
    for row in values:ws.append([cell_value(v) for v in row]);count+=1
    ws.auto_filter.ref=f'A1:{get_column_letter(len(headers))}{count}'
    return count


def dict_sheet(wb,name,rows):
    headers=list(dict.fromkeys(k for row in rows for k in row))
    return write_sheet(wb,name,headers,([r.get(k) for k in headers] for r in rows))


def csv_sheet(wb,name,path):
    def number(x):
        try:return float(x) if '.' in x or 'e' in x.lower() else int(x)
        except ValueError:return x
    with path.open() as f:
        reader=csv.reader(f);header=next(reader)
        return write_sheet(wb,name,header,([number(x) for x in row] for row in reader))


def main():
    rows,audits=collect_images();cal,bins,cal_checks=calibration_audit();sources=source_catalog()
    wb=Workbook(write_only=True);counts={}
    notes=[('ImageMetrics','34 arms, 1000 images each. FID protocols differ across models; lower is better.'),
        ('seconds','Sum of measured batch sampling plus decoding wall time; excludes model loading, disk output, and metric evaluation.'),
        ('one_dimensional_curves','All grid values underlying potentials.png, plus unplotted diagnostic cases. Instantaneous normalized potentials, not diffusion endpoints.'),
        ('two_dimensional_fields','All 241 x 241 grid values underlying curved.png. x increases across image columns; y increases across rows.'),
        ('ODE_endpoints','Actual 400-step toy flow endpoints. Known true distribution: 0.5*N(-2,0.25)+0.5*N(2,0.25).'),
        ('DSM','Training and held-out real noised data; all fitted candidate kernels fail to improve validation. Per-image statistics archived in adjacent NPZ files.'),
        ('audits','Source/input/batch hashes and FP64 FID recomputation from existing features. This is not an independent Inception extractor.'),
        ('raw_root',str(ROOT)),('report','docs/WEAK_REFERENCE_GUIDANCE_RESEARCH_20260914_ZH.md')]
    counts['Readme']=write_sheet(wb,'Readme',['item','description'],notes,width=38)
    for name,values in [('ImageMetrics',rows),('ImageAudits',audits),('DSM_calibration',cal),
                        ('DSM_time_bins',bins),('Sources',sources)]:counts[name]=dict_sheet(wb,name,values)
    for subfolder,label in [('bootstrap','Log'),('bootstrap_band','Band')]:
        boot=c.read(OUT/subfolder/'results.json')
        counts[label+'PairedBootstrap']=dict_sheet(wb,label+'PairedBootstrap',boot['rows'])
        with np.load(OUT/subfolder/'replicates.npz') as d:
            counts[label+'BootstrapReplicates']=write_sheet(wb,label+'BootstrapReplicates',d['arms'].tolist(),d['fid'])
    for name,file in [('ToyRisk','snapshot_risk.csv'),('ToyODE','ode.csv'),('ToyDSM','calibration.csv'),
                      ('RingDiagnostics','curved.csv'),('AttachmentCheck','attachment_point_check.csv')]:
        counts[name]=csv_sheet(wb,name,OUT/'toy'/file)
    for sheet,filename in [('one_dimensional_curves','snapshot_curves.npz'),('ODE_endpoints','ode_endpoints.npz')]:
        with np.load(OUT/'toy'/filename) as d:
            keys=d.files;counts[sheet]=write_sheet(wb,sheet,keys,zip(*(d[k] for k in keys)))
    with np.load(OUT/'toy/curved_fields.npz') as d:
        axis=d['axis'];keys=[k for k in d.files if k!='axis']
        fields=[d[k] for k in keys]
        values=([x,y,*(field[iy,ix] for field in fields)] for iy,y in enumerate(axis) for ix,x in enumerate(axis))
        counts['two_dimensional_fields']=write_sheet(wb,'two_dimensional_fields',['x','y',*keys],values)
    path=OUT/'research_data.xlsx';wb.save(path)
    check=load_workbook(path,read_only=True,data_only=True)
    for name,expected in counts.items():
        actual=sum(1 for _ in check[name].iter_rows());assert actual==expected,(name,actual,expected)
    check.close()
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (OUT/'all_image_results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fields);writer.writeheader();writer.writerows(rows)
    c.atomic(OUT/'bundle_validation.json',dict(complete=True,generated_images=34000,image_arms=34,
        sheet_rows=counts,workbook_sha256=c.sha(path),calibration_checks=cal_checks,
        report_requires_separate_final_review=True))
    print(json.dumps(dict(workbook=str(path),generated_images=34000,sheet_rows=counts),indent=2))


if __name__=='__main__':main()
