"""One-time closeout of preexisting SI reading archive; no network or model work."""
from pathlib import Path
from datetime import datetime, timezone
from decimal import Decimal, localcontext
import hashlib, json, subprocess, struct, time

P = Path(__file__).resolve().parent
N = Path('/home/zhoushunyu/eqvae/docs/RAEV2_GUIDANCE_READING_STOCHASTIC_INTERPOLANTS_20260906_ZH.md')
started = time.perf_counter()
now = lambda: datetime.now(timezone.utc).isoformat()
sha = lambda data: hashlib.sha256(data).hexdigest()
for name in ['closeout_verification.json', 'note_snapshot.md', 'manifest.json']:
    assert not (P / name).exists(), name
recovery = json.loads((P / 'closeout_recovery.json').read_text())
for row in recovery['preexisting_files']:
    b = (P / row['path']).read_bytes()
    assert len(b) == row['bytes'] and sha(b) == row['sha256'], row['path']
raw = subprocess.check_output(['pdftotext', '-layout', str(P/'jmlr_23_1605.pdf'), '-'])
assert raw == (P/'jmlr_23_1605.txt').read_bytes()
pages = raw.decode().split('\f')
assert len(pages) == 81 and not pages[-1].strip()
markers = {'(2.24)': [15], '(2.35)': [17], '(2.45)': [19,66], '(B.23)': [60], '(B.30)': [61], '(B.44)': [66], '(B.45)': [66], '(B.48)': [66]}
for token, expected in markers.items():
    assert [i+1 for i,t in enumerate(pages) if token in t] == expected
blobs = []
for repo, owner in [('stochastic-interpolants','malbergo'), ('jax-interpolants','nmboffi')]:
    tree=json.loads((P/(repo+'_tree.json')).read_text())
    lookup={r['path']:r for r in tree['tree']}
    assert not tree['truncated']
    base=P/'code'/owner/repo
    for f in sorted(base.rglob('*')):
        if not f.is_file(): continue
        b=f.read_bytes(); h=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
        assert h==lookup[str(f.relative_to(base))]['sha']
        blobs.append({'path':str(f.relative_to(P)), 'git_blob_sha1':h, 'commit':tree['sha'], 'matches_archived_tree':True})
with localcontext() as context:
    context.prec=80
    e=Decimal(-1).exp(); mean=2*(1-e); kl=mean*mean/2
    assert Decimal('.5') < kl < Decimal(1)
    check=json.loads((P/'analytic_checks.json').read_text())
    assert abs(float(kl)-check['gaussian_excess_risk_constant']['KL_target_to_learned']) < 1e-15
    independent={'decimal_precision':80,'terminal_mean':str(mean),'KL_target_to_learned':str(kl),'printed_2_45_bound':'0.5','corrected_excess_risk_and_Lemma22_bound':'1','original_analytic_json_KL_agrees_within':1e-15,'counterexample_conditions':'Independent standard Gaussian endpoints and added noise; I=(1-t)Y+tW and gamma^2=2t(1-t) meet Definition 1 and Assumption 5. rho=N(0,1), b=0, s=-x, bhat=1, shat=-x+1. Both approximate fields smooth and OU nonexplosive.','reverse_time_correct_variance_derivative_at_zero':-2*2+2*2,'reverse_time_printed_variance_derivative_at_zero':-2*2+2*1}
pngs=[]
for k in [17,19,52,66,74]:
    b=(P/f'page_{k}.png').read_bytes(); assert b[:8]==b'\x89PNG\r\n\x1a\n'
    pngs.append({'file':f'page_{k}.png','pdf_page_one_based':k,'size_pixels':list(struct.unpack('>II',b[16:24])),'visual_page_number_and_content_checked_during_closeout':True})
report={'closeout_started_utc':recovery['closeout_started_utc'],'closeout_finished_utc':now(),'preexisting_source_or_derived_files_preserved':len(recovery['preexisting_files']),'preexisting_files_all_sha_and_bytes_unchanged':True,'pdf_pages':80,'pdftotext_layout_exact_byte_parity':True,'equation_locations_one_based_pdf_pages':markers,'rendered_pages':pngs,'source_code_blob_checks':blobs,'independent_scalar_checks':independent,'scope':'Existing note and 24 archive files preceded this closeout. Five existing PNGs visually checked; definitions/proofs at pp.9,12-15,17,19,60-61,63-66 independently consulted. Earlier full reading scope is retained as a historical record, not claimed repeated. No new paper, downloads, GPU, training or generated samples. These are local mathematical checks, not author-confirmed errata.','calls':{'network_download':0,'GPU':0,'model_forward':0,'training':0,'samples_generated':0},'closeout_script_wall_seconds_before_report_write':time.perf_counter()-started}
(P/'closeout_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
(P/'note_snapshot.md').write_bytes(N.read_bytes())
static={'jmlr_23_1605.pdf':'https://jmlr.org/papers/volume26/23-1605/23-1605.pdf','jmlr_landing.html':'https://jmlr.org/papers/v26/23-1605.html','arxiv_metadata.html':'https://arxiv.org/abs/2303.08797','author_publications.html':'https://nmboffi.github.io/publications/'}
files=[]
old={r['path'] for r in recovery['preexisting_files']}
for f in sorted(P.rglob('*')):
    if not f.is_file() or f.name=='manifest.json':continue
    rel=str(f.relative_to(P));b=f.read_bytes()
    r={'path':rel,'bytes':len(b),'sha256':sha(b),'present_before_closeout':rel in old}
    if rel in static:
        r.update(source_kind='primary_source_archive',source_url=static[rel],provenance_note='URL recovered from canonical metadata, formal paper link, or the existing reading note; original HTTP acquisition log is not available in this closeout.')
    elif rel.endswith('_repo.json') or rel.endswith('_commit.json') or rel.endswith('_tree.json'):
        meta=json.loads(b);r.update(source_kind='primary_GitHub_API_archive',source_url=meta['url']+('?recursive=1' if rel.endswith('_tree.json') else ''),provenance_note='Resource URL recorded in archived GitHub JSON; recursive query inferred from complete file tree.')
    elif rel.startswith('code/'):
        owner,repo,*rest=rel.split('/')[1:];commit=json.loads((P/(repo+'_commit.json')).read_text())['sha'];r.update(source_kind='primary_author_code',source_url=f'https://raw.githubusercontent.com/{owner}/{repo}/{commit}/'+ '/'.join(rest),derived_from=[repo+'_commit.json',repo+'_tree.json'],provenance_note='Pinned raw URL reconstructed; every code file independently matches its Git blob SHA in the archived commit tree.')
    elif rel=='jmlr_23_1605.txt':r.update(source_kind='derived',derived_from=['jmlr_23_1605.pdf'],derivation='pdftotext -layout; exact byte parity reverified in closeout.')
    elif rel.startswith('page_') and rel.endswith('.png'):r.update(source_kind='derived',derived_from=['jmlr_23_1605.pdf'],derivation='Preexisting selected PDF page rendering; filename is one-based PDF page. Render command not preserved; page and visible content independently checked.')
    elif rel in ['analytic_checks.py','analytic_checks.json']:r.update(source_kind='derived',derived_from=['jmlr_23_1605.pdf']+(['analytic_checks.py'] if rel.endswith('.json') else []),derivation='Earlier deterministic local scalar checks, preserved unchanged; not experimental quality evidence.')
    elif rel=='note_snapshot.md':r.update(source_kind='derived',derived_from=[str(N),'jmlr_23_1605.pdf','closeout_verification.json'],derivation='Exact copy of finalized closeout note, preserving earlier reading and marking independent closeout scope.')
    elif rel=='closeout_recovery.json':r.update(source_kind='derived',derived_from=sorted(old),derivation='Observation at closeout start of preexisting bytes, hashes and mtimes; observed mtimes do not certify acquisition times.')
    elif rel=='closeout_verification.json':r.update(source_kind='derived',derived_from=['finalize_closeout.py','closeout_recovery.json','analytic_checks.json','jmlr_23_1605.pdf','jmlr_23_1605.txt'],derivation='Independent inventory, source blob, equation location, PDF text, and deterministic Decimal checks.')
    elif rel=='finalize_closeout.py':r.update(source_kind='derived',derived_from=['jmlr_23_1605.pdf','analytic_checks.py','closeout_recovery.json'],derivation='Local one-time closeout verifier and manifest generator; no network or model execution.')
    else:raise RuntimeError('unmapped source '+rel)
    files.append(r)
manifest={'schema_version':1,'created_utc':now(),'paper':'Stochastic Interpolants: A Unifying Framework for Flows and Diffusions','paper_identity':'JMLR 26(209):1-80, 2025; arXiv 2303.08797 metadata v4','archive_root':str(P),'closeout_started_utc':recovery['closeout_started_utc'],'preexisting_file_count':len(old),'file_count_excluding_manifest':len(files),'total_bytes_excluding_manifest':sum(r['bytes'] for r in files),'manifest_self_excluded':True,'note_path':str(N),'note_sha256':sha(N.read_bytes()),'note_bytes':N.stat().st_size,'source_boundary':'No earlier HTTP acquisition log claimed; files existed before closeout and retain their original bytes. URLs are recovered from archived canonical/API metadata and the earlier note. Derived materials are explicitly distinguished. This manifest itself is excluded to avoid self-reference.','files':files}
(P/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
for r in files:
    b=(P/r['path']).read_bytes();assert len(b)==r['bytes'] and sha(b)==r['sha256']
print(json.dumps({'files':len(files),'bytes':manifest['total_bytes_excluding_manifest'],'note_sha256':manifest['note_sha256'],'manifest_sha256':sha((P/'manifest.json').read_bytes()),'finished_utc':manifest['created_utc']},indent=2))
