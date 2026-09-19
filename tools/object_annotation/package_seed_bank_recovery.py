"""Export validated recovery QA without raw frames or modifications to source data."""
import csv
import json
import zipfile

from annotate import BASE, PROJECT, sha, write_json
from seed_bank_recovery import DEFAULT


def main():
    root=DEFAULT
    validation=json.loads((root/'validation.json').read_text())
    independent=json.loads((root/'independent_validation.json').read_text())
    assert validation['status']==independent['status']=='PASS'
    score=json.loads((root/'scoring.json').read_text())
    for name,rows in [('summary',score['summary']),('object_frame_scores',score['rows'])]:
        with (root/(name+'.csv')).open('x',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    paths=[*root.glob('*.json'),*root.glob('*.csv'),*root.glob('qa/*/*.mp4'),*root.glob('qa/*/*.jpg'),*root.glob('inspection/*/*.jpg')]
    for name in ('complete.json','events.json','checks.json'):
        paths.extend(root.glob('*/'+name))
    for name in ('seed_bank_recovery.py','validate_seed_bank_recovery.py','test_seed_bank_recovery.py','package_seed_bank_recovery.py','inspect_seed_bank_recovery.py','diagnose_recovery_candidates.py'):
        paths.append(BASE/name)
    paths.append(PROJECT/'docs/identity_seed_bank_recovery_report_20260918.md')
    for v in validation['videos']:assert sha(root/v['path'])==v['sha256']
    paths=sorted(set(paths));files=[]
    target=PROJECT/'deliverables/astribot_seed_bank_recovery_qa_20260918.zip'
    with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for path in paths:
            assert path.is_file()
            name=str(path.relative_to(PROJECT));z.write(path,name)
            files.append(dict(path=name,bytes=path.stat().st_size,sha256=sha(path)))
        z.writestr('README.txt',
            'Open docs/identity_seed_bank_recovery_report_20260918.md first.\n'
            'QA videos: annotations/identity_seed_bank_recovery_20260918_r2/qa/*/comparison.mp4\n'
            'Six panels: RGB | frozen robot QA | single_detector / single_bank | recovery_detector | recovery_bank.\n'
            'Masks are automatic drafts, not accepted ground truth. Confidence is SAM presence, not identity accuracy.\n'
            'No server, GPU, web page or new manual annotation is required to watch these MP4s.\n'
            'Full per-frame RLE outputs and candidate retrieval caches remain on the server.\n'
            'Scripts depend on the existing server repository and annotation environment.\n')
        z.writestr('files.sha256.json',json.dumps(files,indent=2))
    with zipfile.ZipFile(target) as z:assert z.testzip() is None
    result=dict(path=str(target),bytes=target.stat().st_size,sha256=sha(target),files=len(files),zip_test='PASS')
    write_json(target.with_suffix('.zip.manifest.json'),result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
