"""Package verified local MP4s, diagnostics and reproducible transfer code."""
import csv
import io
import json
from pathlib import Path
import zipfile

from annotate import PROJECT,sha,write_json


def main():
    roots=[PROJECT/'annotations'/f'identity_seed_bank_transfer_20260918_{suffix}' for suffix in ('r2','eligible')]
    dest=PROJECT/'deliverables/astribot_seed_bank_transfer_qa_20260918.zip'
    files=[PROJECT/'docs/identity_seed_bank_transfer_report_20260918.md',PROJECT/'tools/object_annotation/SEED_BANK.md']
    files.extend(PROJECT/'tools/object_annotation'/name for name in (
        'seed_bank_transfer.py','seed_bank_transfer_qa.py','validate_seed_bank_transfer.py',
        'test_seed_bank_transfer.py','package_seed_bank_transfer.py'))
    for root in roots:
        assert json.loads((root/'validation.json').read_text())['status']=='PASS'
        qa=json.loads((root/'qa_manifest.json').read_text());assert qa['status']=='PASS'
        for item in qa['videos']:assert sha(root/item['path'])==item['sha256']
        scoring=json.loads((root/'scoring.json').read_text())
        for key,name in [('summary','summary.csv'),('rows','object_frame_scores.csv')]:
            with (root/name).open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(scoring[key][0]));w.writeheader();w.writerows(scoring[key])
        files.extend(root/name for name in ('run_config.json','complete.json','validation.json',
            'scoring.json','summary.csv','object_frame_scores.csv','qa_manifest.json','bank_acceptance.json'))
        files.extend(root.glob('qa_verified/*/*.mp4'));files.extend(root.glob('qa_verified/*/*.jpg'))
        files.extend(root.glob('*/selections.json'));files.extend(root.glob('*/candidate_source.json'))
        files.extend(root.glob('*/complete.json'))
    files.append(roots[0]/'assistant_visual_review.json')
    files.append(roots[0]/'inference_script_snapshot.py')
    readme='''# Seed Bank transfer comparison

Open docs/identity_seed_bank_transfer_report_20260918.md first.
Then play any MP4 under annotations/*/qa_verified/ locally. No website/server is needed.
Panel order: RGB | frozen production | detector / bank_rank | bank_gate | unchanged robot QA.
The legends show object ID, class and SAM presence score, NOT identity accuracy.
r2 includes 12 unseen-episode clips and 9 accepted scoring windows.
eligible contains a same-candidate correction: filter low detector scores before ranking.
These are diagnostic drafts, not approved pseudo-GT. No new human annotation requested.
Raw masks/candidate RLE and full logs remain in the corresponding server directories.
Frozen source files were not modified. GPU inference has finished.
'''
    with zipfile.ZipFile(dest,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        z.writestr('README.md',readme)
        manifest=[]
        for p in sorted(set(files)):
            arc=str(p.relative_to(PROJECT));z.write(p,arc)
            manifest.append(dict(path=arc,bytes=p.stat().st_size,sha256=sha(p)))
        z.writestr('file_manifest.json',json.dumps(manifest,indent=2))
    with zipfile.ZipFile(dest) as z:assert z.testzip() is None
    result=dict(path=str(dest),bytes=dest.stat().st_size,sha256=sha(dest),files=len(manifest),zip_test='PASS')
    write_json(dest.with_suffix('.zip.manifest.json'),result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
