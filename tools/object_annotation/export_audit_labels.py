"""Export only human labels and revision history for return to the HKU server."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import zipfile
from audit_rle import decode_counts


def export(bundle,output=None):
    bundle=Path(bundle).resolve();audit=bundle/'audit';manifest_bytes=(audit/'manifest.json').read_bytes()
    manifest=json.loads(manifest_bytes);samples={s['sample_id']:s for s in manifest['samples']}
    files=sorted((audit/'human_labels').glob('*.json'))
    if not files:raise ValueError('No saved labels yet. Save at least one object in the editor before exporting.')
    data={};completed=set();unique=set()
    for path in files:
        raw=path.read_bytes();row=json.loads(raw);s=samples[row['sample_id']]
        key=(row['sample_id'],row['object_id']);assert key not in unique;unique.add(key)
        assert row['object_id'] in [o['object_id'] for o in s['objects']]
        assert row['source_image_sha256']==s['image_sha256']
        assert row['mask']['size']==[s['image_height'],s['image_width']]
        assert row['human_confirmed'] is True and row['annotation_method']=='manual_pixel_editor'
        assert row['annotator'].strip() and row['instance_id'].strip()
        assert row['visibility'] in ['visible','partial_occlusion','fully_occluded','out_of_view','unknown']
        area=sum(decode_counts(row['mask'])[1::2])
        assert bool(area)==(row['visibility'] in ['visible','partial_occlusion'])
        assert path.name==f'{row["sample_id"]}__{row["object_id"]}.json'
        data['human_labels/'+path.name]=raw;completed.add(f'{row["sample_id"]}:{row["object_id"]}')
    revisions=audit/'human_label_revisions.jsonl'
    if revisions.exists():
        raw=revisions.read_bytes()
        for line in raw.splitlines():json.loads(line)
        data['human_label_revisions.jsonl']=raw
    manifest_sha=hashlib.sha256(manifest_bytes).hexdigest()
    info=dict(schema='astribot.human_mask_audit_return.v1',audit_manifest_sha256=manifest_sha,
        exported_utc=datetime.now(timezone.utc).isoformat(),labels=len(files),frames=len(samples),
        required_labels=sum(len(s['objects']) for s in samples.values()),
        completed_frames=sum(all(f'{s["sample_id"]}:{o["object_id"]}' in completed for o in s['objects']) for s in samples.values()),
        files_sha256={name:hashlib.sha256(raw).hexdigest() for name,raw in data.items()})
    data['return_info.json']=(json.dumps(info,ensure_ascii=False,indent=2)+'\n').encode('utf-8')
    output=Path(output) if output else bundle/f'mask_audit_return_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}.zip'
    with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for name,raw in data.items():z.writestr(name,raw)
    with zipfile.ZipFile(output) as z:assert z.testzip() is None
    digest=hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix('.zip.sha256').write_text(f'{digest}  {output.name}\n',encoding='utf-8')
    print(f'Exported: {output}\nLabels: {len(files)}/{info["required_labels"]}; complete frames: {info["completed_frames"]}/{len(samples)}\nSHA256: {digest}\nUpload this ZIP. Original images are not included. Local progress is retained.')
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--output',type=Path);a=p.parse_args()
    try:export(a.bundle,a.output)
    except (ValueError,AssertionError) as exc:raise SystemExit(str(exc) or 'Invalid label data; export stopped.')


if __name__=='__main__':main()
