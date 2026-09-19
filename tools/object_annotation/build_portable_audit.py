"""Build the exact relocatable audit ZIP, without models or Python dependencies."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

BASE=Path(__file__).resolve().parent
PROJECT=BASE.parents[1]


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--name',required=True);a=p.parse_args()
    root=a.output/a.name;root.mkdir(parents=True,exist_ok=False)
    audit=root/'audit';audit.mkdir();(audit/'images').mkdir();(audit/'human_labels').mkdir()
    manifest=json.loads((a.audit/'manifest.json').read_text());shutil.copy2(a.audit/'manifest.json',audit/'manifest.json')
    for s in manifest['samples']:
        rel=Path(s['image']);assert not rel.is_absolute() and '..' not in rel.parts
        source=a.audit/rel;assert digest(source)==s['image_sha256']
        shutil.copy2(source,audit/rel)
    for source,target in [('audit_server.py','audit_server.py'),('audit_rle.py','audit_rle.py'),
        ('audit_ui.html','audit_ui.html'),('portable_start.py','start.py'),
        ('export_audit_labels.py','export_labels.py'),('PORTABLE_AUDIT_README.md','README_先读我.md')]:
        shutil.copy2(BASE/source,root/target)
    # Existing human work, if any, is retained in the package.
    for path in (a.audit/'human_labels').glob('*.json'):shutil.copy2(path,audit/'human_labels'/path.name)
    if (a.audit/'proposed_labels').exists():
        shutil.copytree(a.audit/'proposed_labels',audit/'proposed_labels')
    history=a.audit/'human_label_revisions.jsonl'
    if history.exists():shutil.copy2(history,audit/history.name)
    info=dict(bundle_id=a.name,created_utc=datetime.now(timezone.utc).isoformat(),frames=len(manifest['samples']),
        object_frame_labels=sum(len(s['objects']) for s in manifest['samples']),
        audit_manifest_sha256=digest(audit/'manifest.json'),python='3.9+ standard library only',
        images='RGB PNG; confirmed labels take precedence over explicitly unconfirmed proposals')
    (root/'bundle_info.json').write_text(json.dumps(info,indent=2)+'\n',encoding='utf-8')
    checks=''.join(f'{digest(path)}  {path.relative_to(root).as_posix()}\n' for path in sorted(root.rglob('*')) if path.is_file())
    (root/'SHA256SUMS').write_text(checks,encoding='utf-8')
    archive=a.output/f'{a.name}.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in sorted(root.rglob('*')):
            if path.is_file():z.write(path,path.relative_to(a.output).as_posix())
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    checksum=digest(archive);archive.with_suffix('.zip.sha256').write_text(f'{checksum}  {archive.name}\n',encoding='utf-8')
    print(json.dumps(dict(path=str(archive),bytes=archive.stat().st_size,sha256=checksum,frames=info['frames']),indent=2))


if __name__=='__main__':main()
