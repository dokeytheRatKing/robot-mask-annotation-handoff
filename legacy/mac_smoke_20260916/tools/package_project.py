"""Archive this whole project and verify an actual extracted copy.

Materialize file symlinks for portability. Retain all other workspace files,
including original ZIPs, model weights, SAM source/history, and prior handoffs.
Exclude only this package's own output files to prevent recursive packaging.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def signature(path):
    st = path.stat()
    return st.st_ino, st.st_size, st.st_mtime_ns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'handoffs/astribot_smoke_test_full_project_20260916.zip')
    args = parser.parse_args()
    output = args.output.resolve()
    part = output.with_suffix('.zip.partial')
    sha_path = output.with_suffix('.zip.sha256')
    report_path = output.with_suffix('.zip.verification.json')
    excluded = {output, part, sha_path, report_path}
    assert not output.exists() and not part.exists(), 'Choose a new output name; refusing to overwrite an archive'
    assert not (ROOT/'PACKAGE_MANIFEST.json').exists(), 'Package the source workspace, not an already archived snapshot'
    paths = []
    for base, dirs, files in os.walk(ROOT, followlinks=False):
        for directory in dirs:
            assert not (Path(base)/directory).is_symlink(), 'Directory links need an explicit portable mapping'
        for name in files:
            path = Path(base)/name
            if path in excluded:
                continue
            assert path.is_file(), f'Not a regular file: {path}'
            assert path.resolve().is_relative_to(ROOT), f'External file link: {path}'
            paths.append(path)
    paths.sort()
    original_signatures = {str(p.relative_to(ROOT)):signature(p) for p in paths}
    prefix = output.stem
    entries = []
    started = time.monotonic()
    print(f'PACKAGING {len(paths)} files; file links become portable regular files', flush=True)
    with zipfile.ZipFile(part, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=3, allowZip64=True) as archive:
        for i, path in enumerate(paths, 1):
            relative = path.relative_to(ROOT).as_posix()
            entry = dict(path=relative, size=path.stat().st_size)
            if path.is_symlink():
                entry['source_symlink_target'] = str(path.readlink())
                entry['packaged_as'] = 'regular_file_with_target_contents'
            info = zipfile.ZipInfo.from_file(path, arcname=f'{prefix}/{relative}')
            info.compress_type = zipfile.ZIP_DEFLATED
            info._compresslevel = 3
            hasher = hashlib.sha256()
            with path.open('rb') as source, archive.open(info, 'w', force_zip64=True) as target:
                for chunk in iter(lambda:source.read(1 << 20), b''):
                    hasher.update(chunk); target.write(chunk)
            assert signature(path) == original_signatures[relative], f'File changed while packaging: {relative}'
            entry['sha256'] = hasher.hexdigest()
            entries.append(entry)
            if i % 1000 == 0:
                print(f'PACKED {i}/{len(paths)} in {round(time.monotonic()-started)}s', flush=True)
        for path in paths:
            assert signature(path) == original_signatures[path.relative_to(ROOT).as_posix()], f'Source changed: {path}'
        package_manifest = dict(schema='astribot.smoke_test_full_project_snapshot.v1',
            created_utc=datetime.now(timezone.utc).isoformat(), archive_root=prefix,
            source='entire annotation workspace', file_count=len(entries),
            materialized_file_symlinks=sum('source_symlink_target' in e for e in entries),
            exclusions=[str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p) for p in sorted(excluded)],
            inventory_note='All payload files are listed; PACKAGE_MANIFEST.json does not hash itself.',
            files=entries)
        archive.writestr(f'{prefix}/PACKAGE_MANIFEST.json', json.dumps(package_manifest, ensure_ascii=False, indent=2)+'\n')
    print('EXTRACTING AND VERIFYING: full file hashes, images, labels, revisions and model', flush=True)
    # Extract all files instead of relying only on the central directory or CRC.
    with tempfile.TemporaryDirectory(prefix='astribot-project-package-check-') as tmp:
        with zipfile.ZipFile(part) as archive:
            names = archive.namelist()
            assert len(names) == len(set(names)) == len(entries)+1
            assert all(n.startswith(prefix+'/') and '..' not in Path(n).parts for n in names)
            archive.extractall(tmp)  # Reading verifies the ZIP member CRCs too.
        extracted_root = Path(tmp)/prefix
        run = subprocess.run([sys.executable, str(extracted_root/'tools/verify_project_snapshot.py')],
            cwd=extracted_root, capture_output=True, text=True, check=True)
        verified = json.loads(run.stdout)
        assert verified['result'] == 'PASS' and verified['package_files_verified'] == len(entries)
        assert verified['confirmed'] == 1860 and verified['frames'] == 420
    part.rename(output)
    archive_sha = sha_file(output)
    sha_path.write_text(f'{archive_sha}  {output.name}\n')
    report = dict(result='PASS', archive=output.name, bytes=output.stat().st_size,
        sha256=archive_sha, payload_files=len(entries), total_zip_members=len(entries)+1,
        payload_bytes=sum(e['size'] for e in entries),
        materialized_file_symlinks=package_manifest['materialized_file_symlinks'],
        extraction='all files extracted into a temporary directory; CRC verified on read',
        extracted_project_verification=verified,
        seconds=round(time.monotonic()-started,2))
    report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    main()
