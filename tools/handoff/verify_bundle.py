"""Verify the delivered manifest using Python standard library only."""
from pathlib import Path
import ast
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = ROOT / 'SHA256SUMS'
    if not manifest.exists():
        raise SystemExit('SHA256SUMS missing; delivery has not been finalized')
    count = 0
    for line in manifest.read_text().splitlines():
        expected, relative = line.split('  ', 1)
        path = (ROOT / relative).resolve()
        assert path.is_relative_to(ROOT) and path.is_file() and not Path(relative).is_absolute()
        assert digest(path) == expected, relative
        count += 1
    snapshot = json.loads((ROOT / 'SOURCE_SNAPSHOT.json').read_text())
    for source in snapshot['sources']:
        assert digest(ROOT / source['target']) == source['exported_sha256'], source['target']
    scripts = list((ROOT/'tools').rglob('*.py')) + list((ROOT/'legacy').rglob('*.py'))
    for path in scripts:
        ast.parse(path.read_text(), filename=str(path))
    print(json.dumps(dict(status='PASS', manifest_files=count, source_snapshot_files=len(snapshot['sources']),
        parsed_python_files=len(scripts), gpu_used=False), indent=2))


if __name__ == '__main__':
    main()
