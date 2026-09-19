"""Make a writable copy of the confirmed demo without modifying the source."""
import argparse
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
source = ROOT / 'examples/confirmed_audit'
dest = a.output.resolve()
if dest.is_relative_to(ROOT/'examples'):
    raise SystemExit('Output must be outside source examples')
shutil.copytree(source, dest)
print(dest)
