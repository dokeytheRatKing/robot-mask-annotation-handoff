"""Preserve only this job's four incomplete release outputs before retry."""
from calibration64 import ROOT, load

cfg=load(ROOT)
dest=ROOT/'failed_release_serialization';dest.mkdir(exist_ok=False)
for c in cfg['clips'][:4]:
    path=ROOT/'clips'/c['clip_id']/'release'
    assert path.is_dir() and not (path/'validation.json').exists()
    path.rename(dest/c['clip_id'])
    print('Archived incomplete output:',c['clip_id'])
