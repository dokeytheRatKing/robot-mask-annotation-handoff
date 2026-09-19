"""Package completed pilot artifacts for headless-server offline inspection."""
from pathlib import Path
import json
import zipfile

from annotate import PROJECT, BASE, sha
from seed_admission_pilot import DEFAULT


if __name__ == '__main__':
    cfg = json.loads((DEFAULT / 'run_config.json').read_text())
    independent = json.loads((DEFAULT / 'independent_validation.json').read_text())
    validation = json.loads((DEFAULT / 'validation.json').read_text())
    assert validation['status'] == independent['status'] == 'PASS'
    assert independent['config_sha256'] == sha(DEFAULT / 'run_config.json')
    assert independent['validator_sha256'] == sha(BASE / 'validate_seed_admission.py')
    files = [p for p in DEFAULT.rglob('*') if p.is_file() and 'seeds_r1' not in p.parts]
    files += [PROJECT / 'docs/seed_admission_assisted_pilot_report_20260918.md']
    files += [BASE / p for p in ('seed_admission_pilot.py', 'validate_seed_admission.py',
                                'test_seed_admission.py', 'revise_admission_prompts.py',
                                'package_seed_admission.py', 'SEED_BANK.md')]
    files += list((BASE / 'config').glob('admission_assistant_prompts_20260918*.json'))
    for case in cfg['cases']:
        assert DEFAULT / case['case_id'] / 'complete.json' in files
    for video in validation['videos']:
        assert sha(DEFAULT / video['path']) == video['sha256']
    files = sorted(set(files))
    target = PROJECT / 'deliverables/astribot_seed_admission_assisted_qa_20260918.zip'
    assert not target.exists()
    manifest = []
    with zipfile.ZipFile(target, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=5) as z:
        for path in files:
            relative = path.relative_to(PROJECT).as_posix()
            manifest.append(f'{sha(path)}  {relative}')
            method = zipfile.ZIP_STORED if path.suffix in ('.jpg', '.mp4', '.gz') else zipfile.ZIP_DEFLATED
            z.write(path, relative, compress_type=method)
        z.writestr('checksums.sha256', '\n'.join(manifest) + '\n')
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        assert len(z.namelist()) == len(files) + 1
    print(json.dumps(dict(path=str(target), files=len(files)+1, bytes=target.stat().st_size,
                          sha256=sha(target), zip_crc='PASS'), indent=2))
