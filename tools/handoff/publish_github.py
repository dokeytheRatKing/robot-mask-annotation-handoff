"""Publish this prepared repository as a NEW private GitHub repository."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def run(args, **kwargs):
    return subprocess.run(args, cwd=ROOT, text=True, check=True, **kwargs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', default='robot-mask-annotation-handoff', help='New name or OWNER/name; always private')
    a = p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)?', a.repo):
        raise ValueError('Invalid GitHub repository name')
    run([sys.executable, str(ROOT/'tools/handoff/verify_bundle.py')])
    auth = subprocess.run(['gh', 'api', 'user', '--jq', '.login'], cwd=ROOT, text=True, capture_output=True)
    if auth.returncode:
        raise SystemExit('GitHub authentication is missing. Run gh auth login --hostname github.com --web in your terminal, then rerun. Do not paste credentials into files or chat.')
    login = auth.stdout.strip()
    repository = a.repo if '/' in a.repo else login+'/'+a.repo
    status = run(['git', 'status', '--porcelain'], capture_output=True).stdout
    if status:
        raise SystemExit('Commit the prepared delivery before publishing; working tree must be clean.')
    local_head = run(['git', 'rev-parse', 'HEAD'], capture_output=True).stdout.strip()
    existing = subprocess.run(['gh', 'repo', 'view', repository, '--json', 'nameWithOwner'], cwd=ROOT, capture_output=True)
    if existing.returncode == 0:
        raise SystemExit('Destination already exists; no existing repository was modified. Select a new name or inspect its intended ownership/history.')
    remotes = run(['git', 'remote'], capture_output=True).stdout.splitlines()
    if 'origin' in remotes:
        raise SystemExit('Origin is already configured; inspect it before retrying a partially completed publication.')
    run(['gh', 'repo', 'create', repository, '--private', '--source', str(ROOT), '--remote', 'origin',
         '--description', 'Private research handoff: GPT-assisted robot mask annotation, SAM2, provenance, real examples and RoboTwin adapters'])
    run(['git', '-c', 'credential.helper=', '-c', 'credential.helper=!gh auth git-credential', 'push', '-u', 'origin', 'main'])
    info = json.loads(run(['gh', 'repo', 'view', repository, '--json', 'url,isPrivate'], capture_output=True).stdout)
    remote_head = run(['gh', 'api', f'repos/{repository}/commits/main', '--jq', '.sha'], capture_output=True).stdout.strip()
    assert info['isPrivate'] and remote_head == local_head
    print(json.dumps(dict(status='published_and_verified', url=info['url'], private=True, commit=remote_head), indent=2))


if __name__ == '__main__':
    main()
