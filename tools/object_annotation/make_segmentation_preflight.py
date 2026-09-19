"""Create a finite queue with exactly the production code/config/assets snapshot."""
import argparse
import json
from pathlib import Path
import shutil

from annotate import sha, write_json
from data import CAMERAS
import full_queue as queue


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--episodes',nargs='+',required=True);p.add_argument('--max-frames',type=int,default=0);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    for folder in ['code','config','identity_gallery','robot_reference','reviewed_reference']:
        if (a.source/folder).exists():shutil.copytree(a.source/folder,a.output/folder)
    for folder in ['logs','overrides','review']:(a.output/folder).mkdir()
    cfg=json.loads((a.source/'run_config.json').read_text())
    cfg['preflight_of_config_sha256']=sha(a.source/'run_config.json')
    cfg['preflight_episode_ids']=a.episodes
    eps=json.loads((a.source/'episodes.json').read_text())['episodes']
    eps=[e for e in eps if e['episode_id'] in a.episodes];assert len(eps)==len(set(a.episodes))
    if a.max_frames:
        assert a.max_frames>0
        for ep in eps:
            ep['preflight_original_frames']=ep['frames'];ep['frames']=min(a.max_frames,ep['frames'])
        cfg['preflight_max_frames']=a.max_frames
    write_json(a.output/'episodes.json',dict(episodes=eps))
    cfg['manifest_sha256']=sha(a.output/'episodes.json')
    write_json(a.output/'run_config.json',cfg)
    queue.initialize(a.output,[(e['episode_id'],c,e['task_id'],e['frames'],0) for e in eps for c in CAMERAS])
    print(json.dumps(dict(episodes=len(eps),streams=len(eps)*3,frames=sum(e['frames'] for e in eps)*3)))


if __name__=='__main__':main()
