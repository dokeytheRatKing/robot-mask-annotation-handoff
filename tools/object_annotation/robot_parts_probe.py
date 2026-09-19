"""RGB and measured-joint evidence for robot-side and flange seed review."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

from annotate import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--episodes',nargs='+',default=['episode_001603','episode_001684'])
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    eps={e['episode_id']:e for e in json.loads((a.root/'episodes.json').read_text())['episodes']}
    report=[]
    for name in a.episodes:
        ep=eps[name];state=np.stack(pq.read_table(ep['parquet'],columns=['observation.state'])['observation.state'].to_numpy())
        report.append(dict(episode=name,side=ep['side'],left_arm_total_motion=np.abs(np.diff(state[:,7:14],axis=0)).sum().item(),
            right_arm_total_motion=np.abs(np.diff(state[:,15:22],axis=0)).sum().item(),
            left_gripper_range=np.ptp(state[:,14]).item(),right_gripper_range=np.ptp(state[:,22]).item()))
        tiles=[]
        for camera in ['head','left_wrist','right_wrist']:
            cap=cv2.VideoCapture(ep['cameras'][camera]['video'])
            for idx in np.linspace(0,ep['frames']-1,4,dtype=int):
                cap.set(cv2.CAP_PROP_POS_FRAMES,int(idx));ok,im=cap.read();assert ok
                if idx==0:cv2.imwrite(str(a.output/f'{name}_{camera}_000000.png'),im)
                tile=cv2.resize(im,(480,270));tile=np.pad(tile,((25,0),(0,0),(0,0)))
                cv2.putText(tile,f'{camera} f{idx} L{state[idx,14]:.2f} R{state[idx,22]:.2f}',(5,18),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
                tiles.append(tile)
            cap.release()
        sheet=np.concatenate([np.concatenate(tiles[j:j+4],1) for j in range(0,len(tiles),4)],0)
        cv2.imwrite(str(a.output/f'{name}_evidence.jpg'),sheet)
    write_json(a.output/'joint_evidence.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
