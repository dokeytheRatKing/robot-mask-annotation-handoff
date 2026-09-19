#!/usr/bin/env python3
"""Full-frame, task-conditioned SAM2 segmentation with independent GPU workers.

Commands: prepare, worker, supervise, status, requeue, validate.
Automatic masks are predictions, never implicitly human-confirmed pseudo-GT.
"""
import argparse
from collections import Counter, OrderedDict, defaultdict
from contextlib import ExitStack
import gc
import gzip
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from annotate import BASE, PROJECT, sha, write_json
from data import CAMERAS, scan, select, frames
from masks import bbox, decode, encode, record
from recovery import Settings, box_iou, diagnostics
import full_queue as queue


def jline(handle, value):
    handle.write(json.dumps(value, allow_nan=False, separators=(',', ':')) + '\n')


def candidates_for_task(detections, ids, settings):
    """Containment is expected in manipulation; reject only near-identical boxes."""
    chosen,rejected={},{}
    for oid in ids:
        hits=sorted([d for d in detections if d['object_id']==oid and d['confidence']>=settings.detection_threshold
                     and d.get('appearance_accepted',True)],
                    key=lambda d:d['confidence'],reverse=True)
        if not hits: continue
        if len(hits)>1 and hits[0]['confidence']-hits[1]['confidence']<settings.ambiguous_margin:
            rejected[oid]='ambiguous_multiple_instances'
        else: chosen[oid]=hits[0]
    conflicts=set()
    for oid,a in chosen.items():
        for other,b in chosen.items():
            if oid>=other or {oid,other}<={17,22,23}: continue
            if box_iou(a['bbox_xyxy'],b['bbox_xyxy'])>.80: conflicts.update([oid,other])
    for oid in conflicts:
        chosen.pop(oid);rejected[oid]='near_identical_cross_class_boxes'
    return chosen,rejected


def blocks(iterator, size):
    """One shared frame transfers masks without pairing different timestamps."""
    previous = None
    while True:
        new = list(itertools.islice(iterator, size))
        if not new:
            return
        yield ([previous] if previous is not None else []) + new, previous is not None
        previous = new[-1]


def episode_frames(ep,camera):
    source_ep=ep|dict(frames=ep.get('preflight_original_frames',ep['frames']))
    return frames(source_ep,camera,max_frames=ep['frames'])


def render(image, rows, robot, title, parts=None):
    out = image.copy()
    robot_rows = parts if parts is not None else [robot] if robot else []
    for row in robot_rows + rows:
        if not row.get('mask') or not row.get('visible'):
            continue
        binary = decode(row['mask'])
        seed = row.get('part_id',row['object_id'])
        seed = seed if isinstance(seed,int) else 999
        from robot_parts import COLORS
        color = np.asarray(COLORS[seed]) if seed in COLORS else np.random.default_rng(seed + 7).integers(65, 245, 3)
        out[binary] = (out[binary] * .5 + color * .5).astype(np.uint8)
        if row['bbox_xyxy']:
            x0,y0,x1,y1 = row['bbox_xyxy']
            cv2.rectangle(out, (x0,y0), (x1,y1), tuple(map(int,color)), 1)
            scale=max(1,image.shape[1]/640)
            score=row.get('confidence');label=row['class_name']+(f' {score:.2f}' if score is not None else '')
            cv2.putText(out,label,(x0,max(int(14*scale),y0)),cv2.FONT_HERSHEY_SIMPLEX,
                        .45*scale,tuple(map(int,color)),max(1,int(scale)))
    canvas = np.concatenate([cv2.resize(image,(640,360)), cv2.resize(out,(640,360))],1)
    canvas = np.pad(canvas, ((30,0),(0,0),(0,0)))
    cv2.putText(canvas, title, (6,20), cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1)
    return canvas


def prepare(args):
    root = args.output.resolve()
    source = args.input.resolve()
    if root == source or source in root.parents:
        raise ValueError('Output must be outside the source dataset')
    episodes = scan(source)
    tasks = sorted({ep['task_id'] for ep in episodes})
    episodes = select(episodes, tasks, 0, 20260917)
    info = json.loads((source/'meta/info.json').read_text())
    assert len(episodes) == info['total_episodes']
    assert sum(ep['frames'] for ep in episodes) == info['total_frames']
    objects = json.loads((BASE/'config/objects.json').read_text())
    descriptions = {
        0:'a white ceramic coffee cup with one handle',
        1:'a light blue toy kettle with a black handle and a short spout',
        2:'a yellow curved toy banana',
        3:'a toy spoon with a round pale blue bowl and a long black handle',
        4:'a rectangular golden woven basket',
        5:'a pale blue toy frying pan with a long light wooden handle',
        6:'a pale blue toy pressure cooker without a lid with a black long handle',
        8:'a rectangular metal wire dish drying rack',
        10:'a pink round toy peach',
        11:'an off-white plastic drawer storage box',
        12:'a colorful patterned cleaning sponge with a long narrow shape',
        13:'a pale blue rectangular Chinese cleaver with a black handle',
        14:'a gray divided rectangular countertop utensil holder',
        15:'a pale blue toy soup pot with two short light wooden handles',
        16:'a white handheld dusting brush with dark bristles',
        18:'a light wooden mug tree with a round base and branching pegs',
        19:'a pale blue slotted toy spatula with a long black handle',
        20:'a red toy watermelon slice with a green rind',
        21:'a green halved toy avocado with a large brown pit',
    }
    for obj in objects:
        if obj['object_id'] in descriptions: obj['detailed_text_prompt']=descriptions[obj['object_id']]
    mapping = json.loads((BASE/'config/task_objects.json').read_text())
    # These containers are named in the task steps but omitted from its ID cell.
    mapping['22'] = sorted(set(mapping['22']) | {8})
    mapping['23'] = sorted(set(mapping['23']) | {5})
    root.mkdir(parents=True, exist_ok=False)
    for name in ['code','config','logs','overrides','review','inventory']:
        (root/name).mkdir()
    for p in BASE.glob('*.py'):
        shutil.copy2(p,root/'code'/p.name)
    write_json(root/'config/objects.json',objects)
    write_json(root/'config/task_objects.json',mapping)
    shutil.copy2(BASE/'config/robot_objects.json',root/'config/robot_objects.json')
    shutil.copy2(BASE/'config/robot_parts.json',root/'config/robot_parts.json')
    if args.appearance_gallery:
        shutil.copytree(args.appearance_gallery,root/'identity_gallery')
    if args.robot_reference:
        shutil.copytree(args.robot_reference,root/'robot_reference')
    reviewed = PROJECT/'annotations/mask_audit_confirmed_20260917'
    shutil.copytree(reviewed/'objects',root/'reviewed_reference')
    write_json(root/'episodes.json',dict(episodes=episodes))
    conf = dict(schema='astribot.segmentation.full.v1',project=str(PROJECT),dataset=str(source),
        cameras=list(CAMERAS),stride=1,chunk_frames=120,detection_period=30,
        detection_threshold=.40,ambiguity_margin=.06,search_cooldown=10,
        sam2_checkpoint=str(PROJECT/'models/sam2/sam2.1_hiera_base_plus.pt'),
        sam2_config='configs/sam2.1/sam2.1_hiera_b+.yaml',
        dino_checkpoint=str(PROJECT/'models/groundingdino/groundingdino_swint_ogc.pth'),
        dino_config=str(PROJECT/'third_party/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'),
        bert=str(PROJECT/'models/groundingdino/bert-base-uncased'),
        inference_precision='SAM2 bfloat16 autocast; DINO float32',mask_threshold_logit=0,
        reviewed_reference=str(reviewed),reviewed_reference_report_sha256=sha(reviewed/'import_report.json'),
        source_manifest_sha256=sha(source/'meta/source_manifest.json'),
        info_sha256=sha(source/'meta/info.json'),manifest_sha256=sha(root/'episodes.json'),
        prompt_provenance='Codex RGB inspection of 29 task head views on 2026-09-17; source IDs/names preserved',
        mapping_notes=['Task22 adds drying rack 8 from task steps; task23 adds frying pan 5.',
            'Task26 keeps source ID 1 (kettle) despite description mismatch; flag for visual review.',
            'Meat whole/parts 17/22/23 are overlapping semantic levels, not mutually exclusive instances.'],
        visibility_semantics='visible is predicted mask existence; uninitialized is null, not proven out-of-view',
        robot_policy='separate union mask, anonymous camera-local components, QA only; never subtract from objects',
        quality_policy='Automatic draft; structural completion is not semantic acceptance. Exceptions go to assistant review.',
        seed=20260917,created_at=time.time())
    if args.appearance_gallery:
        conf.update(schema='astribot.segmentation.full.v2',appearance_gallery='identity_gallery/gallery.npz',
                    appearance_threshold=.50,appearance_margin=.05,
                    appearance_checkpoint_sha256=sha(PROJECT/'models/dinov2/dinov2_vits14_pretrain.pth'),
                    appearance_revision=subprocess.check_output(['git','-C',str(PROJECT/'third_party/dinov2'),'rev-parse','HEAD'],text=True).strip(),
                    identity_policy='Competitive robot/background grounding + frozen exemplar veto; reviewed segment seeds take priority')
    if args.robot_reference:
        conf.update(robot_reference='robot_reference',robot_policy='Independent left/right arm/EE plus robot_unknown; flange included in arm; uncertain boundary is unknown; never subtract from task objects')
    for name in ['sam2','GroundingDINO']:
        conf[name+'_revision'] = subprocess.check_output(['git','-C',str(PROJECT/'third_party'/name),'rev-parse','HEAD'],text=True).strip()
    conf['file_hashes'] = {str(p.relative_to(root)):sha(p) for p in sorted((root/'code').glob('*.py'))}
    conf['file_hashes'].update({str(p.relative_to(root)):sha(p) for p in sorted((root/'config').glob('*.json'))})
    for folder in ['identity_gallery','robot_reference']:
        if (root/folder).exists():
            conf['file_hashes'].update({str(p.relative_to(root)):sha(p) for p in sorted((root/folder).rglob('*')) if p.is_file()})
    conf['sam2_checkpoint_sha256'] = sha(conf['sam2_checkpoint'])
    conf['dino_checkpoint_sha256'] = sha(conf['dino_checkpoint'])
    write_json(root/'run_config.json',conf)
    # Exercise every task/side/camera early, then process the remainder unchanged.
    first = {}
    for ep in episodes:
        first.setdefault((ep['task_id'],ep['side']),ep['episode_id'])
    audited = {'episode_002478','episode_001608','episode_002577'}
    jobs = []
    for ep in episodes:
        priority = 0 if ep['episode_id'] in audited else 10 if ep['episode_id'] == first[(ep['task_id'],ep['side'])] else 20
        for camera in CAMERAS:
            jobs.append((ep['episode_id'],camera,ep['task_id'],ep['frames'],priority))
    queue.initialize(root,jobs)
    result = dict(episodes=len(episodes),streams=len(jobs),camera_frames=sum(j[3] for j in jobs),
                  frames=info['total_frames'],tasks=tasks,task_side_counts=dict(Counter(f"{e['task_id']}:{e['side']}" for e in episodes)))
    write_json(root/'inventory/summary.json',result)
    # Compact scene inventory for actual visual identity checks, not web/UI work.
    tiles = []
    for task in tasks:
        ep = next(e for e in episodes if e['task_id']==task)
        _,_,image = next(frames(ep,'head',max_frames=1))
        tile = cv2.resize(image,(426,240))
        cv2.putText(tile,f"task {task} / {ep['episode_id']}",(5,20),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,255),2)
        tiles.append(tile)
        if len(tiles)==9 or task==tasks[-1]:
            while len(tiles)%3: tiles.append(np.zeros_like(tile))
            page = np.concatenate([np.concatenate(tiles[i:i+3],1) for i in range(0,len(tiles),3)],0)
            cv2.imwrite(str(root/'inventory'/f'tasks_through_{task:02d}.jpg'),page)
            tiles = []
    print(json.dumps(result),flush=True)


class LazyImages:
    """Bounded RGB adapter using the official PIL resize/normalization contract."""
    def __init__(self, images, size, device):
        self.images,self.size,self.device = images,size,device
        self.cache = OrderedDict()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        import torch
        if idx not in self.cache:
            rgb = Image.fromarray(cv2.cvtColor(self.images[idx],cv2.COLOR_BGR2RGB)).resize((self.size,self.size))
            # PIL's default BICUBIC matches official _load_img_as_tensor().
            x = torch.from_numpy(np.asarray(rgb).copy()).permute(2,0,1).float().to(self.device)/255
            mean = x.new_tensor([.485,.456,.406])[:,None,None]
            std = x.new_tensor([.229,.224,.225])[:,None,None]
            self.cache[idx] = (x-mean)/std
            if len(self.cache)>3: self.cache.popitem(last=False)
        return self.cache[idx]


class Engine:
    def __init__(self,root):
        import torch
        from detector import Detector
        from sam2.build_sam import build_sam2_video_predictor
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        self.root,self.cfg = root,json.loads((root/'run_config.json').read_text())
        self.objects = {x['object_id']:x for x in json.loads((root/'config/objects.json').read_text())}
        self.robot_objects = {x['object_id']:x for x in json.loads((root/'config/robot_objects.json').read_text())}
        self.mapping = json.loads((root/'config/task_objects.json').read_text())
        self.appearance=None;self.robot_parts=None
        self.distractors={1000:dict(object_id=1000,class_name='robot arm',detailed_text_prompt='a robot arm'),
            1001:dict(object_id=1001,class_name='robot gripper',detailed_text_prompt='a black robot gripper'),
            1002:dict(object_id=1002,class_name='container',detailed_text_prompt='a plastic container'),
            1003:dict(object_id=1003,class_name='table',detailed_text_prompt='a green table')}
        torch.set_num_threads(4); cv2.setNumThreads(1); torch.manual_seed(self.cfg['seed'])
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.dino = Detector(self.cfg['dino_config'],self.cfg['dino_checkpoint'],self.cfg['bert'],self.objects|self.robot_objects|self.distractors)
        if self.cfg.get('appearance_gallery'):
            from identity_guard import Appearance
            self.appearance=Appearance(root/self.cfg['appearance_gallery'],project=self.cfg['project'])
        if self.cfg.get('robot_reference'):
            from robot_parts import RobotSeeds
            self.robot_parts=RobotSeeds(root/self.cfg['robot_reference'])
        self.sam = build_sam2_video_predictor(self.cfg['sam2_config'],self.cfg['sam2_checkpoint'],device='cuda',
            apply_postprocessing=False,hydra_overrides_extra=['++model.fill_hole_area=0','++model.non_overlap_masks=false'])
        self.image_sam = SAM2ImagePredictor(self.sam)
        import sam2.sam2_video_predictor as predictor_module
        predictor_module.tqdm = lambda iterable,**kwargs: iterable
        self.settings = Settings(detection_threshold=self.cfg['detection_threshold'],ambiguous_margin=self.cfg['ambiguity_margin'])

    def new_state(self,images):
        h,w = images[0].shape[:2]
        adapter = LazyImages(images,self.sam.image_size,self.sam.device)
        # Local adapter only: upstream predictor and raw inputs stay unchanged.
        with patch('sam2.sam2_video_predictor.load_video_frames',return_value=(adapter,h,w)):
            return self.sam.init_state('in_memory',offload_video_to_cpu=False,offload_state_to_cpu=False)

    def detect(self,image,ids):
        import torch
        query=ids+list(self.distractors) if self.appearance and max(ids)<1000 else ids
        with torch.autocast('cuda',enabled=False):
            found=self.dino.detect(image,query,floor=.20)[0]
        if self.appearance and max(ids)<1000:
            return self.appearance.check(image,[d for d in found if d['object_id'] in ids])
        return found

    def robot_seed(self,image,camera):
        """Independent robot union. Generic detections are QA-only, not GT."""
        import torch
        detections = self.detect(image,list(self.robot_objects))
        detections = [d for d in detections if d['confidence']>=.40]
        if self.appearance:
            scored=self.appearance.check(image,[d|dict(object_id=-1,original_object_id=d['object_id']) for d in detections])
            detections=[d|dict(object_id=d['original_object_id']) for d in scored if d['appearance_accepted']]
        selected = []
        for d in sorted(detections,key=lambda x:x['confidence'],reverse=True):
            if all(box_iou(d['bbox_xyxy'],p['bbox_xyxy'])<.65 for p in selected): selected.append(d)
        selected = selected[:4]
        h,w = image.shape[:2]
        self.image_sam.set_image(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        union = np.zeros((h,w),bool)
        details = []
        if selected:
            masks,scores,_ = self.image_sam.predict(box=np.asarray([d['bbox_xyxy'] for d in selected],np.float32),multimask_output=False)
            for m,d in zip(masks.reshape(-1,h,w),selected):
                union |= m.astype(bool)
                details.append(dict(kind='groundingdino_robot_candidate',**d))
        if camera!='head' and self.robot_parts is None:
            # Camera-mounted fingers: same relative prompts visually checked in
            # the existing pilot; union stays separate and is explicitly QA-only.
            for points in [[[.22,.914],[.06,.97]],[[.833,.89],[.953,.97]]]:
                masks,scores,_ = self.image_sam.predict(point_coords=np.asarray(points)*[w,h],
                    point_labels=np.ones(2,np.int32),multimask_output=True)
                chosen = int(np.argmax(scores))
                if masks[chosen].sum()/masks[chosen].size < .40:
                    union |= masks[chosen].astype(bool)
                    details.append(dict(kind='camera_local_finger_points',points_normalized=points,
                                        sam_iou_prediction=float(scores[chosen])))
        self.image_sam.reset_predictor()
        return union,details

    def own_wrist_seed(self,image):
        """Visible camera-mounted fingers, verified against robot exemplars."""
        h,w=image.shape[:2];union=np.zeros((h,w),bool);details=[]
        self.image_sam.set_image(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        for points in [[[.10,.97],[.23,.90]],[[.91,.97],[.80,.90]]]:
            p=np.asarray(points+[[.5,.72]])*[w,h]
            masks,scores,_=self.image_sam.predict(point_coords=p,point_labels=np.array([1,1,0]),multimask_output=True)
            eligible=[i for i,m in enumerate(masks) if 150<m.sum()<h*w*.28 and m[-4:].any()]
            if not eligible:continue
            best=max(eligible,key=lambda i:scores[i]);m=masks[best].astype(bool)
            candidate=dict(object_id=-1,bbox_xyxy=bbox(m),confidence=float(scores[best]))
            scored=self.appearance.check(image,[candidate])[0] if self.appearance else candidate
            if scored.get('appearance_similarity',0)<.55:continue
            scored['appearance_policy']='robot_similarity_and_rigid_camera_anchor; cross_class_margin_not_required'
            union|=m;details.append(scored|dict(points_normalized=points,kind='camera_mounted_finger_points_with_appearance_veto'))
        self.image_sam.reset_predictor()
        return union,details

    def process(self,ep,camera,job):
        import torch
        root,cfg = self.root,self.cfg
        ids = self.mapping[str(ep['task_id'])]
        directory = root/f'task_{ep["task_id"]:02d}'/ep['episode_id']/camera/f'attempt_{job["attempts"]:03d}'
        directory.mkdir(parents=True,exist_ok=False)
        reference_path = root/'reviewed_reference'/f'task_{ep["task_id"]:02d}'/ep['episode_id']/f'{camera}.jsonl'
        references = {}
        if reference_path.exists():
            references = {(r['frame_idx'],r['object_id']):r for r in map(json.loads,reference_path.read_text().splitlines())}
        override_path = root/'overrides'/f'{ep["episode_id"]}.{camera}.json'
        overrides = json.loads(override_path.read_text()) if override_path.exists() else {'seeds':[]}
        write_json(directory/'overrides_snapshot.json',overrides)
        controls = {(r['frame_idx'],r['object_id']):r for r in overrides['seeds']}
        start = time.perf_counter(); processed=0;counts=Counter();review_count=0
        carry = {};previous={};origin={};last_image=None;last_robot=None;verified_robot_sides=set()
        torch.cuda.reset_peak_memory_stats()
        runhash = sha(root/'run_config.json')
        layers=['objects','robot','events','detections','review']+(['robot_parts','robot_candidates'] if self.robot_parts else [])
        output_paths = {k:directory/f'{k}.jsonl.gz' for k in layers}
        samples = []
        with ExitStack() as stack:
            outputs = {k:stack.enter_context(gzip.open(str(path)+'.partial','wt',compresslevel=1)) for k,path in output_paths.items()}
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                for batch,overlap in blocks(episode_frames(ep,camera),cfg['chunk_frames']):
                    images = [b[2] for b in batch];h,w=images[0].shape[:2]
                    indices = {b[0]:i for i,b in enumerate(batch)}
                    keyframes = sorted(set(range(0,len(batch),cfg['detection_period']))|{len(batch)-1})
                    proposals = {};rejects={};raw={}
                    for local in keyframes:
                        detections = self.detect(images[local],ids)
                        chosen,rejected = candidates_for_task(detections,ids,self.settings)
                        raw[local] = detections;proposals[local]=chosen;rejects[local]=rejected
                        jline(outputs['detections'],dict(frame_idx=batch[local][0],candidates=detections,rejected=rejected))
                    state = self.new_state(images)
                    initialized=set();seed_frames=defaultdict(set)
                    protected={oid for (idx,oid) in (controls|references) if idx in indices}
                    for oid,binary in carry.items():
                        self.sam.add_new_mask(state,0,oid,binary)
                        initialized.add(oid);seed_frames[oid].add(0)
                    # Robot is re-localized each bounded segment; never combined
                    # with objects or used to erase their contact surfaces.
                    if self.robot_parts:
                        from robot_parts import PARTS
                        located,details=self.robot_parts.locate(images[0],camera)
                        for side,arm,ee in [('left',1101,1103),('right',1102,1104)]:
                            if any(arm in d.get('part_ids',[]) and ee in d.get('part_ids',[]) for d in details):
                                verified_robot_sides.add(side)
                        if camera!='head':
                            own_id=1103 if camera=='left_wrist' else 1104
                            if not carry.get(own_id,np.zeros((h,w),bool)).any() and not located.get(own_id,np.zeros((h,w),bool)).any():
                                fingers,finger_details=self.own_wrist_seed(images[0])
                                if fingers.any():located[own_id]=fingers;details+=finger_details
                        for oid,m in located.items():
                            # A narrow uncertain flange band is not a physical
                            # SAM object: tracking it would often swallow the arm.
                            if oid==1100:continue
                            # Prefer a continuing anatomical identity over a new
                            # approximate registration in an articulated pose.
                            if m.any() and (oid not in carry or not carry[oid].any()):
                                self.sam.add_new_mask(state,0,oid,m)
                                initialized.add(oid);seed_frames[oid].add(0)
                                origin[oid]=dict(kind='registered_robot_part_seed',frame_idx=batch[0][0],registration=details)
                        if camera!='head' or not all(oid in initialized for oid in [1101,1102,1103,1104]):
                            robot,robot_details=self.robot_seed(images[0],camera)
                            known=np.zeros((h,w),bool)
                            for oid in [1101,1102,1103,1104]:
                                known|=carry.get(oid,located.get(oid,np.zeros((h,w),bool)))
                            # Only separate robot instances not explained by a
                            # named part become anonymous fallback seeds.
                            robot &= ~cv2.dilate(known.astype(np.uint8),np.ones((15,15),np.uint8)).astype(bool)
                            if robot.sum()>250:
                                self.sam.add_new_mask(state,0,1100,robot)
                                initialized.add(1100);seed_frames[1100].add(0)
                                origin[1100]=dict(kind='anatomy_unresolved_robot_candidate',components=robot_details)
                    else:
                        robot,robot_details = self.robot_seed(images[0],camera)
                        if robot.any():
                            self.sam.add_new_mask(state,0,1000,robot)
                            initialized.add(1000);seed_frames[1000].add(0)
                            origin[1000] = dict(kind='robot_union_qa_only',frame_idx=batch[0][0],components=robot_details)
                    for local in keyframes:
                        for oid,detection in proposals[local].items():
                            if oid in protected:continue
                            # Confirm weak isolated hits against a neighbouring
                            # detection; do not turn every high score into truth.
                            support = any(box_iou(detection['bbox_xyxy'],proposals[j].get(oid,{}).get('bbox_xyxy'))>=.15
                                          for j in keyframes if j!=local and abs(j-local)<=cfg['detection_period']+1)
                            if not support and detection['confidence']<.55:
                                rejects[local][oid]='isolated_weak_detection';continue
                            self.sam.add_new_points_or_box(state,local,oid,box=np.asarray(detection['bbox_xyxy'],np.float32))
                            initialized.add(oid);seed_frames[oid].add(local)
                            origin[oid] = dict(kind='automatic_dino_box',frame_idx=batch[local][0],confidence=detection['confidence'],
                                appearance={k:v for k,v in detection.items() if k.startswith('appearance_')})
                            counts['automatic_seeds']+=1
                    # Exact reviewed masks and explicit visual corrections take
                    # priority over automatic boxes, including empty masks.
                    for (idx,oid),r in sorted((controls|references).items()):
                        if idx not in indices or oid not in ids: continue
                        local = indices[idx]
                        if r.get('mask'):
                            binary=decode(r['mask']); assert binary.shape==(h,w)
                            self.sam.add_new_mask(state,local,oid,binary)
                        else:
                            kwargs = {k:np.asarray(r[k]) for k in ['box','points','labels'] if k in r}
                            self.sam.add_new_points_or_box(state,local,oid,**kwargs)
                        initialized.add(oid);seed_frames[oid].add(local)
                        origin[oid] = dict(kind='user_reviewed_mask' if (idx,oid) in references else 'assistant_visual_seed',frame_idx=idx)
                    review_last=defaultdict(lambda:-1000);last_recovery=-1000
                    detector_misses=Counter();terminated_at=set()
                    local=0
                    while local<len(batch):
                        iterator = self.sam.propagate_in_video(state,start_frame_idx=local) if initialized else ((j,[],None) for j in range(local,len(batch)))
                        restarted=False
                        for local,oids,logits in iterator:
                            predicted={oid:m for oid,m in zip(oids,(logits[:,0]>0).cpu().numpy())} if logits is not None else {}
                            presences={}
                            for oi,oid in enumerate(oids):
                                cache=state['output_dict_per_obj'][oi]
                                current=cache['cond_frame_outputs'].get(local,cache['non_cond_frame_outputs'].get(local))
                                presences[oid]=float(current['object_score_logits'].float().sigmoid().reshape(-1)[0])
                            binary_objects={oid:predicted.get(oid,np.zeros((h,w),bool)) for oid in ids}
                            if self.robot_parts:
                                from robot_parts import anatomy_guard
                                part_masks,anatomy_reasons=anatomy_guard({k:predicted[k] for k in PARTS if k in predicted},
                                    camera,verified_robot_sides,(h,w))
                                robot=np.logical_or.reduce(list(part_masks.values()))
                            else:robot=predicted.get(1000,np.zeros((h,w),bool))
                            flags=diagnostics(binary_objects,previous,robot,self.settings)
                            if local in keyframes:
                                for oid in ids:
                                    if oid in proposals[local]: detector_misses[oid]=0
                                    elif (local,oid) not in terminated_at: detector_misses[oid]+=1
                            terminated=[]
                            for oid,m in binary_objects.items():
                                if (local,oid) in terminated_at or (batch[local][0],oid) in references: continue
                                if (detector_misses[oid]>=3 and 'tiny_border_residue' in flags[oid]
                                        and presences.get(oid,1)<.5):
                                    self.sam.add_new_mask(state,local,oid,np.zeros((h,w),bool))
                                    seed_frames[oid].add(local);terminated_at.add((local,oid));terminated.append(oid)
                            if terminated:
                                jline(outputs['events'],dict(frame_idx=batch[local][0],event='low_presence_border_residue_termination',object_ids=terminated))
                                counts['termination_events']+=1;restarted=True;break
                            serious=any(any(x in {'mask_iou_drop','mask_area_jump','bbox_center_jump','tiny_border_residue'}
                                                or x.startswith('object_overlap:') for x in f) for f in flags.values())
                            # Redetect immediately on a new tracking discontinuity,
                            # then restart from the corrected frame exactly once.
                            if serious and local-last_recovery>=cfg['search_cooldown'] and local not in keyframes:
                                last_recovery=local
                                detections=self.detect(images[local],ids)
                                chosen,rejected=candidates_for_task(detections,ids,self.settings)
                                jline(outputs['detections'],dict(frame_idx=batch[local][0],trigger='tracking_anomaly',candidates=detections,rejected=rejected))
                                changes=[]
                                for oid,d in chosen.items():
                                    if oid not in protected and flags[oid] and d['confidence']>=.50:
                                        self.sam.add_new_points_or_box(state,local,oid,box=np.asarray(d['bbox_xyxy'],np.float32))
                                        initialized.add(oid);seed_frames[oid].add(local);changes.append(oid)
                                        origin[oid]=dict(kind='anomaly_redetection',frame_idx=batch[local][0],confidence=d['confidence'])
                                if changes:
                                    counts['recovery_events']+=1
                                    jline(outputs['events'],dict(frame_idx=batch[local][0],event='redetect_reseed',object_ids=changes))
                                    restarted=True;break
                            # An explicit absence mask stops a track. Model-empty
                            # remains predicted absence, not a semantic OOV claim.
                            idx,timestamp,image=batch[local]
                            rows=[]
                            base=dict(episode_id=ep['episode_id'],task_id=ep['task_id'],side=ep['side'],camera=camera,
                                frame_idx=idx,timestamp=timestamp,image_width=w,image_height=h,run_config_sha256=runhash)
                            for oid in ids:
                                m=binary_objects[oid]
                                if (idx,oid) in references:
                                    r=references[idx,oid];m=decode(r['mask']);binary_objects[oid]=m
                                    row=base|r|dict(layer='object',suspicious_flags=flags[oid])
                                else:
                                    known=oid in initialized
                                    row=base|dict(object_id=oid,class_name=self.objects[oid]['class_name'],layer='object',
                                        **record(m),confidence=presences.get(oid),human_confirmed=False,
                                        visibility='predicted_visible' if m.any() else 'unknown',
                                        confidence_semantics='SAM2 presence, not calibrated identity/mask quality',
                                        tracking_status='tracking' if m.any() else 'lost' if known else 'not_initialized',
                                        provenance=origin.get(oid,dict(kind='uninitialized'))|dict(
                                            conditioning_frames=[batch[k][0] for k in sorted(seed_frames[oid])],
                                            segment_start=batch[0][0],offline_future_conditioning=True),
                                        suspicious_flags=flags[oid],track_id=f'{ep["episode_id"]}:{camera}:{oid}')
                                    if not known: row.update(visible=None,mask_status='not_initialized')
                                rows.append(row)
                                if not (overlap and local==0):
                                    counts['visible_rows' if m.any() else 'empty_or_uninitialized_rows']+=1
                            robotrow=base|dict(object_id='robot_mask',class_name='robot',layer='robot',**record(robot),
                                confidence=presences.get(1000),human_confirmed=False,provenance=origin.get(1000,{'kind':'uninitialized'}),
                                suspicious_flags=['robot_mask_not_semantically_verified'],qa_only=True)
                            part_rows=None
                            if self.robot_parts:
                                part_rows=[]
                                for oid,name in PARTS.items():
                                    m=part_masks[oid];known=oid in initialized or (oid==1100 and m.any())
                                    part=base|dict(object_id=name,part_id=oid,class_name=name,layer='robot_part',**record(m),
                                        confidence=presences.get(oid),human_confirmed=False,
                                        visibility='predicted_visible' if m.any() else 'unknown',
                                        provenance=origin.get(oid,{'kind':'uninitialized'}),
                                        anatomical_side='left' if name.startswith('left') else 'right' if name.startswith('right') else None,
                                        flange_boundary_status='approximate_propagated_seed_not_human_GT',
                                        suspicious_flags=['robot_part_not_human_verified']+(['anatomy_unresolved'] if oid==1100 and m.any() else []),
                                        track_id=f'{ep["episode_id"]}:{camera}:{name}',qa_only=True)
                                    if not known:part.update(visible=None,mask_status='not_initialized')
                                    if oid in anatomy_reasons:
                                        part.update(visible=None,mask_status='anatomy_unresolved',
                                            flange_boundary_status='not_observed',
                                            suspicious_flags=part['suspicious_flags']+[anatomy_reasons[oid]])
                                    part_rows.append(part)
                                robotrow.update(confidence=None,provenance={'kind':'union_of_robot_parts_and_unknown'},
                                                confidence_semantics='Union has no calibrated confidence')
                                if not any(k in initialized for k in PARTS):robotrow.update(visible=None,mask_status='not_initialized')
                            elif 1000 not in initialized: robotrow.update(visible=None,mask_status='not_initialized')
                            if not (overlap and local==0):
                                for row in rows: jline(outputs['objects'],row)
                                if part_rows is not None:
                                    for row in part_rows:jline(outputs['robot_parts'],row)
                                    for oid,reason in anatomy_reasons.items():
                                        jline(outputs['robot_candidates'],base|dict(part_id=oid,object_id=PARTS[oid],
                                            class_name=PARTS[oid],layer='robot_part_candidate',**record(predicted[oid]),
                                            anatomical_side='left' if PARTS[oid].startswith('left') else 'right',
                                            confidence=presences.get(oid),human_confirmed=False,
                                            accepted_part_label=False,suspicious_flags=[reason],
                                            provenance=origin.get(oid,{'kind':'uninitialized'})))
                                jline(outputs['robot'],robotrow);processed+=1
                                # Every flag remains in dense output. Only dedupe
                                # the image review queue, never delete masks.
                                issues=[]
                                if anatomy_reasons and idx%30==0:
                                    issues.append(dict(layer='robot',part_ids=list(anatomy_reasons),
                                                       reasons=sorted(set(anatomy_reasons.values()))))
                                for row in rows:
                                    oid=row['object_id']
                                    ff=[f for f in row['suspicious_flags'] if f!='empty_mask']
                                    if local in keyframes and oid in rejects.get(local,{}): ff.append(rejects[local][oid])
                                    if local in keyframes and row.get('tracking_status')=='not_initialized':
                                        ff.append('no_identity_seed')
                                    if ff and local-review_last[oid]>=30:
                                        issues.append(dict(object_id=oid,reasons=ff));review_last[oid]=local
                                if issues:
                                    review_count+=1
                                    jline(outputs['review'],base|dict(issues=issues,status='assistant_review_pending',
                                        original_video=ep['cameras'][camera]['video'],attempt=job['attempts']))
                                if len(samples)<9 and (idx==0 or idx>=len(samples)*max(1,ep['frames']//8)):
                                    preview=directory/f'qa_{idx:06d}.jpg'
                                    cv2.imwrite(str(preview),render(image,rows,robotrow,f'{ep["episode_id"]} {camera} frame {idx}',part_rows))
                                    samples.append(preview.name)
                            previous={oid:m.copy() for oid,m in binary_objects.items()}
                            carry={oid:m.copy() for oid,m in predicted.items()}
                            for oid in ids:
                                if (idx,oid) in references: carry[oid]=binary_objects[oid].copy()
                            last_image=image;last_robot=robotrow
                        if not restarted: break
                    assert local==len(batch)-1
                    queue.heartbeat(root,job,processed)
                    print(json.dumps(dict(event='chunk',episode=ep['episode_id'],camera=camera,processed=processed,
                        total=ep['frames'],seconds=time.perf_counter()-start)),flush=True)
                    del state,images,batch,predicted,logits
                    gc.collect()
            assert processed==ep['frames'],(processed,ep['frames'])
        for path in output_paths.values(): Path(str(path)+'.partial').replace(path)
        result=dict(episode_id=ep['episode_id'],task_id=ep['task_id'],camera=camera,frames=processed,
            object_rows=processed*len(ids),robot_rows=processed,object_ids=ids,review_events=review_count,
            robot_part_rows=processed*5 if self.robot_parts else 0,
            wall_seconds=time.perf_counter()-start,counts=dict(counts),attempt=job['attempts'],
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
            files={k:str(v.relative_to(root)) for k,v in output_paths.items()},
            file_hashes={k:sha(v) for k,v in output_paths.items()},qa_samples=samples,
            status='PROCESSED_AUTOMATIC_DRAFT',semantic_acceptance=False)
        write_json(directory/'complete.json',result)
        write_json(directory.parent/'latest.json',dict(attempt=job['attempts'],complete=str((directory/'complete.json').relative_to(root))))
        return result


def worker(args):
    root=args.output.resolve()
    cfg=json.loads((root/'run_config.json').read_text())
    for relative,expected in cfg['file_hashes'].items():
        if sha(root/relative)!=expected: raise ValueError(f'Frozen code/config changed: {relative}')
    episodes={e['episode_id']:e for e in json.loads((root/'episodes.json').read_text())['episodes']}
    engine=Engine(root)
    import torch
    write_json(root/'logs'/f'worker_{args.gpu}.environment.json',dict(pid=os.getpid(),gpu=args.gpu,
        torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(),started=time.time()))
    processed=0
    while not (root/'STOP').exists():
        job=queue.claim(root,args.gpu)
        if job is None: break
        print(json.dumps(dict(event='start',job=job)),flush=True)
        try:
            result=engine.process(episodes[job['episode']],job['camera'],job)
            queue.finish(root,job,result)
            print(json.dumps(dict(event='complete',**result)),flush=True)
        except Exception:
            error=traceback.format_exc();print(error,flush=True);queue.fail(root,job,error)
            # Recreate the CUDA context after failure; supervisor restarts us.
            raise
        processed+=1
        gc.collect();torch.cuda.empty_cache()
        if args.max_jobs and processed>=args.max_jobs: break


def supervise(args):
    root=args.output.resolve();queue.recover_dead(root)
    processes={};restarts=Counter();handles={}
    launch=time.time()
    try:
        while True:
            current=queue.status(root)
            pending=any(g['status']=='pending' and g['streams'] for g in current['groups'])
            for gpu in args.gpus:
                proc=processes.get(gpu)
                if proc is not None and proc.poll() is None: continue
                if proc is not None:
                    handles[gpu].close();del processes[gpu]
                    if proc.returncode: restarts[gpu]+=1;queue.recover_dead(root)
                if pending and not (root/'STOP').exists() and restarts[gpu]<4:
                    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',
                        OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1',
                        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TQDM_DISABLE='1')
                    handles[gpu]=(root/'logs'/f'gpu{gpu}.log').open('a')
                    processes[gpu]=subprocess.Popen([sys.executable,str(root/'code/full_segmentation.py'),'worker',
                        '--output',str(root),'--gpu',str(gpu)],env=env,stdout=handles[gpu],stderr=subprocess.STDOUT)
            current=queue.status(root);current.update(updated=time.time(),supervisor_started=launch,
                worker_pids={str(g):p.pid for g,p in processes.items() if p.poll() is None},restart_counts=dict(restarts))
            write_json(root/'progress.json',current)
            print(json.dumps({k:current[k] for k in ['groups','completed_episodes','recent_frames','recent_worker_seconds','worker_pids']}),flush=True)
            if not processes:
                write_json(root/'run_finished.json',current|dict(all_streams_processed=all(g['status']=='done' for g in current['groups']),
                    semantic_acceptance=False,quality_review='Separate unresolved assistant review remains'))
                return
            time.sleep(15)
    finally:
        for h in handles.values(): h.close()


def validate(args):
    root=args.output.resolve();results=[]
    with queue.connect(root) as db:
        jobs=db.execute("SELECT result FROM jobs WHERE status='done' ORDER BY id").fetchall()
    for item in jobs[:args.limit or None]:
        result=json.loads(item['result']);ids=result['object_ids'];n=0;robot_count=0
        layers=['objects','robot']+(['robot_parts'] if 'robot_parts' in result['files'] else [])
        part_ids=['left_arm','right_arm','left_ee','right_ee','robot_unknown']
        for layer in layers:
            path=root/result['files'][layer];assert sha(path)==result['file_hashes'][layer]
            layer_count=0
            with gzip.open(path,'rt') as f:
                for i,row in enumerate(map(json.loads,f)):
                    idx=i//len(ids) if layer=='objects' else i//5 if layer=='robot_parts' else i
                    assert row['frame_idx']==idx
                    if layer=='objects': assert row['object_id']==ids[i%len(ids)]
                    if layer=='robot_parts':assert row['object_id']==part_ids[i%5]
                    assert row['episode_id']==result['episode_id'] and row['camera']==result['camera']
                    binary=decode(row['mask']);assert binary.shape==(row['image_height'],row['image_width'])
                    assert bbox(binary)==row['bbox_xyxy'] and int(binary.sum())==row['mask_area']
                    assert row['visible'] is None or row['visible']==bool(binary.any())
                    assert np.isfinite(row['timestamp'])
                    if layer=='objects': n+=1
                    elif layer=='robot': robot_count+=1
                    layer_count+=1
            assert layer_count==result['frames']*(len(ids) if layer=='objects' else 5 if layer=='robot_parts' else 1)
        results.append(dict(episode=result['episode_id'],camera=result['camera'],frames=result['frames'],rows=n))
    report=dict(status='PASS' if results else 'NO_COMPLETED_STREAMS',streams=len(results),results=results,
                semantic_accuracy='Not evaluated by structural validator')
    write_json(root/'validation_latest.json',report);print(json.dumps(report),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','freeze','worker','supervise','status','requeue','validate'])
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--input',type=Path,default=PROJECT/'datasets/lerobot/astribot_full_v21_rgb_h264')
    p.add_argument('--gpu',default='0');p.add_argument('--gpus',nargs='+',type=int,default=list(range(10)))
    p.add_argument('--max-jobs',type=int,default=0);p.add_argument('--limit',type=int,default=0)
    p.add_argument('--episode');p.add_argument('--camera',choices=list(CAMERAS))
    p.add_argument('--appearance-gallery',type=Path);p.add_argument('--robot-reference',type=Path)
    a=p.parse_args()
    if a.command=='prepare': prepare(a)
    elif a.command=='freeze':
        with queue.connect(a.output) as db:
            assert db.execute('SELECT sum(attempts) FROM jobs').fetchone()[0]==0,'Cannot change a started run'
        for path in BASE.glob('*.py'): shutil.copy2(path,a.output/'code'/path.name)
        cfg=json.loads((a.output/'run_config.json').read_text())
        cfg['file_hashes']={str(path.relative_to(a.output)):sha(path)
            for folder in ['code','config','identity_gallery','robot_reference']
            for path in sorted((a.output/folder).rglob('*')) if path.is_file()}
        write_json(a.output/'run_config.json',cfg)
    elif a.command=='worker': worker(a)
    elif a.command=='supervise': supervise(a)
    elif a.command=='validate': validate(a)
    elif a.command=='status': print(json.dumps(queue.status(a.output),indent=2))
    elif a.command=='requeue':
        assert a.episode and a.camera
        with queue.connect(a.output) as db:
            changed=db.execute("UPDATE jobs SET status='pending',priority=-1,processed=0 WHERE episode=? AND camera=? AND status!='running'",
                               (a.episode,a.camera)).rowcount
            assert changed==1,'No eligible completed/failed stream; cannot requeue a running lease'


if __name__=='__main__':
    main()
