"""Official GroundingDINO model, detailed prompts, explicit phrase-token scoring."""
from pathlib import Path
import time
import warnings

import cv2
from PIL import Image
import torch
from torchvision.ops import box_convert, nms

from groundingdino.datasets import transforms as T
from groundingdino.models import build_model
from groundingdino.util.misc import clean_state_dict
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.vl_utils import create_positive_map_from_span


class Detector:
    def __init__(self,config,checkpoint,bert,objects,device='cuda',size=800):
        args=SLConfig.fromfile(config);args.device=device
        # Activation recomputation is a training memory tradeoff, unnecessary for inference.
        args.use_checkpoint=False;args.use_transformer_ckpt=False
        warnings.filterwarnings('ignore',message='The `device` argument is deprecated',category=FutureWarning)
        warnings.filterwarnings('ignore',message='`torch.cuda.amp.autocast',category=FutureWarning)
        if bert: args.text_encoder_type=str(Path(bert).resolve())
        self.model=build_model(args)
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        incompatible=self.model.load_state_dict(clean_state_dict(state['model']),strict=False)
        # The official checkpoint carries training-only label embeddings.
        if incompatible.missing_keys: raise ValueError(f'Missing weights: {incompatible.missing_keys}')
        print('Checkpoint unexpected keys:',incompatible.unexpected_keys,flush=True)
        self.model=self.model.eval().to(device); self.device=device
        self.transform=T.Compose([T.RandomResize([size],max_size=1333),T.ToTensor(),
                                  T.Normalize([.485,.456,.406],[.229,.224,.225])])
        self.objects=objects;self.prompts={}

    def prompt(self,ids):
        key=tuple(ids)
        if key not in self.prompts:
            text='';spans=[]
            for oid in ids:
                obj=self.objects[oid];desc=obj['detailed_text_prompt'].lower().strip().rstrip('.')
                phrase=obj.get('grounding_phrase',obj['class_name']).lower()
                start=len(text)+desc.index(phrase)
                spans.append([[start,start+len(phrase)]])
                text+=desc+' . '
            encoded=self.model.tokenizer(text)
            if len(encoded['input_ids'])>256: raise ValueError('Prompt exceeds model token limit')
            positive=create_positive_map_from_span(encoded,spans,max_text_len=256)
            assert (positive.sum(1)>.99).all(), 'Unmapped object token span'
            self.prompts[key]=(text,positive)
        return self.prompts[key]

    @torch.inference_mode()
    def detect(self,bgr,ids,floor=.15,nms_iou=.5):
        caption,positive=self.prompt(ids)
        image,_=self.transform(Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)),None)
        torch.cuda.synchronize();start=time.perf_counter()
        outputs=self.model(image.to(self.device)[None],captions=[caption])
        torch.cuda.synchronize();seconds=time.perf_counter()-start
        logits=outputs['pred_logits'][0].sigmoid().cpu()
        scores=(positive@logits.T).T  # official token-span mean similarity
        conf,classes=scores.max(1)  # one semantic class per model query
        keep=conf>=floor
        boxes=box_convert(outputs['pred_boxes'][0].cpu()[keep],'cxcywh','xyxy')
        h,w=bgr.shape[:2];boxes*=torch.tensor([w,h,w,h])
        boxes[:,0::2].clamp_(0,w);boxes[:,1::2].clamp_(0,h)
        conf=conf[keep];classes=classes[keep];result=[]
        for k,oid in enumerate(ids):
            selected=torch.where((classes==k)&(boxes[:,2]>boxes[:,0])&(boxes[:,3]>boxes[:,1]))[0]
            chosen=selected[nms(boxes[selected],conf[selected],nms_iou)]
            for i in chosen.tolist():
                result.append(dict(object_id=oid,class_name=self.objects[oid]['class_name'],
                                   bbox_xyxy=boxes[i].tolist(),confidence=float(conf[i])))
        return result,seconds
