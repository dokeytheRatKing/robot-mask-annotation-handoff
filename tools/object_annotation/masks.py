"""COCO compressed RLE and transparent, lossless mask-derived geometry."""
import cv2
import numpy as np
from pycocotools import mask as coco


def encode(binary):
    binary=np.asarray(binary,dtype=np.uint8)
    assert binary.ndim==2 and ((binary==0)|(binary==1)).all()
    result=coco.encode(np.asfortranarray(binary))
    return dict(format='coco_rle',size=[int(x) for x in result['size']],counts=result['counts'].decode('ascii'))


def decode(rle):
    assert rle['format']=='coco_rle'
    return coco.decode(dict(size=rle['size'],counts=rle['counts'].encode('ascii'))).astype(bool)


def bbox(binary):
    y,x=np.where(binary)
    return [int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)] if len(x) else None


def record(binary,**extra):
    # visible is explicitly model-predicted positive mask existence, not verified visibility.
    return dict(mask=encode(binary),bbox_xyxy=bbox(binary),mask_area=int(binary.sum()),
                visible=bool(binary.any()),mask_status='predicted' if binary.any() else 'predicted_empty',**extra)


def overlay(image,objects,robot,title,mode='combined'):
    out=image.copy();h,w=image.shape[:2]
    union=np.zeros((h,w),bool)
    if robot['mask'] is not None:union=decode(robot['mask'])
    out[union]=(out[union]*.55+np.array([255,210,0])*.45).astype(np.uint8)
    contour,_=cv2.findContours(union.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out,contour,-1,(255,230,0),1)
    object_union=np.zeros_like(union)
    if mode=='combined':
        for row in objects:
            if row['mask'] is None or not row['visible']:continue
            binary=decode(row['mask']);object_union|=binary
            color=np.random.default_rng(row['object_id']+7).integers(65,245,3).astype(np.uint8)
            out[binary]=(out[binary]*.5+color*.5).astype(np.uint8)
            x0,y0,x1,y1=row['bbox_xyxy'];c=tuple(int(x) for x in color)
            cv2.rectangle(out,(x0,y0),(x1,y1),c,2)
            cv2.putText(out,f"{row['class_name']} {row['confidence']:.2f}",(x0,max(15,y0-4)),cv2.FONT_HERSHEY_SIMPLEX,.5,c,1)
        # Highlight rather than erase object/robot disagreements.
        out[object_union&union]=[255,0,255]
    if robot['bbox_xyxy']:
        x0,y0,x1,y1=robot['bbox_xyxy'];cv2.rectangle(out,(x0,y0),(x1,y1),(255,210,0),1)
    canvas=np.concatenate([cv2.resize(image,(640,360)),cv2.resize(out,(640,360))],axis=1)
    canvas=np.pad(canvas,((32,0),(0,0),(0,0)))
    score=robot['confidence']
    text=f'{title} | robot={score:.2f} cyan | overlap magenta' if score is not None else title
    cv2.putText(canvas,text,(6,21),cv2.FONT_HERSHEY_SIMPLEX,.47,(255,255,255),1)
    return canvas
