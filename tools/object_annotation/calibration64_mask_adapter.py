"""Model-agnostic mask pooling with explicit temporal support; never assumes VAE stride.

Run geometric crop/resize/letterbox exactly as RGB BEFORE this function. This
utility accepts the resulting ROI and known masks plus actual native VAE frame
support lists. It does not infer token groups or select action chunks.
"""
import cv2
import numpy as np


def pool_mask(roi,known,temporal_support,spatial_shape):
    roi=np.asarray(roi);known=np.asarray(known)
    if roi.shape!=known.shape or roi.ndim!=3:raise ValueError('Expected matching N,H,W arrays')
    if roi.dtype!=bool or known.dtype!=bool:raise ValueError('Boolean ROI/known required')
    h,w=map(int,spatial_shape)
    if h<=0 or w<=0:raise ValueError('Invalid latent spatial shape')
    foreground=[];support=[]
    for indices in temporal_support:
        ids=list(indices)
        if not ids or len(ids)!=len(set(ids)) or min(ids)<0 or max(ids)>=len(roi):
            raise ValueError('Each temporal support must explicitly contain distinct valid source frame IDs')
        # Normalize over all support pixels, not just known pixels: unknown m=0.
        m=(roi[ids]&known[ids]).astype('float32').mean(0)
        k=known[ids].astype('float32').mean(0)
        foreground.append(cv2.resize(m,(w,h),interpolation=cv2.INTER_AREA))
        support.append(cv2.resize(k,(w,h),interpolation=cv2.INTER_AREA))
    return np.stack(foreground),np.stack(support)
