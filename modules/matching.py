import numpy as np
import cv2
import torch
from kornia.feature import LoFTR


_loftr = None
_device = None


def load_loftr(device=None):
    global _loftr, _device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _device = device
    _loftr = LoFTR(pretrained='outdoor').to(device).eval()
    print("LoFTR loaded")


def match_loftr(img1, img2):
    gray1 = torch.from_numpy(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)).float()[None, None].to(_device) / 255.0
    gray2 = torch.from_numpy(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)).float()[None, None].to(_device) / 255.0
    with torch.no_grad(), torch.cuda.amp.autocast():
        result = _loftr({'image0': gray1, 'image1': gray2})
    return result['keypoints0'].cpu().numpy(), result['keypoints1'].cpu().numpy(), result['confidence'].cpu().numpy()


def filter_by_mask(pts1, pts2, mask1, mask2, conf=None):
    valid = []
    for i, (p1, p2) in enumerate(zip(pts1, pts2)):
        x1, y1, x2, y2 = int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1])
        if (0 <= y1 < mask1.shape[0] and 0 <= x1 < mask1.shape[1] and
            0 <= y2 < mask2.shape[0] and 0 <= x2 < mask2.shape[1] and
            mask1[y1, x1] > 0 and mask2[y2, x2] > 0):
            valid.append(i)
    if not valid:
        empty = pts1[:0]
        return (empty, empty, conf[:0]) if conf is not None else (empty, empty)
    if conf is not None:
        return pts1[valid], pts2[valid], conf[valid]
    return pts1[valid], pts2[valid]


def ransac_filter(pts1, pts2, thresh=5.0):
    if len(pts1) < 4: return np.ones(len(pts1), dtype=bool)
    _, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, thresh)
    return mask.ravel() == 1 if mask is not None else np.ones(len(pts1), dtype=bool)
