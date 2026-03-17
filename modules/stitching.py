import numpy as np
import cv2


def _cum_affine(transforms):
    """Build cumulative 2x3 affine matrices from pairwise transforms.

    Each transform is {tx, ty, scale, rotation (degrees, optional)}.
    Returns list of 3x3 matrices (one per image) in homogeneous coords.
    """
    # Identity for image 0
    cum = [np.eye(3)]
    for t in transforms:
        s = t['scale']
        rot = np.radians(t.get('rotation', 0.0))
        c, sn = np.cos(rot), np.sin(rot)
        # Pairwise: maps img2 pixel to img1 coord system
        M = np.array([[s * c, -s * sn, t['tx']],
                      [s * sn,  s * c,  t['ty']],
                      [0,       0,      1]])
        cum.append(cum[-1] @ M)
    return cum


def stitch_trans_scale(images, transforms):
    MAX_SIZE = 20000

    has_rotation = any(t.get('rotation', 0.0) != 0.0 for t in transforms)

    if not has_rotation:
        # Fast path: original scale+translate logic
        cum = [{'tx': 0, 'ty': 0, 'scale': 1.0}]
        for t in transforms:
            prev = cum[-1]
            cum.append({'tx': prev['tx'] + t['tx'] * prev['scale'],
                        'ty': prev['ty'] + t['ty'] * prev['scale'],
                        'scale': prev['scale'] * t['scale']})
        min_x, min_y, max_x, max_y = 0, 0, 0, 0
        for i, img in enumerate(images):
            h, w = img.shape[:2]
            s, tx, ty = cum[i]['scale'], cum[i]['tx'], cum[i]['ty']
            min_x, min_y = min(min_x, tx), min(min_y, ty)
            max_x, max_y = max(max_x, tx + w * s), max(max_y, ty + h * s)
        canvas_w = min(int(max_x - min_x) + 1, MAX_SIZE)
        canvas_h = min(int(max_y - min_y) + 1, MAX_SIZE)
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        for i, img in enumerate(images):
            h, w = img.shape[:2]
            s = cum[i]['scale']
            tx, ty = int(cum[i]['tx'] - min_x), int(cum[i]['ty'] - min_y)
            new_w, new_h = int(w * s), int(h * s)
            if new_w > 0 and new_h > 0:
                img_scaled = cv2.resize(img, (new_w, new_h))
                x1, y1 = max(0, tx), max(0, ty)
                x2, y2 = min(canvas_w, tx + new_w), min(canvas_h, ty + new_h)
                if x2 > x1 and y2 > y1:
                    canvas[y1:y2, x1:x2] = img_scaled[y1-ty:y2-ty, x1-tx:x2-tx]
        return canvas

    # Affine path: handles rotation + scale + translation
    cum = _cum_affine(transforms)

    # Compute canvas bounds from all image corners
    min_x, min_y = 0.0, 0.0
    max_x, max_y = 0.0, 0.0
    for i, img in enumerate(images):
        h, w = img.shape[:2]
        corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], dtype=np.float64)
        corners_t = (cum[i] @ corners.T).T
        min_x = min(min_x, corners_t[:, 0].min())
        min_y = min(min_y, corners_t[:, 1].min())
        max_x = max(max_x, corners_t[:, 0].max())
        max_y = max(max_y, corners_t[:, 1].max())

    canvas_w = min(int(np.ceil(max_x - min_x)) + 1, MAX_SIZE)
    canvas_h = min(int(np.ceil(max_y - min_y)) + 1, MAX_SIZE)
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    for i, img in enumerate(images):
        # Build 2x3 affine with canvas offset
        M = cum[i][:2, :].copy()
        M[0, 2] -= min_x
        M[1, 2] -= min_y
        warped = cv2.warpAffine(img, M, (canvas_w, canvas_h))
        mask = np.any(warped > 0, axis=2)
        canvas[mask] = warped[mask]

    return canvas
