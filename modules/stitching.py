import numpy as np
import cv2


def stitch_trans_scale(images, transforms):
    MAX_SIZE = 20000
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
    canvas_w, canvas_h = min(int(max_x - min_x) + 1, MAX_SIZE), min(int(max_y - min_y) + 1, MAX_SIZE)
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
