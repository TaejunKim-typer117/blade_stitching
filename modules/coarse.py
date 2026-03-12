import math

import numpy as np
import cv2


def radians(degrees):
    return degrees * math.pi / 180


def calc_dcm_matrices(euler321):
    roll, pitch, yaw = euler321
    dcm_yaw = np.array([
        [math.cos(radians(yaw)), math.sin(radians(yaw)), 0],
        [-math.sin(radians(yaw)), math.cos(radians(yaw)), 0],
        [0, 0, 1]
    ])
    dcm_pitch = np.array([
        [math.cos(radians(pitch)), 0, -math.sin(radians(pitch))],
        [0, 1, 0],
        [math.sin(radians(pitch)), 0, math.cos(radians(pitch))]
    ])
    dcm_roll = np.array([
        [1, 0, 0],
        [0, math.cos(radians(roll)), math.sin(radians(roll))],
        [0, -math.sin(radians(roll)), math.cos(radians(roll))]
    ])
    return dcm_roll, dcm_pitch, dcm_yaw


def calc_dcm321(euler321):
    dcm_roll, dcm_pitch, dcm_yaw = calc_dcm_matrices(euler321)
    return dcm_roll @ dcm_pitch @ dcm_yaw


def calc_dcm312(euler321):
    dcm_roll, dcm_pitch, dcm_yaw = calc_dcm_matrices(euler321)
    return dcm_pitch @ dcm_roll @ dcm_yaw


class CoarseStitcher:
    """Computes coarse image positions using GPS/gimbal metadata and DCM transformations."""

    def __init__(self, photos_by_id, section_photo_ids, img_width=720, img_height=480):
        self.photos_by_id = photos_by_id
        self.section_photo_ids = section_photo_ids
        self.img_width = img_width
        self.img_height = img_height

        sensor_width = 35.9
        focal_length = 50.0
        self.angle_width = math.atan2(sensor_width, focal_length * 2)
        self.angle_height = math.atan2(sensor_width * img_height / img_width, focal_length * 2)

        first_meta = photos_by_id[section_photo_ids[0]]['metadata']
        drone = first_meta.get('drone', '')
        self.is_n1 = drone.startswith('NWP2')
        self.is_mrc = drone.startswith('Mavic 3')

    def _calc_corner_points_in_gimbal(self, lidar_distance):
        tan_w = math.tan(self.angle_width)
        tan_h = math.tan(self.angle_height)
        TL = np.array([1, -tan_w, -tan_h]) * lidar_distance
        TR = np.array([1, tan_w, -tan_h]) * lidar_distance
        BL = np.array([1, -tan_w, tan_h]) * lidar_distance
        BR = np.array([1, tan_w, tan_h]) * lidar_distance
        return np.array([TL, TR, BL, BR]).T

    def _calc_corner_points_in_pixel(self, corner_points):
        corners = corner_points.T
        TL, TR, BL, BR = corners
        top_width = np.linalg.norm(TR[1:] - TL[1:])
        bottom_width = np.linalg.norm(BR[1:] - BL[1:])
        left_height = np.linalg.norm(BL[1:] - TL[1:])
        right_height = np.linalg.norm(BR[1:] - TR[1:])
        rate_pm_width = self.img_width / ((top_width + bottom_width) / 2)
        rate_pm_height = self.img_height / ((left_height + right_height) / 2)
        TL_px = [int(rate_pm_width * TL[1]), int(rate_pm_height * TL[2])]
        TR_px = [int(rate_pm_width * TR[1]), int(rate_pm_height * TR[2])]
        BL_px = [int(rate_pm_width * BL[1]), int(rate_pm_height * BL[2])]
        BR_px = [int(rate_pm_width * BR[1]), int(rate_pm_height * BR[2])]
        return np.array([TL_px, TR_px, BL_px, BR_px]).T

    def calc_relative_position(self, curr_photo_id, next_photo_id):
        curr_meta = self.photos_by_id[curr_photo_id]['metadata']
        next_meta = self.photos_by_id[next_photo_id]['metadata']

        alt_correction = -1 if (self.is_mrc or self.is_n1) else 1
        curr_alt = alt_correction * curr_meta.get('alt', 0)
        next_alt = alt_correction * next_meta.get('alt', 0)

        curr_body_pos_ned = np.array([[curr_meta['n'], curr_meta['e'], curr_alt]]).T
        dcm_ned_to_curr_head = calc_dcm321((0, 0, curr_meta.get('body_yaw', 0)))
        dcm_curr_head_to_curr_gimbal = calc_dcm312((
            curr_meta.get('gimbal_roll', 0),
            curr_meta.get('gimbal_pitch', 0),
            curr_meta.get('gimbal_yaw', 0)
        ))

        next_body_pos_ned = np.array([[next_meta['n'], next_meta['e'], next_alt]]).T
        dcm_ned_to_next_head = calc_dcm321((0, 0, next_meta.get('body_yaw', 0)))
        dcm_next_head_to_next_gimbal = calc_dcm312((
            next_meta.get('gimbal_roll', 0),
            next_meta.get('gimbal_pitch', 0),
            next_meta.get('gimbal_yaw', 0)
        ))

        diff_body_pos_ned = next_body_pos_ned - curr_body_pos_ned
        dcm_curr_gimbal_to_next_gimbal = (
            dcm_next_head_to_next_gimbal @
            dcm_ned_to_next_head @
            dcm_ned_to_curr_head.T @
            dcm_curr_head_to_curr_gimbal.T
        )

        curr_lidar = curr_meta.get('measured_distance_to_blade', 7.0)
        next_lidar = next_meta.get('measured_distance_to_blade', 7.0)

        curr_corners_gimbal = self._calc_corner_points_in_gimbal(curr_lidar)
        next_corners_gimbal = self._calc_corner_points_in_gimbal(next_lidar)

        diff_body_pos_next_gimbal = (
            dcm_next_head_to_next_gimbal @ dcm_ned_to_next_head @ diff_body_pos_ned
        )
        diff_broadcasted = np.tile(diff_body_pos_next_gimbal, (1, 4))

        next_corners_curr_gimbal = dcm_curr_gimbal_to_next_gimbal.T @ (
            next_corners_gimbal + diff_broadcasted
        )

        next_pos_pixel = self._calc_corner_points_in_pixel(next_corners_curr_gimbal)
        next_corners = next_pos_pixel.T
        return next_corners[0]

    def compute_all_positions(self):
        positions = {}
        positions[self.section_photo_ids[0]] = (0, 0)

        for i in range(len(self.section_photo_ids) - 1):
            curr_id = self.section_photo_ids[i]
            next_id = self.section_photo_ids[i + 1]
            rel_pos = self.calc_relative_position(curr_id, next_id)
            curr_pos = positions[curr_id]
            curr_center_x = curr_pos[0] + self.img_width / 2
            curr_center_y = curr_pos[1] + self.img_height / 2
            next_x = curr_center_x + rel_pos[0]
            next_y = curr_center_y + rel_pos[1]
            positions[next_id] = (next_x, next_y)

        return positions


def compute_coarse_transforms(photos_by_id, section_photo_ids, img_width=720, img_height=480):
    stitcher = CoarseStitcher(photos_by_id, section_photo_ids, img_width, img_height)
    positions = stitcher.compute_all_positions()

    transforms = []
    for i in range(len(section_photo_ids) - 1):
        curr_id = section_photo_ids[i]
        next_id = section_photo_ids[i + 1]

        curr_pos = positions[curr_id]
        next_pos = positions[next_id]

        tx = next_pos[0] - curr_pos[0]
        ty = next_pos[1] - curr_pos[1]

        curr_lidar = photos_by_id[curr_id]['metadata']['measured_distance_to_blade']
        next_lidar = photos_by_id[next_id]['metadata']['measured_distance_to_blade']
        scale = next_lidar / curr_lidar

        transforms.append({'tx': tx, 'ty': ty, 'scale': scale})

    return transforms


def compute_fine_transform(pts1, pts2, scale):
    if len(pts1) < 1:
        return None
    trans = np.median(pts1 - pts2 * scale, axis=0)
    return {'tx': trans[0], 'ty': trans[1], 'scale': scale}


def compute_iou_in_bbox_for_transform(mask1, mask2, transform):
    """Compute IoU of masks restricted to the overlapping bounding box."""
    h1, w1 = mask1.shape[:2]
    h2, w2 = mask2.shape[:2]

    tx, ty, scale = transform['tx'], transform['ty'], transform['scale']
    new_w2, new_h2 = int(w2 * scale), int(h2 * scale)

    if new_w2 <= 0 or new_h2 <= 0:
        return 0.0

    mask2_scaled = cv2.resize(mask2.astype(np.uint8), (new_w2, new_h2), interpolation=cv2.INTER_NEAREST)

    min_x = min(0, tx)
    min_y = min(0, ty)
    max_x = max(w1, tx + new_w2)
    max_y = max(h1, ty + new_h2)

    canvas_w = int(max_x - min_x)
    canvas_h = int(max_y - min_y)

    x1_offset = int(-min_x)
    y1_offset = int(-min_y)
    x2_offset = int(tx - min_x)
    y2_offset = int(ty - min_y)

    mask1_region = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    mask1_region[y1_offset:y1_offset+h1, x1_offset:x1_offset+w1] = mask1

    mask2_region = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    y2_end = min(y2_offset + new_h2, canvas_h)
    x2_end = min(x2_offset + new_w2, canvas_w)
    y2_start = max(0, y2_offset)
    x2_start = max(0, x2_offset)

    if y2_end > y2_start and x2_end > x2_start:
        src_y_start = y2_start - y2_offset
        src_x_start = x2_start - x2_offset
        src_y_end = src_y_start + (y2_end - y2_start)
        src_x_end = src_x_start + (x2_end - x2_start)
        mask2_region[y2_start:y2_end, x2_start:x2_end] = mask2_scaled[src_y_start:src_y_end, src_x_start:src_x_end]

    bbox_x1 = max(x1_offset, x2_offset)
    bbox_y1 = max(y1_offset, y2_offset)
    bbox_x2 = min(x1_offset + w1, x2_offset + new_w2)
    bbox_y2 = min(y1_offset + h1, y2_offset + new_h2)

    if bbox_x2 > bbox_x1 and bbox_y2 > bbox_y1:
        mask1_crop = mask1_region[bbox_y1:bbox_y2, bbox_x1:bbox_x2]
        mask2_crop = mask2_region[bbox_y1:bbox_y2, bbox_x1:bbox_x2]

        intersection = np.logical_and(mask1_crop > 0, mask2_crop > 0).sum()
        union = np.logical_or(mask1_crop > 0, mask2_crop > 0).sum()
        return intersection / union if union > 0 else 0.0
    return 0.0


def compute_transforms(filtered_results, coarse_transforms, distances, masks=None, mode='fallback',
                       threshold_low=0.2, threshold_high=6.0, angle_threshold=90, iou_threshold=0.4):
    """
    Compute transforms for stitching.

    Fallback policy (use COARSE if any):
        1. No keypoint matches
        2. Angle between fine and coarse directions > 90 degrees
        3. d_f < 0.2 * d_c
        4. d_f > 6.0 * d_c
        5. IoU < 0.4 (mask overlap in bbox)
    """
    transforms = []
    decisions = []

    for i, r in enumerate(filtered_results):
        pts1, pts2 = r['pts1'], r['pts2']
        scale = distances[i + 1] / distances[i]
        coarse_t = coarse_transforms[i]

        if mode == 'coarse':
            print(f"  Pair {i}: COARSE (mode=coarse)")
            transforms.append(coarse_t)
            decisions.append('coarse')
            continue

        fine_t = compute_fine_transform(pts1, pts2, scale)

        if mode == 'fine':
            if fine_t is None:
                print(f"  Pair {i}: No matches, using COARSE as fallback")
                transforms.append(coarse_t)
                decisions.append('coarse')
            else:
                d_f = np.sqrt(fine_t['tx']**2 + fine_t['ty']**2)
                print(f"  Pair {i}: FINE (d={d_f:.1f})")
                transforms.append(fine_t)
                decisions.append('fine')
            continue

        # mode == 'fallback'
        d_c = np.sqrt(coarse_t['tx']**2 + coarse_t['ty']**2)

        # Check 1: No matches
        if fine_t is None:
            print(f"  Pair {i}: No matches -> COARSE")
            transforms.append(coarse_t)
            decisions.append('coarse')
            continue

        d_f = np.sqrt(fine_t['tx']**2 + fine_t['ty']**2)

        # Check 2: Angle between directions
        if d_c > 0 and d_f > 0:
            dot_product = coarse_t['tx'] * fine_t['tx'] + coarse_t['ty'] * fine_t['ty']
            cos_angle = dot_product / (d_c * d_f)
            cos_angle = np.clip(cos_angle, -1, 1)
            angle_deg = np.degrees(np.arccos(cos_angle))
        else:
            angle_deg = 0

        # Check 3 & 4: Distance ratio
        if d_c > 0:
            ratio = d_f / d_c
            ratio_str = f"{ratio:.2f}"
        else:
            ratio = float('inf')
            ratio_str = "inf"

        # Check 5: IoU threshold
        iou = None
        if masks is not None:
            iou = compute_iou_in_bbox_for_transform(masks[i], masks[i+1], fine_t)

        # Apply fallback rules
        use_coarse = False
        reason = ""

        if angle_deg > angle_threshold:
            use_coarse = True
            reason = f"angle={angle_deg:.1f}\u00b0"
        elif d_c > 0 and d_f < threshold_low * d_c:
            use_coarse = True
            reason = f"ratio={ratio_str}<{threshold_low}"
        elif d_c > 0 and d_f > threshold_high * d_c:
            use_coarse = True
            reason = f"ratio={ratio_str}>{threshold_high}"
        elif iou is not None and iou < iou_threshold:
            use_coarse = True
            reason = f"IoU={iou:.2f}<{iou_threshold}"

        iou_str = f", IoU={iou:.2f}" if iou is not None else ""
        if use_coarse:
            print(f"  Pair {i}: d_f={d_f:.1f}, d_c={d_c:.1f}, angle={angle_deg:.1f}\u00b0{iou_str} ({reason}) -> COARSE")
            transforms.append(coarse_t)
            decisions.append('coarse')
        else:
            print(f"  Pair {i}: d_f={d_f:.1f}, d_c={d_c:.1f}, angle={angle_deg:.1f}\u00b0{iou_str} (ratio={ratio_str}) -> FINE")
            transforms.append(fine_t)
            decisions.append('fine')

    return transforms, decisions
