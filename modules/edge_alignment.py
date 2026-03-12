import numpy as np
import cv2


def detect_lines_from_mask(mask, min_length=30):
    edges = cv2.Canny(mask * 255, 50, 150)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=30, minLineLength=min_length, maxLineGap=10)
    return lines.reshape(-1, 4) if lines is not None else []


def compute_line_angle(line):
    x1, y1, x2, y2 = line
    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
    return angle + 180 if angle < 0 else angle


def compute_line_length(line):
    x1, y1, x2, y2 = line
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def angle_distance(a1, a2):
    diff = abs(a1 - a2)
    return min(diff, 180 - diff)


def compute_group_mean_angle(angles):
    rads = [np.radians(a * 2) for a in angles]
    mean_rad = np.arctan2(np.mean([np.sin(r) for r in rads]), np.mean([np.cos(r) for r in rads]))
    mean_angle = np.degrees(mean_rad) / 2
    return mean_angle + 180 if mean_angle < 0 else mean_angle


def group_lines_by_angle(lines, angle_tolerance=15):
    if len(lines) == 0: return []
    line_data = [{'line': l, 'angle': compute_line_angle(l), 'length': compute_line_length(l), 'assigned': False} for l in lines]
    line_data.sort(key=lambda x: x['angle'])
    groups = []
    while True:
        seed_idx = next((i for i, item in enumerate(line_data) if not item['assigned']), None)
        if seed_idx is None: break
        current_group = [line_data[seed_idx]]
        line_data[seed_idx]['assigned'] = True
        changed = True
        while changed:
            changed = False
            group_mean = compute_group_mean_angle([item['angle'] for item in current_group])
            for item in line_data:
                if not item['assigned'] and angle_distance(item['angle'], group_mean) <= angle_tolerance:
                    current_group.append(item)
                    item['assigned'] = True
                    changed = True
        groups.append([(item['line'], item['angle'], item['length']) for item in current_group])
    return groups


def select_best_group(groups):
    if not groups: return [], -1
    best_idx = max(range(len(groups)), key=lambda i: sum(item[2] for item in groups[i]))
    return groups[best_idx], best_idx


def compute_line_intercept(line, reference_angle):
    x1, y1, x2, y2 = line
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    perp_angle_rad = np.radians(reference_angle + 90)
    return mx * np.cos(perp_angle_rad) + my * np.sin(perp_angle_rad)


def group_lines_by_position(lines_with_data):
    if len(lines_with_data) <= 1: return lines_with_data, []
    mean_angle = compute_group_mean_angle([item[1] for item in lines_with_data])
    line_intercepts = sorted([(item, compute_line_intercept(item[0], mean_angle)) for item in lines_with_data], key=lambda x: x[1])
    intercepts = [x[1] for x in line_intercepts]
    max_gap, split_idx = 0, 0
    for i in range(len(intercepts) - 1):
        gap = intercepts[i + 1] - intercepts[i]
        if gap > max_gap: max_gap, split_idx = gap, i + 1
    if (intercepts[-1] - intercepts[0]) > 0 and max_gap < (intercepts[-1] - intercepts[0]) * 0.3:
        return [x[0] for x in line_intercepts], []
    return [x[0] for x in line_intercepts[:split_idx]], [x[0] for x in line_intercepts[split_idx:]]


def project_point_onto_line(point, line_point1, line_point2):
    px, py = point
    x1, y1 = line_point1
    dx, dy = line_point2[0] - x1, line_point2[1] - y1
    line_len = np.sqrt(dx**2 + dy**2)
    if line_len == 0: return 0
    return ((px - x1) * dx + (py - y1) * dy) / line_len


def select_line_from_edge_group(edge_lines, center1, center2):
    if not edge_lines: return None
    if len(edge_lines) == 1: return edge_lines[0][0]
    min_proj, selected = float('inf'), None
    for item in edge_lines:
        line = item[0]
        proj = min(project_point_onto_line((line[0], line[1]), center1, center2),
                   project_point_onto_line((line[2], line[3]), center1, center2))
        if proj < min_proj: min_proj, selected = proj, line
    return selected


def line_intersection(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10: return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))


def clip_line_to_bbox(line, bbox):
    x1, y1, x2, y2 = line
    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox
    dx, dy = x2 - x1, y2 - y1
    if abs(dx) < 1e-10 and abs(dy) < 1e-10: return None
    t_values = []
    if abs(dx) > 1e-10:
        for bx in [bbox_x1, bbox_x2]:
            t = (bx - x1) / dx
            y_at_t = y1 + t * dy
            if bbox_y1 <= y_at_t <= bbox_y2: t_values.append((t, bx, y_at_t))
    if abs(dy) > 1e-10:
        for by in [bbox_y1, bbox_y2]:
            t = (by - y1) / dy
            x_at_t = x1 + t * dx
            if bbox_x1 <= x_at_t <= bbox_x2: t_values.append((t, x_at_t, by))
    if len(t_values) < 2: return None
    t_values.sort(key=lambda x: x[0])
    return [t_values[0][1], t_values[0][2], t_values[-1][1], t_values[-1][2]]


def find_perpendicular_line_crossing_all(edge_lines, perp_angle, bbox, img2_bounds):
    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox
    img2_x1, img2_y1, img2_x2, img2_y2 = img2_bounds
    tol = 2.0
    meeting_edges = []
    if abs(bbox_x1 - img2_x1) < tol:
        seg_y1, seg_y2 = max(bbox_y1, img2_y1), min(bbox_y2, img2_y2)
        if seg_y2 > seg_y1: meeting_edges.append([bbox_x1, seg_y1, bbox_x1, seg_y2])
    if abs(bbox_x2 - img2_x2) < tol:
        seg_y1, seg_y2 = max(bbox_y1, img2_y1), min(bbox_y2, img2_y2)
        if seg_y2 > seg_y1: meeting_edges.append([bbox_x2, seg_y1, bbox_x2, seg_y2])
    if abs(bbox_y1 - img2_y1) < tol:
        seg_x1, seg_x2 = max(bbox_x1, img2_x1), min(bbox_x2, img2_x2)
        if seg_x2 > seg_x1: meeting_edges.append([seg_x1, bbox_y1, seg_x2, bbox_y1])
    if abs(bbox_y2 - img2_y2) < tol:
        seg_x1, seg_x2 = max(bbox_x1, img2_x1), min(bbox_x2, img2_x2)
        if seg_x2 > seg_x1: meeting_edges.append([seg_x1, bbox_y2, seg_x2, bbox_y2])
    if not meeting_edges: return None, [], -1

    def point_on_segment(px, py, x1, y1, x2, y2):
        return min(x1, x2) - tol <= px <= max(x1, x2) + tol and min(y1, y2) - tol <= py <= max(y1, y2) + tol

    meeting_intersections = []
    for edge_line in edge_lines:
        for meeting_seg in meeting_edges:
            pt = line_intersection(edge_line, meeting_seg)
            if pt and point_on_segment(pt[0], pt[1], *meeting_seg):
                meeting_intersections.append((pt, edge_line))
                break
    if len(meeting_intersections) != 4: return None, [], -1

    perp_rad = np.radians(perp_angle)
    perp_dx, perp_dy = np.cos(perp_rad), np.sin(perp_rad)
    candidates = [([pt[0] - 1000*perp_dx, pt[1] - 1000*perp_dy, pt[0] + 1000*perp_dx, pt[1] + 1000*perp_dy], pt) for pt, _ in meeting_intersections]

    best_candidate, best_idx, best_score = None, -1, -1
    for idx, (cand_line, origin_pt) in enumerate(candidates):
        intersections = []
        for edge_line in edge_lines:
            pt = line_intersection(cand_line, edge_line)
            if pt:
                intersections.append(pt)
        if len(intersections) > best_score:
            best_score, best_idx = len(intersections), idx
            if len(intersections) >= 2:
                projections = sorted([((pt[0] - origin_pt[0])*perp_dx + (pt[1] - origin_pt[1])*perp_dy, pt) for pt in intersections])
                pt_min, pt_max = projections[0][1], projections[-1][1]
                length = np.sqrt((pt_max[0] - pt_min[0])**2 + (pt_max[1] - pt_min[1])**2)
                if length > 0:
                    dx, dy = (pt_max[0] - pt_min[0]) / length, (pt_max[1] - pt_min[1]) / length
                    best_candidate = [pt_min[0] - 50*dx, pt_min[1] - 50*dy, pt_max[0] + 50*dx, pt_max[1] + 50*dy]
                else:
                    best_candidate = cand_line
            else:
                best_candidate = cand_line
    return best_candidate, [pt for pt, _ in meeting_intersections], best_idx


def compute_edge_alignment_data(img1, img2, mask1, mask2, transform, angle_tolerance=15):
    """Compute edge alignment data (x_1, x_2, y_1, y_2) for a pair."""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    tx, ty, scale = transform['tx'], transform['ty'], transform['scale']
    new_w2, new_h2 = int(w2 * scale), int(h2 * scale)
    if new_w2 <= 0 or new_h2 <= 0: return None

    mask2_scaled = cv2.resize(mask2.astype(np.uint8), (new_w2, new_h2), interpolation=cv2.INTER_NEAREST)
    min_x, min_y = min(0, tx), min(0, ty)
    max_x, max_y = max(w1, tx + new_w2), max(h1, ty + new_h2)
    canvas_w, canvas_h = int(max_x - min_x), int(max_y - min_y)
    x1_offset, y1_offset = int(-min_x), int(-min_y)
    x2_offset, y2_offset = int(tx - min_x), int(ty - min_y)

    mask1_region = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    mask1_region[y1_offset:y1_offset+h1, x1_offset:x1_offset+w1] = mask1
    mask2_region = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    y2_end, x2_end = min(y2_offset + new_h2, canvas_h), min(x2_offset + new_w2, canvas_w)
    y2_start, x2_start = max(0, y2_offset), max(0, x2_offset)
    if y2_end > y2_start and x2_end > x2_start:
        mask2_region[y2_start:y2_end, x2_start:x2_end] = mask2_scaled[y2_start-y2_offset:y2_end-y2_offset, x2_start-x2_offset:x2_end-x2_offset]

    bbox = (max(x1_offset, x2_offset), max(y1_offset, y2_offset), min(x1_offset + w1, x2_offset + new_w2), min(y1_offset + h1, y2_offset + new_h2))
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]: return None

    center1 = (x1_offset + w1 / 2, y1_offset + h1 / 2)
    center2 = (x2_offset + new_w2 / 2, y2_offset + new_h2 / 2)
    img2_bounds = (x2_offset, y2_offset, x2_offset + new_w2, y2_offset + new_h2)

    # Detect lines in overlap region
    mask1_overlap = mask1_region[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    mask2_overlap = mask2_region[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    lines1 = [[bbox[0]+x1, bbox[1]+y1, bbox[0]+x2, bbox[1]+y2] for x1, y1, x2, y2 in detect_lines_from_mask(mask1_overlap)]
    lines2 = [[bbox[0]+x1, bbox[1]+y1, bbox[0]+x2, bbox[1]+y2] for x1, y1, x2, y2 in detect_lines_from_mask(mask2_overlap)]

    # Select best lines
    selected1, _ = select_best_group(group_lines_by_angle(lines1, angle_tolerance))
    selected2, _ = select_best_group(group_lines_by_angle(lines2, angle_tolerance))
    edge1_1, edge2_1 = group_lines_by_position(selected1)
    edge1_2, edge2_2 = group_lines_by_position(selected2)
    final_lines1 = [l for l in [select_line_from_edge_group(edge1_1, center1, center2), select_line_from_edge_group(edge2_1, center1, center2)] if l]
    final_lines2 = [l for l in [select_line_from_edge_group(edge1_2, center1, center2), select_line_from_edge_group(edge2_2, center1, center2)] if l]

    if len(final_lines1) != 2 or len(final_lines2) != 2: return None

    # Check alignment
    all_lines = final_lines1 + final_lines2
    angles = [compute_line_angle(l) for l in all_lines]
    mean_angle = compute_group_mean_angle(angles)
    if any(angle_distance(a, mean_angle) > angle_tolerance for a in angles): return None

    perp_angle = (mean_angle + 90) % 180
    perp_line, _, _ = find_perpendicular_line_crossing_all(all_lines, perp_angle, bbox, img2_bounds)
    if perp_line is None: return None

    # Find intersections using extended (infinite) lines
    def find_intersections(lines):
        pts = []
        for edge_line in lines:
            pt = line_intersection(perp_line, edge_line)
            if pt:
                pts.append(pt)
        return pts

    intersections1, intersections2 = find_intersections(final_lines1), find_intersections(final_lines2)
    if len(intersections1) != 2 or len(intersections2) != 2: return None

    # Sort along perpendicular direction
    perp_rad = np.radians(perp_angle)
    perp_dx, perp_dy = np.cos(perp_rad), np.sin(perp_rad)
    intersections1.sort(key=lambda pt: pt[0]*perp_dx + pt[1]*perp_dy)
    intersections2.sort(key=lambda pt: pt[0]*perp_dx + pt[1]*perp_dy)

    return {'x_1': intersections1[0], 'x_2': intersections1[1], 'y_1': intersections2[0], 'y_2': intersections2[1]}


def compute_mask_iou_in_bbox(mask1, mask2, transform):
    """Compute IoU of masks restricted to overlapping bbox."""
    h1, w1 = mask1.shape[:2]
    h2, w2 = mask2.shape[:2]
    tx, ty, scale = transform['tx'], transform['ty'], transform['scale']
    new_w2, new_h2 = int(w2 * scale), int(h2 * scale)
    if new_w2 <= 0 or new_h2 <= 0:
        return 0.0
    mask2_scaled = cv2.resize(mask2.astype(np.uint8), (new_w2, new_h2), interpolation=cv2.INTER_NEAREST)
    min_x, min_y = min(0, tx), min(0, ty)
    max_x, max_y = max(w1, tx + new_w2), max(h1, ty + new_h2)
    canvas_w, canvas_h = int(max_x - min_x), int(max_y - min_y)
    x1_off, y1_off = int(-min_x), int(-min_y)
    x2_off, y2_off = int(tx - min_x), int(ty - min_y)
    m1 = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    m1[y1_off:y1_off+h1, x1_off:x1_off+w1] = mask1
    m2 = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    y2_end, x2_end = min(y2_off + new_h2, canvas_h), min(x2_off + new_w2, canvas_w)
    y2_start, x2_start = max(0, y2_off), max(0, x2_off)
    if y2_end > y2_start and x2_end > x2_start:
        m2[y2_start:y2_end, x2_start:x2_end] = mask2_scaled[y2_start-y2_off:y2_end-y2_off, x2_start-x2_off:x2_end-x2_off]
    bx1, by1 = max(x1_off, x2_off), max(y1_off, y2_off)
    bx2, by2 = min(x1_off + w1, x2_off + new_w2), min(y1_off + h1, y2_off + new_h2)
    if bx2 <= bx1 or by2 <= by1:
        return 0.0
    m1_crop, m2_crop = m1[by1:by2, bx1:bx2], m2[by1:by2, bx1:bx2]
    inter = np.logical_and(m1_crop > 0, m2_crop > 0).sum()
    union = np.logical_or(m1_crop > 0, m2_crop > 0).sum()
    return inter / union if union > 0 else 0.0


def compute_edge_aligned_transforms(images, masks, fallback_transforms):
    """
    Compute edge-aligned transforms using two-step process with IoU validation.

    Step 1: Translation - align midpoints of intersection points
    Step 2: Scaling - align intersection points (only if translation improved IoU)

    Returns list of transforms and detailed results.
    """
    aligned_transforms = []
    alignment_results = []

    for i in range(len(images) - 1):
        fallback_t = fallback_transforms[i]
        result = {
            'pair': i,
            'fallback_t': fallback_t,
            'translated_t': None,
            'scaled_t': None,
            'final_t': fallback_t,
            'iou_fallback': 0.0,
            'iou_translated': 0.0,
            'iou_scaled': 0.0,
            'translation_applied': False,
            'scaling_applied': False,
            'alignment_data': None,
            'reason': None
        }

        # Step 0: Compute fallback IoU
        iou_fallback = compute_mask_iou_in_bbox(masks[i], masks[i+1], fallback_t)
        result['iou_fallback'] = iou_fallback

        # Try to get alignment data
        alignment_data = compute_edge_alignment_data(images[i], images[i+1], masks[i], masks[i+1], fallback_t)
        if alignment_data is None:
            result['reason'] = 'edge detection failed'
            print(f"  Pair {i}: fallback (edge detection failed), IoU={iou_fallback:.3f}")
            aligned_transforms.append(fallback_t)
            alignment_results.append(result)
            continue

        result['alignment_data'] = alignment_data

        # Compute midpoints and distances
        x_1, x_2 = np.array(alignment_data['x_1']), np.array(alignment_data['x_2'])
        y_1, y_2 = np.array(alignment_data['y_1']), np.array(alignment_data['y_2'])

        mid_x = (x_1 + x_2) / 2  # midpoint of image 1's intersection points
        mid_y = (y_1 + y_2) / 2  # midpoint of image 2's intersection points

        d_1 = np.linalg.norm(x_2 - x_1)  # distance between image 1's points
        d_2 = np.linalg.norm(y_2 - y_1)  # distance between image 2's points

        if d_2 < 1e-6:
            result['reason'] = 'd_2 too small'
            print(f"  Pair {i}: fallback (d_2 too small), IoU={iou_fallback:.3f}")
            aligned_transforms.append(fallback_t)
            alignment_results.append(result)
            continue

        # Step 1: Translation only - move mid_y to mid_x
        translation = mid_x - mid_y
        translated_t = {
            'tx': fallback_t['tx'] + translation[0],
            'ty': fallback_t['ty'] + translation[1],
            'scale': fallback_t['scale']
        }
        result['translated_t'] = translated_t

        iou_translated = compute_mask_iou_in_bbox(masks[i], masks[i+1], translated_t)
        result['iou_translated'] = iou_translated

        # Check if translation improves IoU
        if iou_translated <= iou_fallback:
            result['reason'] = f'translation no gain ({iou_fallback:.3f}->{iou_translated:.3f})'
            print(f"  Pair {i}: fallback (translation no gain, IoU {iou_fallback:.3f}->{iou_translated:.3f})")
            aligned_transforms.append(fallback_t)
            alignment_results.append(result)
            continue

        # Translation improved IoU
        result['translation_applied'] = True
        print(f"  Pair {i}: translation applied (IoU {iou_fallback:.3f}->{iou_translated:.3f})")

        # Step 2: Scaling - align intersection points
        scale_factor = d_1 / d_2

        # Skip scaling if factor is out of range
        if scale_factor < 0.9 or scale_factor > 1.15:
            result['reason'] = f'scale factor out of range (sf={scale_factor:.4f})'
            result['final_t'] = translated_t
            print(f"  Pair {i}: +translation only (sf={scale_factor:.4f} out of [0.9, 1.15])")
            aligned_transforms.append(translated_t)
            alignment_results.append(result)
            continue

        scaled_t = {
            'tx': translated_t['tx'] * scale_factor + mid_x[0] * (1 - scale_factor),
            'ty': translated_t['ty'] * scale_factor + mid_x[1] * (1 - scale_factor),
            'scale': translated_t['scale'] * scale_factor
        }
        result['scaled_t'] = scaled_t

        iou_scaled = compute_mask_iou_in_bbox(masks[i], masks[i+1], scaled_t)
        result['iou_scaled'] = iou_scaled

        # Check if scaling improves IoU over translated
        if iou_scaled <= iou_translated:
            result['reason'] = f'scaling no gain ({iou_translated:.3f}->{iou_scaled:.3f})'
            result['final_t'] = translated_t
            print(f"  Pair {i}: +translation only (scaling no gain, IoU {iou_translated:.3f}->{iou_scaled:.3f})")
            aligned_transforms.append(translated_t)
            alignment_results.append(result)
            continue

        # Both translation and scaling improved IoU
        result['scaling_applied'] = True
        result['final_t'] = scaled_t
        print(f"  Pair {i}: +translation +scaling (sf={scale_factor:.4f}, IoU {iou_translated:.3f}->{iou_scaled:.3f})")
        aligned_transforms.append(scaled_t)
        alignment_results.append(result)

    return aligned_transforms, alignment_results
