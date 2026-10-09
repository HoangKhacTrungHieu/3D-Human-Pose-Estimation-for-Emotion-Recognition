import cv2
import numpy as np
from ultralytics import YOLO


def _bbox_center(b):
    return np.array([(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0])


def _pick_centered_track(track_ids, boxes_xyxy, frame_w, frame_h,
                          edge_margin_ratio, min_height_ratio=0.35):
    """
    Given detections in a single frame, return (track_id, det_idx) of the
    most horizontally-centered one, subject to two eligibility filters:

      1. Size gate: bbox height must be at least `min_height_ratio` of the
         frame height. This excludes small/background people (e.g. someone
         sitting far away) from ever winning purely on x-position.
      2. Edge gate: bbox center must not fall within `edge_margin_ratio` of
         the frame width from the left/right edge.

    Both gates are relaxed in order (edge gate first, then size gate) if
    no candidate satisfies them, so we always return *something* rather
    than crash on an empty eligible set.
    """
    heights = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]
    big_enough = heights >= (min_height_ratio * frame_h)

    frame_center_x = frame_w / 2.0
    centers_x = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2.0
    dist_to_center = np.abs(centers_x - frame_center_x)

    margin_px = edge_margin_ratio * frame_w
    near_edge = (centers_x < margin_px) | (centers_x > (frame_w - margin_px))

    eligible_mask = big_enough & ~near_edge
    if not np.any(eligible_mask):
        eligible_mask = big_enough  # relax edge constraint before size constraint
    if not np.any(eligible_mask):
        eligible_mask = np.ones_like(near_edge, dtype=bool)  # last resort

    eligible_indices = np.where(eligible_mask)[0]
    best_idx = int(eligible_indices[np.argmin(dist_to_center[eligible_indices])])
    return track_ids[best_idx], best_idx, dist_to_center[best_idx], near_edge[best_idx]


def _pick_by_continuity(track_ids, boxes_xyxy, prev_bbox, frame_w, max_jump_ratio=0.25):
    """
    Pick the candidate whose bbox center is closest to prev_bbox's center
    (i.e. spatial continuity with the last known detection). Returns None
    if there's no prior bbox to anchor on, or if even the closest candidate
    jumped an implausible distance (meaning the real subject likely left
    frame and we shouldn't just lock onto whoever happens to be nearest).
    """
    if prev_bbox is None:
        return None

    centers = np.stack([
        (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2.0,
        (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2.0
    ], axis=1)  # (N, 2)

    prev_center = _bbox_center(prev_bbox)
    dists = np.linalg.norm(centers - prev_center, axis=1)
    best_idx = int(np.argmin(dists))

    max_jump_px = max_jump_ratio * frame_w
    if dists[best_idx] <= max_jump_px:
        return track_ids[best_idx], best_idx, dists[best_idx]

    return None  # no continuity match, caller should re-acquire fresh


def gen_video_kpts_yolo(video_path, imgsz=416, model_name="checkpoint/YOLO26l-pose.pt",
                         conf_threshold=0.7, center_edge_margin_ratio=0.15,
                         min_height_ratio=0.35, max_jump_ratio=0.25):
    """
    Detects and tracks 2D keypoints in a video using YOLO Pose, keeping
    the identity of a single centered subject stable across frames.

    Re-acquisition strategy, in order of preference:
      1. Continuity: if we already have a `prev_bbox`, prefer whichever
         currently-visible track's bbox center is closest to it (bounded
         by `max_jump_ratio` so we don't lock onto an implausible jump).
      2. Size + center: if continuity fails (no prior bbox, or nothing
         plausibly close), fall back to picking the most centered track
         among those large enough to plausibly be the foreground subject
         (`min_height_ratio` of frame height), excluding near-edge boxes
         when a better option exists.

    Args:
        center_edge_margin_ratio: fraction of frame width (from each edge)
            considered "too close to the edge" to be acceptable.
        min_height_ratio: minimum bbox height (as a fraction of frame
            height) for a candidate to be eligible during fresh
            re-acquisition. Filters out small/background people.
        max_jump_ratio: maximum allowed bbox-center displacement (as a
            fraction of frame width) between frames for continuity
            matching to be trusted.

    Returns:
        keypoints: np.ndarray of shape (1, T, 17, 2)
        scores: np.ndarray of shape (1, T, 17)
    """
    model = YOLO(model_name)

    results = model.track(source=video_path, imgsz=imgsz, persist=True, conf=conf_threshold, verbose=False)

    T = len(results)

    keypoints_all = np.zeros((1, T, 17, 2), dtype=np.float32)
    scores_all = np.zeros((1, T, 17), dtype=np.float32)

    prev_kpts = np.zeros((17, 2), dtype=np.float32)
    prev_scores = np.zeros(17, dtype=np.float32)
    prev_bbox = None  # last known bbox [x1, y1, x2, y2] of the followed subject

    current_track_id = None  # the identity we're currently following

    for t, r in enumerate(results):
        if r.boxes is None or r.boxes.id is None or r.keypoints is None:
            # No detections at all this frame -- freeze last known pose.
            keypoints_all[0, t] = prev_kpts
            scores_all[0, t] = prev_scores
            continue

        track_ids = r.boxes.id.int().cpu().tolist()
        boxes_xyxy = r.boxes.xyxy.cpu().numpy()
        frame_kpts = r.keypoints.xy.cpu().numpy()
        frame_confs = r.keypoints.conf.cpu().numpy()

        if len(track_ids) == 0:
            keypoints_all[0, t] = prev_kpts
            scores_all[0, t] = prev_scores
            continue

        frame_h, frame_w = r.orig_shape[0], r.orig_shape[1]
        margin_px = center_edge_margin_ratio * frame_w

        need_reacquire = True
        det_idx = None

        # If we're already following someone, check whether they're
        # still present AND still acceptably centered this frame.
        if current_track_id is not None and current_track_id in track_ids:
            idx = track_ids.index(current_track_id)
            center_x = (boxes_xyxy[idx, 0] + boxes_xyxy[idx, 2]) / 2.0
            is_near_edge = center_x < margin_px or center_x > (frame_w - margin_px)
            if not is_near_edge:
                det_idx = idx
                need_reacquire = False

        if need_reacquire:
            # 1. Try continuity first: stick close to where the subject
            #    last was, rather than jumping to whoever's most centered.
            continuity_result = _pick_by_continuity(
                track_ids, boxes_xyxy, prev_bbox, frame_w, max_jump_ratio
            )

            if continuity_result is not None:
                new_track_id, new_det_idx, dist = continuity_result
                if new_track_id != current_track_id:
                    print(f"[frame {t}] Re-acquiring via continuity: switching from "
                          f"track {current_track_id} to track {new_track_id} "
                          f"(bbox_jump={dist:.1f}px)")
            else:
                # 2. Fall back to size + centeredness (fresh re-acquire).
                new_track_id, new_det_idx, dist, was_near_edge = _pick_centered_track(
                    track_ids, boxes_xyxy, frame_w, frame_h,
                    center_edge_margin_ratio, min_height_ratio
                )
                if new_track_id != current_track_id:
                    print(f"[frame {t}] Re-acquiring via size+center: switching from "
                          f"track {current_track_id} to track {new_track_id} "
                          f"(center_dist={dist:.1f}px, near_edge_fallback={was_near_edge})")

            current_track_id = new_track_id
            det_idx = new_det_idx

        kpt = frame_kpts[det_idx]
        score = frame_confs[det_idx]

        keypoints_all[0, t] = kpt
        scores_all[0, t] = score

        prev_kpts = kpt
        prev_scores = score
        prev_bbox = boxes_xyxy[det_idx]

    return keypoints_all, scores_all