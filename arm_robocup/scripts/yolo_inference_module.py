import cv2
import numpy as np
from ultralytics import YOLO
import pathlib
import os
import colorsys

# --- Configurations ---
EXCLUDE_GRIPPER = True
HIDE_GRIPPER_IN_ANNOTATION = True
COLOR_PATCH_RADIUS = 2  
COLOR_PACK_CHANNEL_ORDER = "RGB"  
OVERLAY_TOP1_INFO = True
GRIPPER_EXCLUSION_POLYGONS_REL = [
    [(0.00, 0.82), (0.34, 0.82), (0.26, 1.00), (0.00, 1.00)],
    [(0.66, 0.82), (1.00, 0.82), (1.00, 1.00), (0.74, 1.00)],
]

OBJECT_LABELS = {
    "0": "BOTTLE",
    "1": "CAN",
    "2": "CUBE",
    "3": "SPAM",
    "4": "MARKER",
}


def _label_from_cls_id(cls_id, yolo_names=None):
    """Return an uppercase label for a YOLO class id."""
    try:
        cls_int = int(cls_id)
    except Exception:
        return ""

    # Prefer explicit mapping if provided
    mapped = OBJECT_LABELS.get(str(cls_int))
    if mapped:
        return str(mapped).upper()

    # Fallback to YOLO model names
    if yolo_names is not None:
        try:
            name = yolo_names.get(cls_int)
            if name is not None:
                return str(name).upper()
        except Exception:
            pass

    return ""


def _cube_bin_from_rgb(r, g, b):
    """Return bin id for cube based on sampled center RGB.

    Rules:
    - GREEN or PURPLE cube -> 1 (green bin)
    - RED or BLUE cube -> 2 (blue bin)
    """
    if r < 0 or g < 0 or b < 0:
        # If color sampling failed, default to blue bin (conservative for sorting)
        return 2.0

    rr, gg, bb = float(r) / 255.0, float(g) / 255.0, float(b) / 255.0
    h, s, v = colorsys.rgb_to_hsv(rr, gg, bb)

    # Low-saturation/very-dark: fall back to simple channel heuristics
    if v < 0.12 or s < 0.18:
        # purple: red+blue high and green low
        if (r > 120 and b > 120 and g < 100):
            return 1.0
        # dominant green/red/blue
        if g >= r and g >= b:
            return 1.0
        return 2.0

    # Hue bands (0..1)
    is_red = (h < 0.05) or (h > 0.95)
    is_yellow = (0.11 <= h <= 0.18)
    is_green = (0.20 <= h <= 0.45)
    is_blue = (0.52 <= h <= 0.75)
    is_purple = (0.75 < h <= 0.92)

    if is_green or is_purple:
        return 1.0
    if is_red or is_blue:
        return 2.0

    # If ambiguous (e.g., yellow-ish), pick nearest among {green,purple} vs {red,blue}
    if is_yellow:
        return 1.0
    return 2.0


def _bin_target(cls_id, r, g, b, yolo_names=None):
    """Return bin target value: 1 (green), 2 (blue)."""
    label = _label_from_cls_id(cls_id, yolo_names=yolo_names)

    if label in {"CAN", "SPAM"}:
        return 1.0
    if label in {"BOTTLE", "MARKER"}:
        return 2.0
    if label == "CUBE":
        return _cube_bin_from_rgb(r, g, b)

    # Unknown label: default to blue bin
    return 2.0


def _is_point_in_poly(pt, poly):
    return cv2.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), False) >= 0

def _is_in_gripper_exclusion_zone(cx, cy, img_w, img_h):
    if not EXCLUDE_GRIPPER: return False
    for poly_rel in GRIPPER_EXCLUSION_POLYGONS_REL:
        poly_px = np.array([(int(x * img_w), int(y * img_h)) for (x, y) in poly_rel], dtype=np.int32)
        if _is_point_in_poly((cx, cy), poly_px): return True
    return False

def _hide_gripper_region(annotated_img, src_img, img_w, img_h):
    if not (EXCLUDE_GRIPPER and HIDE_GRIPPER_IN_ANNOTATION): return annotated_img
    if src_img is None: return annotated_img

    src = np.ascontiguousarray(src_img)
    if src.shape[:2] != annotated_img.shape[:2]: return annotated_img
    if src.dtype != annotated_img.dtype:
        src_f = src.astype(np.float32)
        if src_f.size > 0 and np.nanmax(src_f) <= 1.0: src_f = src_f * 255.0
        src = np.clip(src_f, 0, 255).astype(annotated_img.dtype)

    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for poly_rel in GRIPPER_EXCLUSION_POLYGONS_REL:
        poly_px = np.array([(int(x * img_w), int(y * img_h)) for (x, y) in poly_rel], dtype=np.int32)
        cv2.fillPoly(mask, [poly_px], 1)
    annotated_img[mask == 1] = src[mask == 1]
    return annotated_img

def _to_uint8_img(img_any):
    if img_any is None: return None
    arr = np.ascontiguousarray(img_any)
    if arr.dtype == np.uint8: return arr
    arr_f = arr.astype(np.float32)
    if arr_f.size > 0 and np.nanmax(arr_f) <= 1.0: arr_f = arr_f * 255.0
    return np.clip(arr_f, 0, 255).astype(np.uint8)

def _sample_center_rgb(color_img_u8, cx, cy, patch_radius, channel_order):
    if color_img_u8 is None: return -1.0, -1.0, -1.0
    h, w = color_img_u8.shape[:2]
    x, y = int(round(float(cx))), int(round(float(cy)))
    if x < 0 or y < 0 or x >= w or y >= h: return -1.0, -1.0, -1.0

    r = int(patch_radius)
    patch = color_img_u8[max(0, y-r):min(h, y+r+1), max(0, x-r):min(w, x+r+1)]
    if patch.size == 0: return -1.0, -1.0, -1.0

    mean_c = patch.reshape(-1, patch.shape[-1]).mean(axis=0)
    c0, c1, c2 = [int(round(v)) for v in mean_c[:3]]
    if channel_order.upper() == "BGR": return float(c2), float(c1), float(c0)
    return float(c0), float(c1), float(c2)

# --- NEW DEPTH SAMPLING FUNCTION ---
def _sample_center_depth(depth_img, cx, cy, patch_radius):
    if depth_img is None: return -1.0
    h, w = depth_img.shape[:2]
    x, y = int(round(float(cx))), int(round(float(cy)))
    if x < 0 or y < 0 or x >= w or y >= h: return -1.0

    r = int(patch_radius)
    patch = depth_img[max(0, y-r):min(h, y+r+1), max(0, x-r):min(w, x+r+1)]
    if patch.size == 0: return -1.0

    # Filter out 0 values (often represents invalid depth data)
    valid_depths = patch[patch > 0]
    if valid_depths.size == 0: return -1.0
    
    # Use median to avoid extreme noise spikes
    return float(np.median(valid_depths))

model = None
MODEL_PATH = pathlib.Path(__file__).parent / "yolo_model/best_v1_blender.pt"

# --- UPDATED: PREDICT NOW TAKES depth_data ---
def predict(img_data, depth_data=None, model_in=None):
    # Formatting
    # Sort by closest first (smallest valid depth); if depth is invalid, fall back
    # to the image-center distance. For ties, prefer higher confidence.
    def _sort_key(obj):
        conf_v = obj[4]
        center_dist_v = obj[-2]
        return (center_dist_v, -conf_v)
    
    global model
    if model is None:
        model = YOLO(str(MODEL_PATH))

    img_np = np.array(img_data)
    img = np.ascontiguousarray(img_np)
    
    depth_img = None
    if depth_data is not None:
        depth_img = np.ascontiguousarray(np.array(depth_data))

    # YOLO Inference
    results = model(img, verbose=False)[0]
    # Build an output image that will contain ONLY the selected top-1 object.
    # Draw on the original input image (img_data).
    annotated_img = _to_uint8_img(img)
    if annotated_img is None:
        annotated_img = _to_uint8_img(getattr(results, "orig_img", None))
    annotated_img = np.ascontiguousarray(annotated_img).copy()
    
    object_list = []
    img_h, img_w = results.orig_shape
    img_center_x, img_center_y = img_w / 2, img_h / 2

    color_src_img_u8 = _to_uint8_img(img)
    if color_src_img_u8 is None:
        color_src_img_u8 = _to_uint8_img(getattr(results, "orig_img", None))

    annotated_img = _hide_gripper_region(annotated_img, img, img_w, img_h)
    
    if results.masks is not None:
        for det_idx, (box, mask) in enumerate(zip(results.boxes, results.masks)):
            pixel_points = mask.xy[0].astype(np.float32)
            if len(pixel_points) > 0:
                rect = cv2.minAreaRect(pixel_points)
                (cx, cy), (w_p, h_p), angle = rect

                if _is_in_gripper_exclusion_zone(cx, cy, img_w, img_h):
                    continue
                
                yaw_deg = angle if w_p > h_p else angle - 90
                yaw = abs(float(np.deg2rad(yaw_deg)))
                rel_x = (cx - img_center_x) / img_center_x
                rel_y = (cy - img_center_y) / img_center_y
                cls_id = float(box.cls[0])
                conf = float(box.conf[0]) if getattr(box, "conf", None) is not None else -1.0

                r_val, g_val, b_val = _sample_center_rgb(color_src_img_u8, cx, cy, COLOR_PATCH_RADIUS, COLOR_PACK_CHANNEL_ORDER)
                
                # --- NEW: Get Depth ---
                depth_val = _sample_center_depth(depth_img, cx, cy, COLOR_PATCH_RADIUS)

                # --- NEW: Bin target (1=green, 2=blue) ---
                bin_target = _bin_target(cls_id, r_val, g_val, b_val, yolo_names=getattr(results, "names", None))
                
                distance = np.sqrt(rel_x**2 + rel_y**2)
                
                # Exported params (7): rel_x, rel_y, yaw, cls_id, conf, depth, bin_target
                # Sorting-only param: distance
                # Keep det_idx so we can draw only the selected top-1 object.
                object_list.append([rel_x, rel_y, yaw, cls_id, conf, depth_val, bin_target, distance, float(det_idx)])

    object_list.sort(key=_sort_key)
    num_found = len(object_list)

    # Draw only the selected detection (top 1)
    if num_found > 0:
        best_det_idx = int(round(float(object_list[0][-1])))
        overlay_text = None
        if OVERLAY_TOP1_INFO:
            try:
                rel_x_v = float(object_list[0][0])
                rel_y_v = float(object_list[0][1])
                cls_id_v = float(object_list[0][3])
                conf_v = float(object_list[0][4])
                label = _label_from_cls_id(cls_id_v, yolo_names=getattr(results, "names", None))
                if label == "":
                    label = str(int(round(cls_id_v)))
                overlay_text = f"{label} {conf_v:.2f}  x={rel_x_v:.2f} y={rel_y_v:.2f}"
            except Exception:
                overlay_text = None
        try:
            b = results.boxes[best_det_idx]
            x1, y1, x2, y2 = [int(round(v)) for v in b.xyxy[0].tolist()]
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            poly = np.array(results.masks.xy[best_det_idx], dtype=np.int32)
            if poly.size > 0:
                cv2.polylines(annotated_img, [poly], True, (0, 255, 0), 2)

            if overlay_text:
                org = (max(0, x1), max(0, y1 - 8))
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.55
                # Outline then text for readability
                cv2.putText(annotated_img, overlay_text, org, font, font_scale, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(annotated_img, overlay_text, org, font, font_scale, (0, 255, 0), 1, cv2.LINE_AA)
        except Exception:
            pass
    
    #  1 objects x 7 parameters ---
    MAX_DETECTIONS = 1
    final_detections = np.full((MAX_DETECTIONS, 7), -1.0, dtype=np.float64)
    
    limit = min(num_found, MAX_DETECTIONS)
    for i in range(limit):
        final_detections[i] = object_list[i][:7]
        
    flat_det = final_detections.flatten()

    # Annotated image as flat array of floats (Fortran order for MATLAB/Simulink)
    safe_out_img = np.ravel(annotated_img, order='F').astype(np.float64)
    img_len = float(safe_out_img.size)
        
    # Output layout:
    #   [img_len, flat_image..., num_found, flat_detections...]
    # flat_detections are 1x7:
    #   [rel_x, rel_y, yaw, cls_id, conf, depth, bin_target]
    # where bin_target: 1=green bin, 2=blue bin
    return np.concatenate([[img_len], safe_out_img, [float(num_found)], flat_det])

def test():
    raise RuntimeError("Testing YOLO inference module...")