"""Core computer-vision helpers: IoU, NMS (from scratch), drawing."""
import cv2
import numpy as np


def iou(box_a, box_b):
    """Intersection over Union for boxes in [x1, y1, x2, y2] format."""
    xa, ya = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    xb, yb = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def nms(boxes, scores, iou_thresh=0.5):
    """Non-Maximum Suppression. Returns indices of boxes to keep."""
    order = np.argsort(scores)[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(int(i))
        order = np.array([j for j in order[1:] if iou(boxes[i], boxes[j]) < iou_thresh])
    return keep


def color_for(class_id):
    rng = np.random.default_rng(class_id + 7)
    return tuple(int(c) for c in rng.integers(60, 255, 3))


def draw_boxes(frame, boxes, scores, class_ids, names):
    for (x1, y1, x2, y2), s, c in zip(boxes, scores, class_ids):
        color = color_for(int(c))
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{names[int(c)]} {s:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - h - 6), (x1 + w + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def yolo_to_xyxy(cx, cy, w, h, img_w, img_h):
    """Convert normalized YOLO label (cx, cy, w, h) to pixel xyxy."""
    return ((cx - w / 2) * img_w, (cy - h / 2) * img_h,
            (cx + w / 2) * img_w, (cy + h / 2) * img_h)
