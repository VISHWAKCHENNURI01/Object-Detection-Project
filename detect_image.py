"""Run detection on an image.
Usage: python src/detect_image.py --source samples/street.jpg --weights yolov8n.pt
"""
import argparse
import cv2
from ultralytics import YOLO
from utils import draw_boxes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--weights", default="yolov8n.pt")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    p.add_argument("--out", default="outputs/result.jpg")
    a = p.parse_args()

    model = YOLO(a.weights)
    img = cv2.imread(a.source)
    if img is None:
        raise FileNotFoundError(a.source)
    r = model.predict(img, conf=a.conf, iou=a.iou, verbose=False)[0]

    boxes = r.boxes.xyxy.cpu().numpy()
    scores = r.boxes.conf.cpu().numpy()
    cls = r.boxes.cls.cpu().numpy()
    out = draw_boxes(img.copy(), boxes, scores, cls, r.names)
    cv2.imwrite(a.out, out)
    print(f"{len(boxes)} objects detected -> {a.out}")
    for b, s, c in zip(boxes, scores, cls):
        print(f"  {r.names[int(c)]:<12} {s:.2f}  {b.astype(int).tolist()}")


if __name__ == "__main__":
    main()
