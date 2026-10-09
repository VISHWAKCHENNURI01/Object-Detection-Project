"""Evaluate a trained model: precision, recall, mAP@0.5, mAP@0.5:0.95.
Usage: python src/evaluate.py --weights outputs/train/weights/best.pt
"""
import argparse
from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--data", default="data/dataset.yaml")
    a = p.parse_args()

    m = YOLO(a.weights).val(data=a.data, project="outputs", name="eval")
    print(f"Precision   : {m.box.mp:.3f}")
    print(f"Recall      : {m.box.mr:.3f}")
    print(f"mAP@0.5     : {m.box.map50:.3f}")
    print(f"mAP@0.5:.95 : {m.box.map:.3f}")


if __name__ == "__main__":
    main()
