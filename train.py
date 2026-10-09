"""Train / fine-tune a YOLO detector on a custom dataset.
Usage: python src/train.py --data data/dataset.yaml --epochs 50
"""
import argparse
from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/dataset.yaml")
    p.add_argument("--model", default="yolov8n.pt", help="pretrained weights")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default=None, help="'cpu', '0' for GPU")
    a = p.parse_args()

    model = YOLO(a.model)
    model.train(
        data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
        device=a.device, project="outputs", name="train",
        augment=True, mosaic=1.0, fliplr=0.5, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    )
    print("Best weights: outputs/train/weights/best.pt")


if __name__ == "__main__":
    main()
