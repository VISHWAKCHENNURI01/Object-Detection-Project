"""Classical (non-deep-learning) detection techniques for comparison.
  hog      - HOG + linear SVM pedestrian detector
  haar     - Haar cascade face detector
  color    - HSV color segmentation + contours
Usage: python src/classical_cv.py --method hog --source samples/street.jpg
"""
import argparse
import cv2
import numpy as np


def hog_people(img):
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    rects, _ = hog.detectMultiScale(img, winStride=(8, 8), padding=(8, 8), scale=1.05)
    for (x, y, w, h) in rects:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return img, len(rects)


def haar_faces(img):
    path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(path)
    gray = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return img, len(faces)


def color_objects(img, lower=(35, 80, 80), upper=(85, 255, 255)):
    """Default range detects green objects. Tune HSV bounds per target."""
    hsv = cv2.cvtColor(cv2.GaussianBlur(img, (7, 7), 0), cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    n = 0
    for c in cnts:
        if cv2.contourArea(c) > 500:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            n += 1
    return img, n


METHODS = {"hog": hog_people, "haar": haar_faces, "color": color_objects}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--out", default="outputs/classical_result.jpg")
    a = p.parse_args()
    img = cv2.imread(a.source)
    if img is None:
        raise FileNotFoundError(a.source)
    out, n = METHODS[a.method](img)
    cv2.imwrite(a.out, out)
    print(f"{n} detections -> {a.out}")


if __name__ == "__main__":
    main()
