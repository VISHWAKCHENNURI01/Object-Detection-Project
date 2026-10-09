"""Quick test of from-scratch IoU/NMS. Run: python tests_iou_nms.py"""
import sys
sys.path.insert(0, "src")
from utils import iou, nms

assert abs(iou([0, 0, 10, 10], [0, 0, 10, 10]) - 1.0) < 1e-9
assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
boxes = [[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]]
assert nms(boxes, [0.9, 0.8, 0.7], 0.5) == [0, 2]
print("All tests passed")
