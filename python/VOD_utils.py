import os
import sys
import torch

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_dir)
from yolov5.utils.metrics import box_iou, bbox_iou


def xywhTOxyxy(x, y, w, h):
    return x - w//2, y - h//2, x + w//2, y + h//2

def iou(box1, box2):
    """Returns the iou between two boxes"""
    box1 = torch.tensor(xywhTOxyxy(*box1[:4]))
    box2 = torch.tensor(xywhTOxyxy(*box2[:4]))

    return bbox_iou(box1, box2, xywh=False).item()


def iou_matrix(boxes1, boxes2):
    """"Calculates iou between every box in boxes1 and every box in boxes2,
        matrix[n][m] is the iou between box n in boxes1 and box m in boxes2"""
    
    boxes1 = torch.tensor([xywhTOxyxy(*box1[:4]) for box1 in boxes1])
    boxes2 = torch.tensor([xywhTOxyxy(*box2[:4]) for box2 in boxes2])

    return box_iou(boxes1, boxes2).numpy()