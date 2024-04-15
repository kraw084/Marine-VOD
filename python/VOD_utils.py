import os
import sys
import torch
import cv2
import numpy as np
import colorsys
from Video_utils import annotate_image, resize_image, play_frame_by_frame

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_dir)
from yolov5.utils.metrics import box_iou, bbox_iou

NUM_OF_COLOURS = 8
colours = [colorsys.hsv_to_rgb(hue, 0.8, 1) for hue in np.linspace(0, 1, NUM_OF_COLOURS + 1)][:-1]
colours = [(round(255 * c[2]), round(255 * c[1]), round(255 * c[0])) for c in colours]

class Tracklet:
    def __init__(self, id):
        self.__boxes = []
        self.__frame_indexes = []
        self.__id = id

    def add_box(self, box, frame_index):
        self.__boxes.append(box)
        self.__frame_indexes.append(frame_index)
        
    def tracklet_length(self):
        return len(self.__boxes)
    
    def id(self):
        return self.__id
    
    def __iter__(self):
        self.__i = 0
        return self
    
    def __next__(self):
        if self.__i >= len(self.__boxes): raise StopIteration
        val = (self.__frame_indexes[self.__i], self.__boxes[self.__i])
        self.__i += 1
        return val


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


def display_tracklet(frames, tracklet, num_to_label, num_to_colour):
    """Draws a single tracklet on a video segment and displays it frame by frame"""
    id = tracklet.id()

    for frame_index, box in tracklet:
        frame = frames[frame_index]
        annotate_image(frame, [box], num_to_label, num_to_colour)
        frame = resize_image(frame, 640)
        cv2.imshow(f"Tracklet {id}", frame)
        cv2.waitKey(0)


def display_VOD(frames, tracklets, num_to_label, size=640, delay=0):
    """Draws a set of tracklets onto a video and plays it frame by frame"""
    frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in frames]

    for i, tracklet in enumerate(tracklets):
        id = tracklet.id()
        colour = colours[i%len(colours)]
        for frame_index, box in tracklet:
            annotate_image(frames[frame_index], [box], num_to_label, [colour] * len(num_to_label), ids=[id])

    play_frame_by_frame(frames, delay, size=size)
    