import os
import sys
import torch
import cv2
import numpy as np
import colorsys

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_dir)
from yolov5.utils.metrics import box_iou, bbox_iou

NUM_OF_COLOURS = 8
colours = [colorsys.hsv_to_rgb(hue, 0.8, 1) for hue in np.linspace(0, 1, NUM_OF_COLOURS + 1)][:-1]
colours = [(round(255 * c[0]), round(255 * c[1]), round(255 * c[2])) for c in colours]

def annotate_image(im, prediction, num_to_label, num_to_colour, draw_labels=True, ids=None):
        """Draws xywhcl boxes onto a single image. Colours are BGR"""
        thickness = 2
        font_size = 0.75

        label_data = []
        for i, pred in enumerate(prediction):
            top_left = (int(pred[0]) - int(pred[2])//2, int(pred[1]) - int(pred[3])//2)
            bottom_right = (top_left[0] + int(pred[2]), top_left[1] + int(pred[3]))
            label = num_to_label[int(pred[5])]

            colour = num_to_colour[int(pred[5])]

            #Draw boudning box
            im = cv2.rectangle(im, top_left, bottom_right, colour, thickness)

            label_data.append((f"{f'{ids[i]}. ' if ids else ''}{label} - {float(pred[4]):.2f}", top_left, colour))
        
        #Draw text over boxes
        if draw_labels:
            for data in label_data:
                text_size = cv2.getTextSize(data[0], cv2.FONT_HERSHEY_SIMPLEX, font_size, thickness)[0]
                text_box_top_left = (data[1][0], data[1][1] - text_size[1])
                text_box_bottom_right = (data[1][0] + text_size[0], data[1][1])
                im = cv2.rectangle(im, text_box_top_left, text_box_bottom_right, data[2], -1)
                im = cv2.putText(im, data[0], data[1], cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), thickness - 1, cv2.LINE_AA)


class Tracklet:
    def __init__(self, id):
        self.__boxes = []
        self.__frame_indexes = []
        self.__id = id
        self.__start_frame = None
        self.__end_frame = -1

    def add_box(self, box, frame_index):
        self.__boxes.append(box)
        self.__frame_indexes.append(frame_index)

        if self.__start_frame is None: self.__start_frame = frame_index

        if frame_index < self.__start_frame: self.__start_frame = frame_index
        if frame_index > self.__end_frame: self.__end_frame = frame_index
        
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


class TrackletSet:
    def __init__(self, video, tracklets, num_to_label):
        self.__video = video
        self.__tracklets = tracklets
        self.__objects_count = len(self.__tracklets)

        self.__num_to_label = num_to_label


    def count_per_frame(self):
        counts = [0] * len(self.__video.num_of_frames())
        totals = [0] * len(self.__video.num_of_frames())

        for tracklet in self.__tracklets:
            for frame_index, _ in tracklet:
                counts[frame_index] += 1
                for i in range(frame_index, len(totals) - 1): totals[i] += 1

        return counts, totals
    
    def play_video(self, fps=None, size=1080):
        frames = self.__video.frames()

        for i, tracklet in enumerate(self.__tracklets):
            id = tracklet.id()
            colour = colours[i%len(colours)]
            for frame_index, box in tracklet:
                annotate_image(frames[frame_index], [box], self.__num_to_label, 
                               [colour] * len(self.__num_to_label), ids=[id])
                
        self.__video.play(fps=fps, resize=size)


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