import os
import sys
import torch
import cv2
import numpy as np
import colorsys
import math

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_dir)
from yolov5.utils.metrics import box_iou, bbox_iou

from Video_utils import Video

NUM_OF_COLOURS = 8
colours = [colorsys.hsv_to_rgb(hue, 0.8, 1) for hue in np.linspace(0, 1, NUM_OF_COLOURS + 1)][:-1]
colours = [(round(255 * c[0]), round(255 * c[1]), round(255 * c[2])) for c in colours]

def annotate_image(im, prediction, num_to_label, num_to_colour, draw_labels=True, ids=None):
        """Draws xywhcl boxes onto a single image. Colours are BGR"""
        thickness = 3
        font_size = 1

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


def draw_data(im, data_dict):
    thickness = 2
    font_size = 1.5
    start_point = (40, 60)
    gap = 70

    for i, data in enumerate(data_dict):
        text = f"{data}: {data_dict[data]}"
        im = cv2.putText(im, text, (start_point[0], start_point[1] + i * gap), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), thickness, cv2.LINE_AA)


class Tracklet:
    def __init__(self, id):
        self.boxes = []
        self.frame_indexes = []
        self.id = id
        self.start_frame = None
        self.end_frame = -1

    def add_box(self, box, frame_index):
        self.boxes.append(box)
        self.frame_indexes.append(frame_index)

        if self.start_frame is None: self.start_frame = frame_index

        if frame_index < self.start_frame: self.start_frame = frame_index
        if frame_index > self.end_frame: self.end_frame = frame_index
        
    def tracklet_length(self):
        return len(self.boxes)
    
    def __iter__(self):
        self.i = 0
        return self
    
    def __next__(self):
        if self.i >= len(self.boxes): raise StopIteration
        val = (self.frame_indexes[self.i], self.boxes[self.i])
        self.i += 1
        return val


class TrackletSet:
    def __init__(self, video, tracklets, num_to_label):
        self.video = video
        self.tracklets = tracklets
        self.objects_count = len(self.tracklets)

        self.num_to_label = num_to_label

    def count_per_frame(self):
        counts = [0] * self.video.num_of_frames
        totals = [0] * self.video.num_of_frames

        for tracklet in self.tracklets:
            start_i, end_i = tracklet.start_frame, tracklet.end_frame

            for i in range(start_i, end_i + 1): counts[i] += 1
            for i in range(start_i, self.video.num_of_frames): totals[i] += 1

        return counts, totals
    
    def draw_tracklets(self):
        frames = self.video.frames
        counts, totals = self.count_per_frame()

        for i, tracklet in enumerate(self.tracklets):
            id = tracklet.id
            colour = colours[i%len(colours)]
            for frame_index, box in tracklet:
                annotate_image(frames[frame_index], [box], self.num_to_label, 
                               [colour] * len(self.num_to_label), ids=[id])
        
        for i, frame in enumerate(frames):
            draw_data(frame, {"Objects":counts[i], "Total":totals[i]})


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


def frame_by_frame_VOD(model, video, no_save=False):
    frame_predictions = [model.xywhcl(frame) for frame in video]
    print("Finished predicting")

    total = 0
    for frame, frame_pred in zip(video, frame_predictions):
        count = len(frame_pred)
        total += count

        annotate_image(frame, frame_pred, model.num_to_class, model.num_to_colour)
        draw_data(frame, {"Objects":count, "Total":total})

    print("Finished drawing")
    if not no_save:
        print("Saving result . . .")
        count = len([name for name in os.listdir("results") if name[:name.rfind("_")] == f"{video.name}_fbf"])
        video.save(f"results/{video.name}_fbf_{count}.mp4")


def frame_by_frame_VOD_with_tracklets(model, video, no_save=False):
    frame_predictions = [model.xywhcl(frame) for frame in video]
    print("Finished predicting")

    tracklets = []
    for i, frame_pred in enumerate(frame_predictions):
        for box in frame_pred:
            new_tracklet = Tracklet(i)
            new_tracklet.add_box(box, i)
            tracklets.append(new_tracklet)

    ts = TrackletSet(video, tracklets, model.num_to_class)    

    if not no_save:
        print("Saving result . . .")
        ts.draw_tracklets()
        count = len([name for name in os.listdir("results") if name[:name.rfind("_")] == f"{video.name}_fbf"])
        video.save(f"results/{video.name}_fbf_{count}.mp4")

    return ts


def frame_skipping(full_video, vod_method, model, n=1, extend_over_last_gap = True):
    new_frames = []
    kept_frame_indices = []

    #skip every n frames
    for i in range(0, full_video.num_of_frames, 1 + n):
        new_frames.append(full_video.frames[i])
        kept_frame_indices.append(i)
    print("Created new frame set")

    #create new video with skipped frame and run VOD
    frame_skipped_vid = Video(f"{full_video.path}/{full_video.name}_frame_skipped_n.{full_video.file_type}", init_empty=True)
    frame_skipped_vid.set_frames(new_frames, math.ceil(full_video.fps/(1 + n)))
    print(f"Reduced from {full_video.num_of_frames} to {len(new_frames)}")

    skipped_tracklet_set = vod_method(model = model, video = frame_skipped_vid)
    new_tracklets = []

    skipped_tracklet_set.play(1200)

    print("Interpolating boxes")
    #interpolate boxes over skipped frames
    for skipped_tracklet in skipped_tracklet_set:
        prev_box = skipped_tracklet.boxes[0]
        new_boxes = [prev_box]
        new_frame_indices = [kept_frame_indices[skipped_tracklet.start_frame]]

        for box_index in range(len(skipped_tracklet.boxes)):
            #starting box
            box = skipped_tracklet[box_index]

            #for each skipped frame
            for i in range(n):
                target_unskipped_index = kept_frame_indices[skipped_tracklet.start_frame + box_index] + i + 1
                if target_unskipped_index >= full_video.num_of_frames: break

                t = (i + 1)/(n + 1)
                interpolated_box = prev_box + t * (box - prev_box)
                interpolated_box[5] = round(interpolated_box[5])
                new_boxes.append(interpolated_box)
                new_frame_indices.append(target_unskipped_index)

            prev_box = box

        #create new tracklet with interpolated boxes
        unskipped_tracklet = Tracklet(skipped_tracklet.id)
        for box, frame_i in zip(new_boxes, frame_i):
            unskipped_tracklet.add_box(box, frame_i)
        
        new_tracklets.append(unskipped_tracklet)

    return TrackletSet(full_video, new_tracklets, model.num_to_class)