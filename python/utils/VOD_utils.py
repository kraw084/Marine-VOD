import os
import sys
import torch
import cv2
import numpy as np
import colorsys
import math
import contextlib

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_dir)
from yolov5.utils.metrics import box_iou, bbox_iou

from Video_utils import Video
from Config import Config

NUM_OF_COLOURS = 8
colours = [colorsys.hsv_to_rgb(hue, 0.8, 1) for hue in np.linspace(0, 1, NUM_OF_COLOURS + 1)][:-1]
colours = [(round(255 * c[0]), round(255 * c[1]), round(255 * c[2])) for c in colours]


def silence(func):
    def wrapper(*args,  silence=False, **kwargs):
        if silence:
            with contextlib.redirect_stdout(open(os.devnull, 'w')):
                result = func(*args, **kwargs)
                return result
        else:
            return func(*args, **kwargs)

    return wrapper


def annotate_image(im, prediction, num_to_label, num_to_colour, ids=None):
    """Draws a list xywhcl boxes onto a single image"""
    for id, box in zip(ids if not ids is None else [None for _ in range(len(prediction))], prediction):
        label = num_to_label[int(box[5])]
        colour = num_to_colour[int(box[5])]
        draw_box(im, box, label, colour, id)


def draw_box(im, box, label, colour, id = None):
    """Draws a single box onto an image"""
    box_thickness = 3
    text_thickness = math.floor(3 * Config.label_font_thickness)
    font_size = 1 * Config.label_font_size
    top_left = (int(box[0]) - int(box[2])//2, int(box[1]) - int(box[3])//2)
    bottom_right = (top_left[0] + int(box[2]), top_left[1] + int(box[3]))
    cv2.rectangle(im, top_left, bottom_right, colour, box_thickness)

    if not Config.minimal_labels:
        label = f"{f'{id}. ' if not id is None else ''}{label} - {float(box[4]):.2f}"
    else:
        label = f"{f'{id} -' if not id is None else ''} {float(box[4]):.2f}"

    if Config.draw_labels:
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_size, text_thickness)[0]
        text_box_top_left = (top_left[0], top_left[1] - (text_size[1] if not Config.labels_in_box else 0))
        text_box_bottom_right = (top_left[0] + text_size[0], top_left[1] + (text_size[1] if Config.labels_in_box else 0))
        cv2.rectangle(im, text_box_top_left, text_box_bottom_right, colour, -1)
        cv2.putText(im, label, (top_left[0], top_left[1] + (text_size[1] if Config.labels_in_box else 0)), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), text_thickness - 1, cv2.LINE_AA)


def draw_data(im, data_dict):
    """Draws key and dict item in top left corner of an image"""
    thickness = 2
    font_size = 1.5
    start_point = (40, 60)
    gap = 70

    for i, data in enumerate(data_dict):
        text = f"{data}: {data_dict[data]}"
        im = cv2.putText(im, text, (start_point[0], start_point[1] + i * gap), cv2.FONT_HERSHEY_SIMPLEX, font_size, Config.data_text_colour, thickness, cv2.LINE_AA)


def draw_detections(vid, model):
    for im in vid.frames:
        dets = model.xywhcl(im)
        annotate_image(im, dets, model.num_to_class, [(255, 255, 255)] * len(model.num_to_class))


class Tracklet:
    def __init__(self, id):
        self.boxes = []
        self.frame_indexes = []
        self.id = id
        self.start_frame = None
        self.end_frame = -1

    def add_box(self, box, frame_index, im_shape=None):
        if not im_shape is None:
            box = self.clip_box(box, im_shape[1], im_shape[0])

        self.boxes.append(box)
        self.frame_indexes.append(frame_index)

        if self.start_frame is None: self.start_frame = frame_index

        if frame_index < self.start_frame: self.start_frame = frame_index
        if frame_index > self.end_frame: self.end_frame = frame_index
        
    def tracklet_length(self):
        return len(self.boxes)


    def clip_box(self, box, im_w, im_h):
        x_c, y_c, box_w, box_h = box[:4]
        x_min, y_min = x_c - box_w/2, y_c - box_h/2
        x_max, y_max = x_c + box_w/2, y_c + box_h/2

        if x_min < 0: x_min = 0
        if y_min < 0: y_min = 0
        if x_max > im_w: x_max = im_w
        if y_max > im_h: y_max = im_h

        x_c_new = (x_max + x_min)/2
        y_c_new = (y_max + y_min)/2
        box_w_new = (x_max - x_min)
        box_h_new = (y_max - y_min)

        return np.array([x_c_new, y_c_new, box_w_new, box_h_new, box[4], box[5]])

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
        """Creates two lists where counts[i] and totals[i] is the number of objects and the total number of objects seen at frame i"""
        counts = [0] * self.video.num_of_frames
        totals = [0] * self.video.num_of_frames

        for tracklet in self.tracklets:
            start_i, end_i = tracklet.start_frame, tracklet.end_frame

            for i in range(start_i, end_i + 1): counts[i] += 1
            for i in range(start_i, self.video.num_of_frames): totals[i] += 1

        return counts, totals
    
    def draw_tracklets(self, correct_id = None, annotate_data=True):
        """Draws all tracklets and counts/totals on the video, randomizes colours"""
        frames = self.video.frames
        counts, totals = self.count_per_frame()

        for i, tracklet in enumerate(self.tracklets):
            id = tracklet.id
            colour = colours[i%len(colours)]

            for frame_index, box in tracklet:
                if not correct_id is None:
                    colour = (19, 235, 76) if (frame_index, id) in correct_id else (232, 42, 21)
                draw_box(frames[frame_index], box, self.num_to_label[int(box[5])], colour, id)

        if annotate_data:
            for i, frame in enumerate(frames):
                draw_data(frame, {"Objects":counts[i], "Total":totals[i]})

    def __iter__(self):
        self.i = 0
        return self
    
    def __next__(self):
        if self.i >= len(self.tracklets): raise StopIteration
        val = self.tracklets[self.i]
        self.i += 1
        return val


def draw_single_tracklet(video, tracklet, label, colour):
    for frame_index, box in tracklet:
        draw_box(video.frames[frame_index], box, label, colour, tracklet.id)


def save_VOD(ts, VOD_method_name):
    print("Saving result . . .")
    ts.draw_tracklets()
    count = len([name for name in os.listdir("results") if name[:name.rfind("_")] == f"{ts.video.name}_{VOD_method_name}"])
    ts.video.save(f"results/{ts.video.name}_{VOD_method_name}_{count}.mp4")


def xywhTOxyxy(x, y, w, h):
    """Converts an xywh box to xyxy"""
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


def round_box(box):
    """rounds all values of a box (xywhcl) to the nearest integer (except confidence)"""
    rounded_box = np.rint(box)
    rounded_box[4] = box[4]
    return rounded_box

@silence
def frame_skipping(full_video, vod_method, model, n=1, **vod_kwargs):
    """Used to speed up VOD methods by skipping frames. Boxes are interpolated to fit the orignal video.
        arguments:
            full_video: the full length video to perfom VOD on
            vod_method: the VOD function to be used
            model: single image detector to be passed to vod_method
            n: number of frames to skip over
            **vod_kwargs: key word args to pass to vod_method
        returns:
            TrackletSet of interpolated boxes"""
    
    new_frames = []
    kept_frame_indices = []

    #skip every n frames
    for i in range(0, full_video.num_of_frames, 1 + n):
        new_frames.append(full_video.frames[i])
        kept_frame_indices.append(i)

    #create new video with skipped frame and run VOD
    frame_skipped_vid = Video(f"{full_video.path}/{full_video.name}_frame_skipped_n.{full_video.file_type}", init_empty=True)
    frame_skipped_vid.set_frames(new_frames, math.ceil(full_video.fps/(1 + n)))
    print(f"Reduced from {full_video.num_of_frames} to {len(new_frames)}")

    skipped_tracklet_set = vod_method(model = model, video = frame_skipped_vid, no_save = True, **vod_kwargs)
    new_tracklets = []

    #interpolate boxes over skipped frames
    for skipped_tracklet in skipped_tracklet_set:
        prev_box = skipped_tracklet.boxes[0]
        new_boxes = [prev_box]
        new_frame_indices = [kept_frame_indices[skipped_tracklet.start_frame]]

        for box_index in range(1, len(skipped_tracklet.boxes)):
            #starting box
            box = skipped_tracklet.boxes[box_index]
            frame_index = skipped_tracklet.frame_indexes[box_index]

            #for each skipped frame
            for i in range(n):
                target_unskipped_index = kept_frame_indices[frame_index - 1] + i + 1
                if target_unskipped_index >= full_video.num_of_frames: break

                t = (i + 1)/(n + 1)
                interpolated_box = prev_box + t * (box - prev_box)
                interpolated_box = round_box(interpolated_box)
                new_boxes.append(interpolated_box)
                new_frame_indices.append(target_unskipped_index)

            new_boxes.append(box)
            new_frame_indices.append(kept_frame_indices[frame_index])
            prev_box = box

        #create new tracklet with interpolated boxes
        unskipped_tracklet = Tracklet(skipped_tracklet.id)
        for box, frame_i in zip(new_boxes, new_frame_indices):
            unskipped_tracklet.add_box(box, frame_i)
        
        new_tracklets.append(unskipped_tracklet)

    print("Finish frame skipping reconstruction")
    return TrackletSet(full_video, new_tracklets, model.num_to_class)


def trackletSet_frame_by_frame(tracklets):
    """Seperates a tracklet set into a list of boxes and ids for each frame"""
    boxes_by_frame = [[] for i in range(tracklets.video.num_of_frames)] 
    ids_by_frame = [[] for i in range(tracklets.video.num_of_frames)] 

    for track in tracklets:
        for frame_index, box in track:
            boxes_by_frame[frame_index].append(box)
            ids_by_frame[frame_index].append(track.id)

    return boxes_by_frame, ids_by_frame


