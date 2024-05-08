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
from yolov5.val import process_batch

from Video_utils import Video

NUM_OF_COLOURS = 8
colours = [colorsys.hsv_to_rgb(hue, 0.8, 1) for hue in np.linspace(0, 1, NUM_OF_COLOURS + 1)][:-1]
colours = [(round(255 * c[0]), round(255 * c[1]), round(255 * c[2])) for c in colours]

def annotate_image(im, prediction, num_to_label, num_to_colour, draw_labels=True, ids=None):
    """Draws a list xywhcl boxes onto a single image"""
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
    """Draws key and dict item in top left corner of an image"""
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
        """Creates two lists where counts[i] and totals[i] is the number of objects and the total number of objects seen at frame i"""
        counts = [0] * self.video.num_of_frames
        totals = [0] * self.video.num_of_frames

        for tracklet in self.tracklets:
            start_i, end_i = tracklet.start_frame, tracklet.end_frame

            for i in range(start_i, end_i + 1): counts[i] += 1
            for i in range(start_i, self.video.num_of_frames): totals[i] += 1

        return counts, totals
    
    def draw_tracklets(self, correct_id = None):
        """Draws all tracklets and counts/totals on the video, randomizes colours"""
        frames = self.video.frames
        counts, totals = self.count_per_frame()

        for i, tracklet in enumerate(self.tracklets):
            id = tracklet.id
            colour = colours[i%len(colours)]

            for frame_index, box in tracklet:
                if not correct_id is None:
                    colour = (19, 235, 76) if (frame_index, id) in correct_id else (232, 42, 21)

                annotate_image(frames[frame_index], [box], self.num_to_label, 
                               [colour] * len(self.num_to_label), ids=[id])
        
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


def frame_by_frame_VOD(model, video, no_save=False):
    """Perfoms VOD by running the detector on each frame independently"""
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
    """Same as fbf_VOD but returns results as a TrackletSet rather than directly drawing on the video"""
    frame_predictions = [model.xywhcl(frame) for frame in video]
    print("Finished predicting")

    tracklets = []
    id_counter = 0
    for i, frame_pred in enumerate(frame_predictions):
        for box in frame_pred:
            new_tracklet = Tracklet(id_counter)
            new_tracklet.add_box(box, i)
            tracklets.append(new_tracklet)
            id_counter += 1

    ts = TrackletSet(video, tracklets, model.num_to_class)    

    if not no_save: save_VOD(ts, "fbf")
    return ts


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
    print("Created new frame set")

    #create new video with skipped frame and run VOD
    frame_skipped_vid = Video(f"{full_video.path}/{full_video.name}_frame_skipped_n.{full_video.file_type}", init_empty=True)
    frame_skipped_vid.set_frames(new_frames, math.ceil(full_video.fps/(1 + n)))
    print(f"Reduced from {full_video.num_of_frames} to {len(new_frames)}")

    skipped_tracklet_set = vod_method(model = model, video = frame_skipped_vid, no_save = True, **vod_kwargs)
    new_tracklets = []

    print("Interpolating boxes")
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


def correct_preds(gt, preds, iou=0.5):
    """Determines which predictions are correct and returns the matching
        Arguments:
            gt: list of ground truth np.array boxes in the form xywhcl
            preds: list of predicted np.array boxes in the form xywhcl
        Returns:
            correct: array of len(preds) where the ith element is true if pred[i] has a match
            match_indices: Nx2 array of box index matchings (gt box, pred box)"""
    gt = np.array(gt)
    preds = np.array(preds)
    correct = np.zeros((preds.shape[0],)).astype(bool)

    if gt.shape[0] == 0 or preds.shape[0] == 0: return correct, None

    iou_mat = iou_matrix(gt, preds)
    correct_label = gt[:, 5:6] == preds[:, 5].T

    match_indices = np.argwhere((iou_mat >= iou) & correct_label)
    if match_indices.shape[0]:
        iou_of_matches = iou_mat[match_indices[:, 0], match_indices[:, 1]]
        match_indices = match_indices[iou_of_matches.argsort()[::-1]]
        match_indices = match_indices[np.unique(match_indices[:, 1], return_index=True)[1]]
        match_indices = match_indices[np.unique(match_indices[:, 0], return_index=True)[1]]

        correct[match_indices[:, 1]] = True

    return correct, match_indices


def trackletSet_frame_by_frame(tracklets):
    """Seperates a tracklet set into a list of boxes and ids for each frame"""
    boxes_by_frame = [[] for i in range(tracklets.video.num_of_frames)] 
    ids_by_frame = [[] for i in range(tracklets.video.num_of_frames)] 

    for track in tracklets:
        for frame_index, box in track:
            boxes_by_frame[frame_index].append(box)
            ids_by_frame[frame_index].append(track.id)

    return boxes_by_frame, ids_by_frame


def single_vid_metrics(gt_tracklets, pred_tracklets, return_correct_ids = False, return_components = False):
    """Calculates multiple metrics using a set of ground truth tracklets
        Arguments:
            gt_tracklets: tracklet set representing the true labels
            pred_tracklets: tracklet set from any VOD method
            return_correct_ids: if true retruns two lists of items of the form (frame_index, tracklet_id) for correct preds
        Returns:
            precision, recall, multi-object tracking accuray, multi-object tracking precision, 
            mostly tracked, partially tracked, mostly lost"""
    tp, fp, fn, id_switches, dist, total_matches = 0, 0, 0, 0, 0, 0
    gt_correct_ids, pred_correct_ids = [], []
    gt_detection_lifetimes = {gt_track.id:0 for gt_track in gt_tracklets}

    #get boxes for each frame
    gt_boxes_by_frame, gt_ids_by_frame = trackletSet_frame_by_frame(gt_tracklets)
    preds_by_frame, pred_ids_by_frame = trackletSet_frame_by_frame(pred_tracklets)

    if len(gt_boxes_by_frame) != len(preds_by_frame):
        raise ValueError("Tracklet set videos have a different number of frames")

    #Loop over predictions in each frame to get the components of each metric
    corresponding_id = {}
    for i in range(len(gt_boxes_by_frame)):
        correct, matches = correct_preds(gt_boxes_by_frame[i], preds_by_frame[i])
        num_correct = np.sum(correct)

        tp += num_correct
        fp += correct.shape[0] - num_correct
        fn += len(gt_boxes_by_frame[i]) - num_correct

        if not matches is None:
            for gt_index, pred_index in matches:
                gt_tracklet_id = gt_ids_by_frame[i][gt_index]
                pred_tracklet_id = pred_ids_by_frame[i][pred_index]

                if not gt_tracklet_id in corresponding_id:
                    #first time tracklet has a match
                    corresponding_id[gt_tracklet_id] = pred_tracklet_id
                else:
                    #tracklet has been matched before
                    if corresponding_id[gt_tracklet_id] != pred_tracklet_id:
                        #id switch has occured
                        #print(f"frame: {i} - id switch: gt {gt_index}, original match {corresponding_id[gt_tracklet_id]}, new match {pred_tracklet_id}")
                        id_switches += 1
                        corresponding_id[gt_tracklet_id] = pred_tracklet_id

                dist += 1 - iou(gt_boxes_by_frame[i][gt_index], preds_by_frame[i][pred_index])
                total_matches += 1

                gt_detection_lifetimes[gt_tracklet_id] += 1

                #get ids of matched gts and preds for colouring later
                if return_correct_ids:
                    gt_correct_ids.append((i, gt_ids_by_frame[i][gt_index]))
                    pred_correct_ids.append((i, pred_ids_by_frame[i][pred_index]))  


    #calculate metrics from equations
    p = tp / (tp + fp)
    r = tp / (tp + fn)

    gt_total = sum([len(gt_boxes) for gt_boxes in gt_boxes_by_frame])
    mota = 1 - ((fn + fp + id_switches)/gt_total)

    motp = dist / total_matches

    #calculate mostly tracked, partially tracked and mostly lost based on gt tracklet lifetime
    gt_detection_lifetimes = [gt_detection_lifetimes[gt_track.id]/len(gt_track.frame_indexes) for gt_track in gt_tracklets]
    mt, pt, ml = 0, 0, 0
    for duration in gt_detection_lifetimes:
        if duration >= 0.8:
            mt += 1
        elif duration <= 0.2:
            ml += 1
        else:
            pt += 1

    if return_components: return tp, fp, fn, id_switches, gt_total, dist, total_matches, mt, pt, ml

    mt /= len(gt_detection_lifetimes)
    pt /= len(gt_detection_lifetimes)
    ml /= len(gt_detection_lifetimes)

    if return_correct_ids: return p, r, mota, motp, mt, pt, ml, gt_correct_ids, pred_correct_ids
    return p, r, mota, motp, mt, pt, ml
    

def mutiple_vid_metrics(gt_tracklet_sets, pred_tracklet_sets):
    """Computes the metrics over a set of videos"""
    #tp, fp, fn, id_switches, gt_total, dist, total_matches, mt, pt, ml
    components = np.zeros((10,))

    for gt_tracklet_set, pred_tracklet_set in zip(gt_tracklet_sets, pred_tracklet_sets):
        components +=  np.array(*[single_vid_metrics(gt_tracklet_set, pred_tracklet_set, False, True)])

    p = components[0]/(components[0] + components[1])
    r = components[0]/(components[0] + components[2])
    mota = 1 - ((components[1] + components[2] + components[3])/components[4])
    motp = components[5]/components[6]
    mt = components[7]/(components[7] + components[8] + components[9])
    pt = components[8]/(components[7] + components[8] + components[9])   
    ml = components[9]/(components[7] + components[8] + components[9])

    return p, r, mota, motp, mt, pt, ml