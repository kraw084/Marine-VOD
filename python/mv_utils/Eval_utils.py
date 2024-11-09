import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt    

from TrackEval.scripts.run_mot_challenge import main

from .VOD_utils import iou_matrix, trackletSet_frame_by_frame, iou


def correct_preds(gt, preds, iou_th=0.5):
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

    match_indices = np.argwhere((iou_mat >= iou_th) & correct_label)
    if match_indices.shape[0]:
        iou_of_matches = iou_mat[match_indices[:, 0], match_indices[:, 1]]
        match_indices = match_indices[iou_of_matches.argsort()[::-1]]
        match_indices = match_indices[np.unique(match_indices[:, 1], return_index=True)[1]]
        match_indices = match_indices[np.unique(match_indices[:, 0], return_index=True)[1]]

        correct[match_indices[:, 1]] = True

    return correct, match_indices


def correct_ids(gt_tracklets, pred_tracklets, match_iou=0.5):
    """Finds the ids and frame indices of all gt and pred boxes that get a successful match"""
    gt_correct_ids, pred_correct_ids = [], []
    gt_boxes_by_frame, gt_ids_by_frame = trackletSet_frame_by_frame(gt_tracklets)
    preds_by_frame, pred_ids_by_frame = trackletSet_frame_by_frame(pred_tracklets)

    for i in range(len(gt_boxes_by_frame)):
        _, matches = correct_preds(gt_boxes_by_frame[i], preds_by_frame[i], iou_th=match_iou)

        if not matches is None:
            for gt_index, pred_index in matches:
                gt_correct_ids.append((i, gt_ids_by_frame[i][gt_index]))
                pred_correct_ids.append((i, pred_ids_by_frame[i][pred_index]))

    return gt_correct_ids, pred_correct_ids


def single_vid_metrics(gt_tracklets, pred_tracklets, match_iou = 0.5, return_correct_ids = False, return_components = False):
    """Calculates multiple metrics using a set of ground truth tracklets
        Arguments:
            gt_tracklets: tracklet set representing the true labels
            pred_tracklets: tracklet set from any VOD method
            return_correct_ids: if true retruns two lists of items of the form (frame_index, tracklet_id) for correct preds
        Returns:
            precision, recall, multi-object tracking accuray, multi-object tracking precision, 
            mostly tracked, partially tracked, mostly lost"""
    tp, fp, fn, id_switches, dist, total_matches, frag = 0, 0, 0, 0, 0, 0, 0
    gt_correct_ids, pred_correct_ids = [], []
    gt_detection_lifetimes = {gt_track.id:0 for gt_track in gt_tracklets}
    gt_is_tracked = {gt_track.id:False for gt_track in gt_tracklets}

    #get boxes for each frame
    gt_boxes_by_frame, gt_ids_by_frame = trackletSet_frame_by_frame(gt_tracklets)
    preds_by_frame, pred_ids_by_frame = trackletSet_frame_by_frame(pred_tracklets)

    if len(gt_boxes_by_frame) != len(preds_by_frame):
        raise ValueError("Tracklet set videos have a different number of frames")

    #Loop over predictions in each frame to get the components of each metric
    corresponding_id = {}
    for i in range(len(gt_boxes_by_frame)):
        correct, matches = correct_preds(gt_boxes_by_frame[i], preds_by_frame[i], iou=match_iou)
        num_correct = np.sum(correct)

        tp += num_correct
        fp += correct.shape[0] - num_correct
        fn += len(gt_boxes_by_frame[i]) - num_correct

        new_gt_is_tracked = {gt_track.id:False for gt_track in gt_tracklets}

        if not matches is None:
            for gt_index, pred_index in matches:
                gt_tracklet_id = gt_ids_by_frame[i][gt_index]
                pred_tracklet_id = pred_ids_by_frame[i][pred_index]

                new_gt_is_tracked[gt_tracklet_id] = True

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

                dist += iou(gt_boxes_by_frame[i][gt_index], preds_by_frame[i][pred_index])
                total_matches += 1

                gt_detection_lifetimes[gt_tracklet_id] += 1

                #get ids of matched gts and preds for colouring later
                if return_correct_ids:
                    gt_correct_ids.append((i, gt_ids_by_frame[i][gt_index]))
                    pred_correct_ids.append((i, pred_ids_by_frame[i][pred_index]))  

        for gt_tracklet_id in [gt_track.id for gt_track in gt_tracklets]:
            prev_tracked = gt_is_tracked[gt_tracklet_id]
            currently_tracked = new_gt_is_tracked[gt_tracklet_id]
            if prev_tracked and not currently_tracked:
                frag += 1

        gt_is_tracked = new_gt_is_tracked

    #calculate metrics from equations
    p = tp / (tp + fp) if not (tp + fp) == 0 else 1
    r = tp / (tp + fn) if not (tp + fn) == 0 else 1

    gt_total = sum([len(gt_boxes) for gt_boxes in gt_boxes_by_frame])
    mota = 1 - ((fn + fp + id_switches)/(gt_total + 1E-9))

    motp = dist / total_matches if not total_matches == 0 else 1

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

    if return_components: return np.array([tp, fp, fn, id_switches, gt_total, dist, total_matches, mt, pt, ml, frag])

    mt /= len(gt_detection_lifetimes)
    pt /= len(gt_detection_lifetimes)
    ml /= len(gt_detection_lifetimes)

    if return_correct_ids: return p, r, mota, motp, mt, pt, ml, id_switches, frag, gt_correct_ids, pred_correct_ids
    return p, r, mota, motp, mt, pt, ml, id_switches, frag
    

def metrics_from_components(components):
    """Computes the metrics from component variables, used to calculate metrics on a set of videos"""
    #components are in the form [tp, fp, fn, id_switches, gt_total, dist, total_matches, mt, pt, ml, frag]
    p = components[0]/(components[0] + components[1]) if not (components[0] + components[1]) == 0 else 1
    r = components[0]/(components[0] + components[2]) if not (components[0] + components[2]) == 0 else 1
    mota = 1 - ((components[1] + components[2] + components[3])/components[4] + 1E-9)
    motp = components[5]/components[6] if not components[6] == 0 else 1
    mt = components[7]/(components[7] + components[8] + components[9]) 
    pt = components[8]/(components[7] + components[8] + components[9])   
    ml = components[9]/(components[7] + components[8] + components[9])
    frag = components[10]
    id_switches = components[3]

    return p, r, mota, motp, mt, pt, ml, id_switches, frag


def print_metrics(p, r, mota, motp, mt, pt, ml, id_switchs, frag):
    print("-------------------------------------")
    print(f"P: {round(p, 3)}, R: {round(r, 3)}")
    print(f"MOTA: {round(mota, 3)}, MOTP: {round(motp, 3)}")
    print(f"MT: {mt}, PT: {pt}, ML: {ml}")
    print(f"IDSW: {id_switchs}, FM: {frag}")
    print("-------------------------------------")


def save_track_result(trackletSet, seq_name, tracker_name, dataset_name, sub_name=""):
    #setup directories if they dont exist
    if not os.path.isdir(f"TrackEval_results/{dataset_name}"):
        os.mkdir(f"TrackEval_results/{dataset_name}")

    if not os.path.isdir(f"TrackEval_results/{dataset_name}/{tracker_name}"):
        os.mkdir(f"TrackEval_results/{dataset_name}/{tracker_name}")

    if sub_name and not os.path.isdir(f"TrackEval_results/{dataset_name}/{tracker_name}/{sub_name}"):
        os.mkdir(f"TrackEval_results/{dataset_name}/{tracker_name}/{sub_name}")

    target_dir = f"TrackEval_results/{dataset_name}/{tracker_name}/{sub_name}" if sub_name else f"TrackEval_results/{tracker_name}"

    f = open(target_dir + f"/{seq_name}.txt", "w")

    for t in trackletSet:
        id = t.id
        for frame_i, box in t:
            x = box[0] - box[2]/2
            y = box[1] - box[3]/2
            w = box[2]
            h = box[3]
            conf = box[4]
            to_write = ", ".join(map(str, [frame_i + 1, id, x, y, w, h, conf, -1, -1, -1]))
            f.write(to_write + "\n")

    f.close()
    print("Tracklet results saved")


def track_eval(tracker_name, sub_name, dataset_name = "MOT17", split = "train", metrics = None):
    if metrics is None:
        metrics = ["HOTA", "CLEAR"]

    if dataset_name == "MOT17":
        bench = "MOT17"
        preproc = "True"
        classes = ["pedestrian"]
        gt_dir = "TrackEval/data/gt/mot_challenge/"
    elif dataset_name == "BrackishMOT":
        bench = "BrackishMOT"
        preproc = "False"
        classes = ["Jellyfish", "Fish", "Crab", "Shrimp", "Starfish", "Smallfish", "UNKNOWN"]
        gt_dir = "BrackishMOT/"
    elif dataset_name == "MOT17-half":
        bench = "MOT17-half"
        preproc = "True"
        classes = ["pedestrian"]
        gt_dir = "TrackEval/data/gt/mot_challenge/"
    elif dataset_name == "UrchinNZ":
        bench = "UrchinsNZ"
        preproc = "False"
        classes = ["pedestrian"]
        gt_dir = "TrackEval/data/gt/UrchinsNZ/"

    tracker_dir = "TrackEval_results"

    main(METRICS = metrics,
         BENCHMARK = bench,
         SPLIT_TO_EVAL = split,
         TRACKERS_TO_EVAL = [tracker_name],
         CLASSES_TO_EVAL = classes,
         DO_PREPROC = preproc,
         THRESHOLD = "0.5",

         GT_FOLDER = gt_dir,
         TRACKERS_FOLDER =  tracker_dir,
         TRACKER_SUB_FOLDER = sub_name,
         OUTPUT_SUB_FOLDER = sub_name,

         PRINT_CONFIG = "False",
         TIME_PROGRESS = "False"
         )
    
    
class Evaluator:
    def __init__(self, tracker_name, match_iou = 0.5):
        self.tracker_name = tracker_name
        self.match_iou = match_iou
        
        self.tp, self.fp, self.fn = 0, 0, 0
        self.mt, self.pt, self.ml = 0, 0, 0
        self.id_switches, self.dist, self.total_matches, self.frag = 0, 0, 0, 0    
        self.gt_total = 0
        
        
    def set_tracklets(self, gt_tracklets, pred_tracklets):
        """Sets the gt and pred tracklets to be evaluated"""
        self.pred_tracklets = pred_tracklets
        self.gt_tracklets = gt_tracklets
                
        self.gt_detection_lifetimes = {gt_track.id:0 for gt_track in gt_tracklets}
        self.gt_is_tracked = {gt_track.id:False for gt_track in gt_tracklets}
        self.corresponding_id = {}

        self.match_matrix = np.zeros((len(gt_tracklets), len(pred_tracklets)))
        
        self.gt_boxes_by_frame, self.gt_ids_by_frame = trackletSet_frame_by_frame(gt_tracklets)
        self.preds_by_frame, self.pred_ids_by_frame = trackletSet_frame_by_frame(pred_tracklets)
    
        
    def eval_frame(self, frame_index):
        """Calculates metric components for a single frame"""
        gt_boxes = self.gt_boxes_by_frame[frame_index]
        pred_boxes = self.preds_by_frame[frame_index]
        
        correct, matches = correct_preds(gt_boxes, pred_boxes, iou_th=self.match_iou)
        num_correct = np.sum(correct)

        self.tp += num_correct
        self.fp += correct.shape[0] - num_correct
        self.fn += len(self.gt_boxes_by_frame[frame_index]) - num_correct
        
        self.gt_total += len(gt_boxes)

        new_gt_is_tracked = {gt_track.id:False for gt_track in self.gt_tracklets}
        
        if not matches is None:
            for gt_index, pred_index in matches:
                gt_tracklet_id = self.gt_ids_by_frame[frame_index][gt_index]
                pred_tracklet_id = self.pred_ids_by_frame[frame_index][pred_index]

                self.match_matrix[self.gt_tracklets.id_to_index[gt_tracklet_id], 
                                  self.pred_tracklets.id_to_index[pred_tracklet_id]] += 1

                new_gt_is_tracked[gt_tracklet_id] = True

                if not gt_tracklet_id in self.corresponding_id:
                    #first time tracklet has a match
                    self.corresponding_id[gt_tracklet_id] = pred_tracklet_id
                else:
                    #tracklet has been matched before
                    if self.corresponding_id[gt_tracklet_id] != pred_tracklet_id:
                        #id switch has occured
                        #print(f"frame: {i} - id switch: gt {gt_index}, original match {corresponding_id[gt_tracklet_id]}, new match {pred_tracklet_id}")
                        self.id_switches += 1
                        self.corresponding_id[gt_tracklet_id] = pred_tracklet_id

                self.dist += iou(gt_boxes[gt_index], pred_boxes[pred_index])
                self.total_matches += 1

                self.gt_detection_lifetimes[gt_tracklet_id] += 1

        for gt_tracklet_id in [gt_track.id for gt_track in self.gt_tracklets]:
            prev_tracked = self.gt_is_tracked[gt_tracklet_id]
            currently_tracked = new_gt_is_tracked[gt_tracklet_id]
            if prev_tracked and not currently_tracked:
                self.frag += 1

        self.gt_is_tracked = new_gt_is_tracked


    def eval_video(self, loading_bar=False):
        """Evaluates an entire video"""
        if loading_bar:
            print("Evaluating:")
            for i in tqdm(range(len(self.gt_boxes_by_frame)), bar_format="{l_bar}{bar:30}{r_bar}"):
                self.eval_frame(i)
        else:
            for i in range(len(self.gt_boxes_by_frame)):
                self.eval_frame(i)
            
        #calculate mostly tracked, partially tracked and mostly lost based on gt tracklet lifetime
        gt_lifetimes_props = self.compute_gt_track_status()
        for duration in gt_lifetimes_props:
            if duration >= 0.8:
                self.mt += 1
            elif duration <= 0.2:
                self.ml += 1
            else:
                self.pt += 1
   
                
    def compute_gt_track_status(self):
        """Computes the proportion of frames that a gt tracklet had a match"""
        return [self.gt_detection_lifetimes[gt_track.id]/len(gt_track.frame_indexes) for gt_track in self.gt_tracklets]


    def id_of_best_match(self):
        """Returns a list of the pred tracklet id that each gt had the most number of hits with"""
        return [self.pred_tracklets.tracklets[i].id for i in np.argmax(self.match_matrix, axis=1)]


    def compute_metrics(self):
        """Computes metrics from the components"""
        p = self.tp / (self.tp + self.fp) if not (self.tp + self.fp) == 0 else 1
        r = self.tp / (self.tp + self.fn) if not (self.tp + self.fn) == 0 else 1

        mota = 1 - ((self.fn + self.fp + self.id_switches)/(self.gt_total + 1E-9))

        motp = self.dist / self.total_matches if not self.total_matches == 0 else 1

        return p, r, mota, motp, self.mt, self.pt, self.ml, self.id_switches, self.frag
    
    
    def print_metrics(self, print_vid_name = False):
        """Calculates and prints metrics"""
        p, r, mota, motp, mt, pt, ml, id_switchs, frag = self.compute_metrics()
        print(f"Tracker: {self.tracker_name}{f' - video: {self.pred_tracklets.video.name}' if print_vid_name else ''}")
        print(f"P: {round(p, 3)}, R: {round(r, 3)}")
        print(f"MOTA: {round(mota, 3)}, MOTP: {round(motp, 3)}")
        print(f"MT: {mt}, PT: {pt}, ML: {ml}")
        print(f"IDSW: {id_switchs}, FM: {frag}")
        print("")
        
        
    def metrics_fbf(self):
        """Generator to get metrics at every frame (does not work for mt, pt and ml)"""
        for i in range(len(self.gt_boxes_by_frame)):
            self.eval_frame(i)
            yield self.compute_metrics()
   
   
