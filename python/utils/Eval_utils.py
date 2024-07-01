import numpy as np
import os
import sys

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_dir)
from TrackEval.scripts.run_mot_challenge import main

from python.utils.VOD_utils import iou_matrix, trackletSet_frame_by_frame, iou, Tracklet


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

    motp = dist / total_matches if not total_matches == 0 else 0

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
    motp = components[5]/components[6] if not components[6] == 0 else 0
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
    print(f"MT: {round(mt, 3)}, PT: {round(pt, 3)}, ML: {round(ml, 3)}")
    print(f"IDSW: {round(id_switchs, 3)}, FM: {frag}")
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
    

if __name__ == "__main__":
    track_eval("SORT", "init_test")