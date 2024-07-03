import time
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from python.utils.VOD_utils import TrackletSet, save_VOD, silence
from Sort_utils import SortTracklet,det_tracklet_matches, handle_successful_match, cleanup_dead_tracklets

"""Bewley, A., Ge, Z., Ott, L., Ramos, F., & Upcroft, B. (2016, September). 
   Simple online and realtime tracking. In 2016 IEEE international conference on image processing
   (ICIP) (pp. 3464-3468). IEEE."""
    
@silence
def SORT(model, video, iou_min = 0.5, t_lost = 1, probation_timer = 3, min_hits = 5, greedy_assoc = False, no_save = False, kf_tracklets=False):
    start_time = time.time()
    active_tracklets = []
    deceased_tracklets = []
    id_counter = 0

    print("Starting SORT")
    for i, frame in tqdm(list(enumerate(video)), bar_format="{l_bar}{bar:30}{r_bar}"):
        frame_pred = model.xywhcl(frame)
        tracklet_predictions = [t.kalman_predict() for t in active_tracklets]

        if kf_tracklets:
            for t, state_est in zip(active_tracklets, tracklet_predictions):
                t.kalman_state_tracklet.add_box(state_est, i, frame.shape)
                
        
        tracklet_indices, detection_indices, unassigned_track_indices, unassigned_det_indices = det_tracklet_matches(tracklet_predictions, frame_pred, iou_min, greedy_assoc)
       
        #successful match, update kf preds with det measurements
        for track_i, detect_i in zip(tracklet_indices, detection_indices):
            assigned_tracklet = active_tracklets[track_i]
            assigned_det = frame_pred[detect_i]
            handle_successful_match(assigned_tracklet, assigned_det, i, frame.shape)
                
        #tracklet had no match, update with just kf pred
        for track_i in unassigned_track_indices:
            track = active_tracklets[track_i]
            track.add_box(tracklet_predictions[track_i], i, frame.shape)
            track.miss_streak += 1

        #detection had no match, create new tracklet
        for det_i in unassigned_det_indices:
            new_tracklet = SortTracklet(id_counter, frame_pred[det_i], i, probation_timer)
            id_counter += 1
            active_tracklets.append(new_tracklet)

        cleanup_dead_tracklets(active_tracklets, deceased_tracklets, unassigned_track_indices, t_lost)
        
    #remove tracklets that did not meet the requirement minimum number of hits
    combined_tracklets = deceased_tracklets + active_tracklets
    for track_i in range(len(combined_tracklets) - 1, -1, -1):
        if combined_tracklets[track_i].hits < min_hits:
            combined_tracklets.pop(track_i)

    ts = TrackletSet(video, combined_tracklets, model.num_to_class)

    duration = round((time.time() - start_time)/60, 2)
    print(f"Finished SORT in {duration}mins")
    print(f"{id_counter + 1} tracklets created")
    print(f"{len(combined_tracklets)} tracklets kept")
    if not no_save: save_VOD(ts, "SORT")
    return ts


