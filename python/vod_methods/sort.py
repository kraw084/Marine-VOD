import numpy as np
import math
from filterpy.kalman import KalmanFilter
import time
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

from ..utils.VOD_utils import Tracklet, iou_matrix, TrackletSet, save_VOD, silence, correct_preds, draw_single_tracklet

"""Bewley, A., Ge, Z., Ott, L., Ramos, F., & Upcroft, B. (2016, September). 
   Simple online and realtime tracking. In 2016 IEEE international conference on image processing
   (ICIP) (pp. 3464-3468). IEEE."""


def box_to_state(box):
    """Takes a box [x, y, w, h] and converts to to the kalman state [x, y, s, r]"""
    state = np.zeros((4, 1), np.float32)
    state[0] = box[0]
    state[1] = box[1]
    state[2] = box[2] * box[3]
    state[3] = box[2] / box[3]

    return state


def state_to_box(state):
    """Takes a kalman state [x, y, s, r] and converts to to the kalman state [x, y, w, h]"""
    state = np.reshape(state, (4,))
    box = np.zeros((4,))
    box[0] = state[0]
    box[1] = state[1]
    box[2] = math.sqrt(state[2] * state[3])
    box[3] = state[2] / box[2]

    return np.rint(box)


class KalmanTracker():
    def __init__(self, initial_box):
        self.kf = KalmanFilter(7, 4) 
        #state is [x, y, s, r, x_v, y_v, s_v]
        #measurement in [x, y, s, r]

        #State transition matrix
        self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0],
                              [0, 1, 0, 0, 0, 1, 0],
                              [0, 0, 1, 0, 0, 0, 1],
                              [0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 1]], np.float32)
        #Measurment matrix
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0]], np.float32)
        
        self.kf.R[2:,2:] *= 10.0
        self.kf.P[4:,4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01
        
        self.kf.x[:4] = box_to_state(initial_box[:4])

    
    def predict(self):
        #if scale velocity would make scale negative, set scale velocity to 0
        if self.kf.x[2] + self.kf.x[6] <= 0: self.kf.x[6] = 0

        self.kf.predict()
        predicted_state = self.kf.x

        return state_to_box(predicted_state[:4])
    
    def update(self, box):
        self.kf.update(box_to_state(box[:4]))
        updated_state = self.kf.x

        return state_to_box(updated_state[:4])
    

class SortTracklet(Tracklet):
    def __init__(self, track_id, initial_box, initial_frame_index, timer):
        super().__init__(track_id)
        self.kf_tracker = KalmanTracker(initial_box)
        self.miss_streak = 0
        self.hits = 1
        self.timer = timer
        
        self.add_box(initial_box, initial_frame_index)

        self.kalman_state_tracklet = Tracklet(-track_id)

    def kalman_predict(self):
        predicted_box = self.kf_tracker.predict()
        conf = self.boxes[-1][4]
        label = self.boxes[-1][5]
        return np.array([*predicted_box, conf, label])
    
    def kalman_update(self, measurement):
        updated_box = self.kf_tracker.update(measurement)
        conf = measurement[4]
        label = measurement[5]
        return np.array([*updated_box, conf, label])
    
    def dec_timer(self):
        if self.timer != 0: self.timer -= 1
    

@silence
def SORT(model, video, iou_min = 0.5, t_lost = 1, probation_timer = 3, min_hits = 5, greedy_assoc = False, no_save = False):
    start_time = time.time()
    active_tracklets = []
    deceased_tracklets = []
    id_counter = 0

    print("Starting SORT")
    for i, frame in tqdm(list(enumerate(video)), bar_format="{l_bar}{bar:30}{r_bar}"):
        frame_pred = model.xywhcl(frame)
        tracklet_predictions = [t.kalman_predict() for t in active_tracklets]

        for t, state_est in zip(active_tracklets, tracklet_predictions):
            t.kalman_state_tracklet.add_box(state_est, i, frame.shape)

        tracklet_indices, detection_indices = [], []
        unassigned_track_indices, unassigned_det_indices = [], []
        if len(tracklet_predictions) == 0: unassigned_det_indices = [j for j in range(len(frame_pred))]
        if len(frame_pred) == 0: unassigned_track_indices = [j for j in range(len(active_tracklets))]

        if len(tracklet_predictions) != 0 and len(frame_pred) != 0:
            #determine tracklet kf pred and model pred matching using optimal linear sum assignment
            iou_mat = iou_matrix(tracklet_predictions, frame_pred)
            if not greedy_assoc:
                tracklet_indices, detection_indices = linear_sum_assignment(iou_mat, True)
            else:
                _, match_indices = correct_preds(tracklet_predictions, frame_pred, iou=iou_min)
                tracklet_indices = list(match_indices[:, 0])
                detection_indices = list(match_indices[:, 1])

            unassigned_track_indices = [j for j in range(len(active_tracklets)) if j not in tracklet_indices]
            unassigned_det_indices = [j for j in range(len(frame_pred)) if j not in detection_indices]

            for track_i, detect_i in zip(tracklet_indices, detection_indices):
                iou = iou_mat[track_i][detect_i]
                assigned_tracklet = active_tracklets[track_i]
                assigned_det = frame_pred[detect_i]

                if iou >= iou_min:
                    #successful match, update kf preds with det measurements
                    updated_kf_box = assigned_tracklet.kalman_update(assigned_det)
                    assigned_tracklet.add_box(updated_kf_box, i, frame.shape)
                    assigned_tracklet.miss_streak = 0
                    assigned_tracklet.hits += 1
                else:
                    #match is not successful, unassign tracklet and detection
                    unassigned_det_indices.append(detect_i)
                    unassigned_track_indices.append(track_i)
        
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

        #clean up dead tracklets
        for track_i in range(len(active_tracklets) - 1, -1, -1):
            track = active_tracklets[track_i]

            if track_i in unassigned_track_indices and track.timer > 0: #tracklet has had a miss before timer is up
                active_tracklets.pop(track_i)

            if track.miss_streak >= t_lost:
                deceased_tracklets.append(track)
                active_tracklets.pop(track_i)

        for tracklet in active_tracklets: #adjust the timer of each tracklet that survived
            tracklet.dec_timer()

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


def play_sort_with_kf(ts):
    """Draws the tracklet set (and the kf state tracklets) onto its video"""
    ts.draw_tracklets()

    for tracklet in ts:
        kf_tracklet = tracklet.kalman_state_tracklet
        draw_single_tracklet(ts.video, kf_tracklet, "", (255, 255, 255))

    ts.video.play(1080, start_paused = True)