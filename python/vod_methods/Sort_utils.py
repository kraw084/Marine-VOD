import numpy as np
import math
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

from python.utils.VOD_utils import Tracklet, iou_matrix, draw_single_tracklet
from python.utils.Eval_utils import correct_preds

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


def play_sort_with_kf(ts):
    """Draws the tracklet set (and the kf state tracklets) onto its video"""
    ts.draw_tracklets()

    for tracklet in ts:
        kf_tracklet = tracklet.kalman_state_tracklet
        draw_single_tracklet(ts.video, kf_tracklet, "", (255, 255, 255))

    ts.video.play(1080, start_paused = True)
    
    
def det_tracklet_matches(tracklet_preds, detections, iou_min = 0.5, greedy_assoc = False):
    """Perfoms association between tracklets and detections
       Args:
           tracklet_preds: list of tracklet state estimates for every active tracklet
           detections: list of detections in the current frame
           iou_min: minimum iou for a detection to be associated with a tracklet
           greedy_assoc: perform greedy association instead of linear sum assignment
       Returns:
           tracklet_indices: list of tracklet indices with matching detections
           detection_indices: list of detection indices with matching tracklets
           unassigned_track_indices: list of tracklet indices with no matching detections
           unassigned_det_indices: list of detection indices with no matching tracklets
    """
    num_tracklets = len(tracklet_preds)
    num_detections = len(detections)
    tracklet_indices, detection_indices = [], []
    
    if num_tracklets == 0: 
        #if there are no active tracklets all detections are unassigned
        unassigned_det_indices = [j for j in range(num_detections)]
    
    if num_detections == 0:
        #if there are no detections all tracklets are unassigned 
        unassigned_track_indices = [j for j in range(num_tracklets)]
        
    if num_tracklets and num_detections:
        #determine tracklet pred and detection matching
        iou_mat = iou_matrix(tracklet_preds, detections)
        if not greedy_assoc:
            tracklet_indices, detection_indices = linear_sum_assignment(iou_mat, True)
            
            #remove any matches that are below the iou threshold
            for track_i, det_i in zip(tracklet_indices[::-1], detection_indices[::-1]):
                if iou_mat[track_i, det_i] < iou_min:
                    tracklet_indices.remove(track_i)
                    detection_indices.remove(det_i)
                    unassigned_track_indices.append(track_i)
                    unassigned_det_indices.append(det_i)
        else:
            _, match_indices = correct_preds(tracklet_preds, detections, iou=iou_min)
            tracklet_indices = list(match_indices[:, 0])
            detection_indices = list(match_indices[:, 1])

        unassigned_track_indices = [j for j in range(num_tracklets) if j not in tracklet_indices]
        unassigned_det_indices = [j for j in range(num_detections) if j not in detection_indices]
        
    return tracklet_indices, detection_indices, unassigned_track_indices, unassigned_det_indices
    
    
def handle_successful_match(assigned_tracklet, assigned_det, frame_index, frame_size):
    """Handles a successful match between a tracklet and detection by updating the kf state
       Args:
           assigned_tracklet: tracklet that was matched
           assigned_det: detection that was matched
           frame_index: index of the current frame
           frame_size: size of the current frame (used for clipping box)
    """
    updated_kf_box = assigned_tracklet.kalman_update(assigned_det)
    assigned_tracklet.add_box(updated_kf_box, frame_index, frame_size)
    assigned_tracklet.miss_streak = 0
    assigned_tracklet.hits += 1
    
def cleanup_dead_tracklets(active_tracklets, deceased_tracklets, unassigned_track_indices, t_lost):
    """Removes tracklets that failed probation or have exceeded the miss streak
       Args:
           active_tracklets: list of active tracklets
           deceased_tracklets: list of tracklets that have died
           unassigned_track_indices: list of tracklet that did not have a match this frame
           t_lost: maximum number of misses before a tracklet is removed
    """
    for track_i in range(len(active_tracklets) - 1, -1, -1):
        track = active_tracklets[track_i]

        #tracklet has had a miss before timer is up
        if track_i in unassigned_track_indices and track.timer > 0: 
            deceased_tracklets.append(track)
            active_tracklets.pop(track_i)

        #tracklets miss streak is too high
        if track.miss_streak >= t_lost:
            deceased_tracklets.append(track)
            active_tracklets.pop(track_i)

    #adjust the timer of each tracklet that survived
    for tracklet in active_tracklets: 
        tracklet.dec_timer()