import numpy as np
from filterpy.kalman import KalmanFilter
import time
from tqdm import tqdm

from python.utils.VOD_utils import Tracklet, TrackletSet, save_VOD, silence
from python.utils.Cmc import CameraMotionCompensation

"""Aharon, N., Orfaig, R., & Bobrovsky, B. Z. (2022). BoT-SORT: Robust associations multi-pedestrian tracking. 
arXiv preprint arXiv:2206.14651."""


class BoTKalmanTracker():
    def __init__(self, initial_box):
        self.kf = KalmanFilter(8, 4) 
        #state is [x, y, w, h, x_v, y_v, w_v, h_v]
        #measurement in [x, y, w, h]

        #State transition matrix
        self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0 ,0],
                              [0, 1, 0, 0, 0, 1, 0, 0],
                              [0, 0, 1, 0, 0, 0, 1, 0],
                              [0, 0, 0, 1, 0, 0, 0, 1],
                              [0, 0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 0, 1]], np.float32)
        #Measurment matrix
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0, 0]], np.float32)
        

        self.kf.P[4:,4:] *= 1000.0
        self.kf.P *= 10.0
        
        pos_std = 0.05
        vel_std = 0.00625
        w = initial_box[2]
        h = initial_box[3]

        q_diag_values = np.array([pos_std * w, pos_std * h, 
                                  pos_std * w, pos_std * h,
                                  vel_std * w, vel_std * h,
                                  vel_std * w, vel_std * h])
        
        q_diag_values[:4] *= 2
        q_diag_values[4:] *= 10
        
        self.kf.Q = np.diag(np.square(q_diag_values))

        r_diag_values = np.array([pos_std * w, pos_std * h, 
                                  pos_std * w, pos_std * h])
        
        self.kf.R = np.diag(np.square(r_diag_values))

        
        self.kf.x[:4] = initial_box[:4].reshape((4, 1))

    
    def predict(self, mat=None):
        #if scale velocity would make scale negative, set scale velocity to 0
        if self.kf.x[2] + self.kf.x[6] <= 0: self.kf.x[6] = 0

        self.kf.predict()

        if not mat is None: #if a transformation mat is supplied use it on kf prediction
            self.apply_transform(mat)

        predicted_state = self.kf.x
        return predicted_state[:4].reshape((4,))
    

    def update(self, box):
        self.kf.update(box[:4].reshape((4, 1)))
        updated_state = self.kf.x

        return updated_state[:4].reshape((4,))
    

    def apply_transform(self, mat):
        #extract components of the transformation matrix
        M = mat[:2, :2]
        if mat.shape[0] == 2: #affine transform
            translation_vector = mat[:, 2]
        else: #homography
            M /= mat[2, 2]
            translation_vector = mat[:2, 2]

        M_diag = np.zeros((8, 8))
        for i in range(4):
            M_diag[i:i+2, i:i+2] = M

        #update state vector
        self.kf.x = M_diag @ self.kf.x
        self.kf.x[:2, 0] += translation_vector

        #update covariance mat
        self.kf.P = M_diag @ self.kf.P @ M_diag.T
    

class BoTSortTracklet(Tracklet):
    def __init__(self, track_id, initial_box, initial_frame_index, timer):
        super().__init__(track_id)
        self.kf_tracker = BoTKalmanTracker(initial_box)
        self.miss_streak = 0
        self.hits = 1
        self.timer = timer
        
        self.add_box(initial_box, initial_frame_index)

        self.kalman_state_tracklet = Tracklet(-track_id)

    def kalman_predict(self, mat = None):
        predicted_box = self.kf_tracker.predict(mat)
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
    
"""
@silence
def BoT_SORT(model, video, iou_min = 0.5, t_lost = 1, probation_timer = 3, min_hits = 5, greedy_assoc = False, no_save = False):
    start_time = time.time()
    active_tracklets = []
    deceased_tracklets = []
    id_counter = 0

    cam_comp = CameraMotionCompensation()

    print("Starting BoT-SORT")
    for i, frame in tqdm(list(enumerate(video)), bar_format="{l_bar}{bar:30}{r_bar}"):
        frame_pred = model.xywhcl(frame)

        cam_comp_mat = cam_comp.find_transform(frame)
        tracklet_predictions = [t.kalman_predict(cam_comp_mat) for t in active_tracklets]

        for t, state_est in zip(active_tracklets, tracklet_predictions):
            t.kalman_state_tracklet.add_box(state_est, i, frame.shape)

        tracklet_indices, detection_indices, unassigned_track_indices, \
        unassigned_det_indices = det_tracklet_matches(tracklet_predictions, frame_pred, iou_min, greedy_assoc)
       
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
            new_tracklet = BoTSortTracklet(id_counter, frame_pred[det_i], i, probation_timer)
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
    print(f"Finished BoT-SORT in {duration}mins")
    print(f"{id_counter + 1} tracklets created")
    print(f"{len(combined_tracklets)} tracklets kept")
    if not no_save: save_VOD(ts, "BoT-SORT")
    return ts"""