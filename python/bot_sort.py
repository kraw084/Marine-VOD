import numpy as np
import math
from filterpy.kalman import KalmanFilter
import time
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
import cv2

from VOD_utils import Tracklet, iou_matrix, TrackletSet, save_VOD, silence, correct_preds


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

    
    def predict(self):
        #if scale velocity would make scale negative, set scale velocity to 0
        if self.kf.x[2] + self.kf.x[6] <= 0: self.kf.x[6] = 0

        self.kf.predict()
        predicted_state = self.kf.x

        return predicted_state[:4].reshape((4,))
    
    def update(self, box):
        self.kf.update(box[:4].reshape((4, 1)))
        updated_state = self.kf.x

        return updated_state[:4].reshape((4,))
    

class BoTSortTracklet(Tracklet):
    def __init__(self, track_id, initial_box, initial_frame_index, timer):
        super().__init__(track_id)
        self.kf_tracker = BoTKalmanTracker(initial_box)
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
def BoT_SORT(model, video, iou_min = 0.5, t_lost = 1, probation_timer = 3, min_hits = 5, greedy_assoc = False, no_save = False):
    start_time = time.time()
    active_tracklets = []
    deceased_tracklets = []
    id_counter = 0

    print("Starting BoT-SORT")
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
            new_tracklet = BoTSortTracklet(id_counter, frame_pred[det_i], i, probation_timer)
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
    print(f"Finished BoT-SORT in {duration}mins")
    print(f"{id_counter + 1} tracklets created")
    print(f"{len(combined_tracklets)} tracklets kept")
    if not no_save: save_VOD(ts, "BoT-SORT")
    return ts



class CameraMotionCompensation:
    def __init__(self):
        self.previous_points = None
        self.prev_frame_grey = None

    def find_transform(self, new_frame):
        new_frame_grey = cv2.cvtColor(new_frame, cv2.COLOR_RGB2GRAY)
        new_points = cv2.goodFeaturesToTrack(new_frame_grey, maxCorners=1000, qualityLevel=0.01, 
                                             minDistance=1, blockSize=3, useHarrisDetector=False, k=0.04)
        
        if self.previous_points is None:
            self.previous_points = new_points
            self.prev_frame_grey = new_frame_grey

        moved_points, status, err = cv2.calcOpticalFlowPyrLK(self.prev_frame_grey, new_frame_grey, self.previous_points, None)
       
        indicies_to_keep = [i for i in range(status.shape[0]) if status[i] == 1]
        
        prev_points_keep = self.previous_points[indicies_to_keep]
        moved_points_keep = moved_points[indicies_to_keep]

        affine_mat, _ = cv2.estimateAffine2D(prev_points_keep, moved_points_keep, method=cv2.RANSAC)

        self.previous_points = new_points
        self.prev_frame_grey = new_frame_grey

        return affine_mat

from MOT17 import load_MOT17_video
vid = load_MOT17_video("MOT17-13", "train")

print("Starting")
GMC = CameraMotionCompensation()
GMC.find_transform(vid.frames[0])

for i in range(1, 5):
    mat = GMC.find_transform(vid.frames[i])
    aligned = cv2.warpAffine(vid.frames[1], mat, vid.frames[0].shape[::-1][:2])
    

output = cv2.addWeighted(vid.frames[0], 0.5, vid.frames[1], 0.5, 0)
cv2.imshow("test", output)
cv2.waitKey()