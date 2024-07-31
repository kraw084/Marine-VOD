import numpy as np
from filterpy.kalman import KalmanFilter

from mv_utils.Cmc import CameraMotionCompensation
from mv_utils.VOD_utils import silence

from .sort import SORT_Tracker, SortTracklet, KalmanTracker

"""Aharon, N., Orfaig, R., & Bobrovsky, B. Z. (2022). BoT-SORT: Robust associations multi-pedestrian tracking. 
arXiv preprint arXiv:2206.14651."""


class BoTKalmanTracker(KalmanTracker):
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
        
        #self.kf.R[2:,2:] *= 10.0
        #self.kf.P[4:,4:] *= 1000.0
        #self.kf.P *= 10.0
        #self.kf.Q[-2:,-2:] *= 0.01
        #self.kf.Q[4:,4:] *= 0.01

        self.kf.x[:4] = initial_box[:4].reshape((4, 1))

    def predict(self, mat=None):
        #if width or height would become negative, set velocity to 0
        if self.kf.x[2] + self.kf.x[6] <= 0: self.kf.x[6] = 0
        if self.kf.x[3] + self.kf.x[7] <= 0: self.kf.x[7] = 0

        self.kf.predict()

        if not mat is None: #if a transformation mat is supplied use it on kf prediction
            self.apply_transform(mat)

        predicted_state = self.kf.x
        return self.state_to_box(predicted_state[:4])
    
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
  
    def box_to_state(self, box):
        return box.reshape((4, 1))
    
    def state_to_box(self, state):
        return state.reshape((4,))
 
    
class BoTSortTracklet(SortTracklet):
    def __init__(self, track_id, initial_box, initial_frame_index, timer):
        super().__init__(track_id, initial_box, initial_frame_index, timer)
        self.kf_tracker = BoTKalmanTracker(initial_box)
        
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
    
    
class BoTSORT_Tracker(SORT_Tracker):
    def __init__(self, model, video, iou_min = 0.5, t_lost = 1, probation_timer = 3, min_hits = 5, greedy_assoc = False, no_save = False):
        super().__init__(model, video, iou_min, t_lost, probation_timer, min_hits, greedy_assoc, no_save)
        self.tracklet_type = BoTSortTracklet
        self.name = "BoT SORT"
        self.kf_est_for_unmatched = False
        
        self.enable_cmc = True
        self.cam_comp = CameraMotionCompensation()

    
    def get_preds(self, frame_index):
        """Get the model and tracklet predictions (with camera motion compensation) for the current frame"""
        cam_comp_mat = self.cam_comp.find_transform(self.video.frames[frame_index]) if self.enable_cmc else None
        frame_pred =self.model.xywhcl(self.video.frames[frame_index])
        tracklet_predictions = [t.kalman_predict(cam_comp_mat) for t in self.active_tracklets]

        return tracklet_predictions, frame_pred
    
    
@silence
def BoT_SORT(model, video, iou_min = 0.5, t_lost = 1, probation_timer = 3, min_hits = 5, greedy_assoc = False, no_save = False):
    """Create and run the SORT tracker with a single function"""
    tracker = BoTSORT_Tracker(model, video, iou_min, t_lost, probation_timer, min_hits, greedy_assoc, no_save)
    return tracker()