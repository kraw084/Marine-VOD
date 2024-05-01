import numpy as np
import math
from filterpy.kalman import KalmanFilter

from VOD_utils import Tracklet

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
        self.kf.predict()
        predicted_state = self.kf.x

        return state_to_box(predicted_state[:4])
    
    def update(self, box):
        self.kf.update(box_to_state(box[:4]))
        updated_state = self.kf.x

        return state_to_box(updated_state[:4])
    

class SortTracklet(Tracklet):
    def __init__(self, id, initial_box, initial_frame_index):
        super().__init__(id)
        self.kf_tracker = KalmanTracker(initial_box)
        self.miss_streak = 0
        self.probation_timer = 0

        self.add_box(initial_box, initial_frame_index)

    def kalman_predict(self):
        predicted_box = self.kf_tracker.predict()
        conf = self.boxes[-1][4]
        label = self.boxes[-1][5]
        return np.array([*predicted_box, conf, label])
    
    def kalman_update(self, measurement):
        return self.kf_tracker.update(measurement)