import time
from tqdm import tqdm
import numpy as np
import math
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

from mv_utils.VOD_utils import TrackletSet, save_VOD, silence, tracklet_off_screen
from mv_utils.VOD_utils import Tracklet, iou_matrix
from mv_utils.Eval_utils import correct_preds

"""Bewley, A., Ge, Z., Ott, L., Ramos, F., & Upcroft, B. (2016, September). 
   Simple online and realtime tracking. In 2016 IEEE international conference on image processing
   (ICIP) (pp. 3464-3468). IEEE."""
   

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
        
        self.kf.x[:4] = self.box_to_state(initial_box[:4])

    def predict(self):
        #if scale velocity would make scale negative, set scale velocity to 0
        if self.kf.x[2] + self.kf.x[6] <= 0: self.kf.x[6] = 0

        self.kf.predict()
        predicted_state = self.kf.x

        return self.state_to_box(predicted_state[:4])
    
    def update(self, box):
        self.kf.update(self.box_to_state(box[:4]))
        updated_state = self.kf.x

        return self.state_to_box(updated_state[:4])
    
    def box_to_state(self, box):
        """Takes a box [x, y, w, h] and converts to to the kalman state [x, y, s, r]"""
        state = np.zeros((4, 1), np.float32)
        state[0] = box[0]
        state[1] = box[1]
        state[2] = box[2] * box[3]
        state[3] = box[2] / box[3]

        return state

    def state_to_box(self, state):
        """Takes a kalman state [x, y, s, r] and converts to to the kalman state [x, y, w, h]"""
        state = np.reshape(state, (4,))
        box = np.zeros((4,))
        box[0] = state[0]
        box[1] = state[1]
        box[2] = math.sqrt(state[2] * state[3])
        box[3] = state[2] / box[2]

        return np.rint(box)
    

class SortTracklet(Tracklet):
    def __init__(self, track_id, initial_box, initial_frame_index, timer):
        super().__init__(track_id)
        self.kf_tracker = KalmanTracker(initial_box)
        self.miss_streak = 0
        self.hits = 1
        self.timer = timer
        
        self.add_box(initial_box, initial_frame_index)

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


class SORT_Tracker:
    def __init__(self, model, video, iou_min = 0.5, t_lost = 1, probation_timer = 3, min_hits = 5, greedy_assoc = False, no_save = False):
        self.model = model
        self.video = video
        self.iou_min = iou_min
        self.t_lost = t_lost
        self.probation_timer = probation_timer
        self.min_hits = min_hits
        self.greedy_assoc = greedy_assoc
        self.no_save = no_save
        self.frame_size = video.frames[0].shape
        self.tracklet_type = SortTracklet
        self.name = "SORT"
        
        self.start_time = time.time()
        self.active_tracklets = []
        self.deceased_tracklets = []
        self.id_counter = 0
        
    def get_preds(self, frame_index):
        """Get the model and tracklet predictions for the current frame"""
        frame_pred =self.model.xywhcl(self.video.frames[frame_index])
        tracklet_predictions = [t.kalman_predict() for t in self.active_tracklets]
      
        return tracklet_predictions, frame_pred
    
    def det_tracklet_matches(self, tracklet_preds, detections):
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
        unassigned_track_indices, unassigned_det_indices = [], []
         
        if num_tracklets and num_detections:
            #determine tracklet pred and detection matching
            iou_mat = iou_matrix(tracklet_preds, detections)
            if not self.greedy_assoc:
                tracklet_indices, detection_indices = linear_sum_assignment(iou_mat, True)
                tracklet_indices, detection_indices = list(tracklet_indices), list(detection_indices)
                
                #remove any matches that are below the iou threshold
                for track_i, det_i in zip(tracklet_indices[::-1], detection_indices[::-1]):
                    if iou_mat[track_i, det_i] < self.iou_min:
                        tracklet_indices.remove(track_i)
                        detection_indices.remove(det_i)
                        unassigned_track_indices.append(track_i)
                        unassigned_det_indices.append(det_i)
            else:
                _, match_indices = correct_preds(tracklet_preds, detections, iou=self.iou_min)
                tracklet_indices = list(match_indices[:, 0])
                detection_indices = list(match_indices[:, 1])

        unassigned_track_indices = [j for j in range(num_tracklets) if j not in tracklet_indices]
        unassigned_det_indices = [j for j in range(num_detections) if j not in detection_indices]
            
        return tracklet_indices, detection_indices, unassigned_track_indices, unassigned_det_indices

    def handle_successful_match(self, assigned_tracklet, assigned_det, frame_index):
        """Handles a successful match between a tracklet and detection by updating the kf state
        Args:
            assigned_tracklet: tracklet that was matched
            assigned_det: detection that was matched
            frame_index: index of the current frame
            frame_size: size of the current frame (used for clipping box)
        """
        updated_kf_box = assigned_tracklet.kalman_update(assigned_det)
        assigned_tracklet.add_box(updated_kf_box, frame_index, self.frame_size)
        assigned_tracklet.miss_streak = 0
        assigned_tracklet.hits += 1
    
    def cleanup_dead_tracklets(self, unassigned_track_indices):
        """Removes tracklets that failed probation or have exceeded the miss streak
        Args:
            unassigned_track_indices: list of tracklet that did not have a match this frame
        """
        for track_i in range(len(self.active_tracklets) - 1, -1, -1):
            track = self.active_tracklets[track_i]

            #tracklet has had a miss before timer is up
            if track_i in unassigned_track_indices and track.timer > 0: 
                self.active_tracklets.pop(track_i)

            #tracklets miss streak is too high
            if track.miss_streak >= self.t_lost:
                self.deceased_tracklets.append(track)
                self.active_tracklets.pop(track_i)

        #adjust the timer of each tracklet that survived
        for tracklet in self.active_tracklets: 
            tracklet.dec_timer()

    def cleanup_min_hits(self, tracklets):
        #remove tracklets that did not meet the requirement minimum number of hits
        for track_i in range(len(tracklets) - 1, -1, -1):
            if tracklets[track_i].hits < self.min_hits:
                tracklets.pop(track_i)

    def cleanup_off_screen(self):
        for track_i in range(len(self.active_tracklets) - 1, -1, -1):
            track = self.active_tracklets[track_i]

            if tracklet_off_screen(self.video.frames[0], track):
                self.deceased_tracklets.append(track)
                self.active_tracklets.pop(track_i) 

    def process_matches(self, track_is, det_is, un_track_is, un_det_is, track_preds, detections, frame_i, kf_bbox_unmatched_tracks=True):
        """Processes the tracklets and detections based on whether they were matched or not
           Args:
                track_is: list of tracklet indices with matching detections
                det_is: list of detection indices with matching tracklets
                un_track_is: list of tracklet indices with no matching detections
                un_det_is: list of detection indices with no matching tracklets
                track_preds: list of tracklet state estimates for every active tracklet
                detections: list of detections in the current frame
                frame_i: index of the current frame
        """
        #successful match, update kf preds with det measurements
        for track_i, detect_i in zip(track_is, det_is):
            assigned_tracklet = self.active_tracklets[track_i]
            assigned_det = detections[detect_i]
            self.handle_successful_match(assigned_tracklet, assigned_det, frame_i)
                
        #tracklet had no match, update with just kf pred
        for track_i in un_track_is:
            track = self.active_tracklets[track_i]
            if kf_bbox_unmatched_tracks:
                track.add_box(track_preds[track_i], frame_i, self.frame_size)
            track.miss_streak += 1

        #detection had no match, create new tracklet
        for det_i in un_det_is:
            new_tracklet = self.tracklet_type(self.id_counter, detections[det_i], frame_i, self.probation_timer)
            self.id_counter += 1
            self.active_tracklets.append(new_tracklet)

    def save_and_return(self, tracklets):
        ts = TrackletSet(self.video, tracklets, self.model.num_to_class)

        duration = round((time.time() - self.start_time)/60, 2)
        print(f"Finished {self.name} in {duration}mins")
        print(f"{self.id_counter + 1} tracklets created")
        print(f"{len(tracklets)} tracklets kept")
        if not self.no_save: save_VOD(ts, self.name)
        return ts

    def track(self):
        """Runs the SORT algorithm for every frame in the video"""
        print(f"Starting {self.name}")
        for i in tqdm(range(self.video.num_of_frames), bar_format="{l_bar}{bar:30}{r_bar}"):
            tracklet_predictions, detections = self.get_preds(i)
            
            tracklet_indices, detection_indices, unassigned_track_indices, \
            unassigned_det_indices = self.det_tracklet_matches(tracklet_predictions, detections)
            
            self.process_matches(tracklet_indices, detection_indices, unassigned_track_indices, unassigned_det_indices, tracklet_predictions, detections, i)
            
            self.cleanup_dead_tracklets(unassigned_track_indices)
            
            self.cleanup_off_screen()
           
        combined_tracklets = self.deceased_tracklets + self.active_tracklets
        self.cleanup_min_hits(combined_tracklets) 
        return self.save_and_return(combined_tracklets)
 
    def __call__(self):
        return self.track()
    
@silence
def SORT(model, video, iou_min = 0.5, t_lost = 1, probation_timer = 3, min_hits = 5, greedy_assoc = False, no_save = False):
    """Create and run the SORT tracker with a single function"""
    tracker = SORT_Tracker(model, video, iou_min, t_lost, probation_timer, min_hits, greedy_assoc, no_save)
    return tracker()
