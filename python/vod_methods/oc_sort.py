import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from python.vod_methods.sort import SORT_Tracker, SortTracklet
from python.mv_utils.VOD_utils import iou_matrix, silence

"""Cao, J., Pang, J., Weng, X., Khirodkar, R., & Kitani, K. (2023). 
   Observation-centric sort: Rethinking sort for robust multi-object tracking. 
   In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition 
   (pp. 9686-9696)."""


def lerp_box(box1, box2, t):
    #linearly interpolate two boxes except for the label value
    lerped_box = box1 + t * (box2 - box1)
    lerped_box[5] = box1[5]
    return lerped_box
   
    
class OC_SortTracklet(SortTracklet):
    def __init__(self, track_id, initial_box, initial_frame_index, timer):
        super().__init__(track_id, initial_box, initial_frame_index, timer)
        self.last_observation = initial_box
    
    def add_box(self, box, frame_index, im_shape=None, observation=None):
        #if this is the first observation seen after a miss streak trigger ORU
        if (not observation is None) and self.miss_streak != 0:
            self.observation_centric_reupdate(observation)
        
        #save last observation and KF state for ORU
        if (not observation is None):
            self.last_observation = observation
            self.last_kf_x = self.kf_tracker.kf.x
            self.last_kf_p = self.kf_tracker.kf.P
            self.last_list_index = len(self.boxes) - 1
        
        self.boxes.append(box)
        self.frame_indexes.append(frame_index)

        if self.start_frame is None: self.start_frame = frame_index

        if frame_index < self.start_frame: self.start_frame = frame_index
        if frame_index > self.end_frame: self.end_frame = frame_index
        
    def observation_centric_reupdate(self, new_observation):         
        #resest kf to before miss streak
        self.kf_tracker.kf.x = self.last_kf_x
        self.kf_tracker.kf.P = self.last_kf_p
          
        #for each missing frame run the kf with the interpolated observation to re-update the parameters
        for i in range(self.miss_streak):
            tracklet_pred = self.kalman_predict()
            t = i/(self.miss_streak + 1)
            interpolated_box = lerp_box(self.last_observation, new_observation, t)
            updated_box = self.kalman_update(interpolated_box)
            
            #self.boxes[self.last_list_index + i] = updated_box


class OC_SORT_Tracker(SORT_Tracker):
    def __init__(self, 
                 model, 
                 video, 
                 iou_min = 0.5, 
                 t_lost = 1, 
                 probation_timer = 3, 
                 min_hits = 5,
                 orm_factor = 0.2,
                 orm_t = 3, 
                 no_save = False
                 ):
        super().__init__(model, video, iou_min, t_lost, probation_timer, min_hits, greedy_assoc=False, no_save=no_save)
        self.name = "OC-SORT"
        self.tracklet_type = OC_SortTracklet
        self.orm_factor = orm_factor
        self.orm_t = orm_t
        
    def handle_successful_match(self, assigned_tracklet, assigned_det, frame_index):
        """Handles a successful match between a tracklet and detection by updating the kf state
        Args:
            assigned_tracklet: tracklet that was matched
            assigned_det: detection that was matched
            frame_index: index of the current frame
            frame_size: size of the current frame (used for clipping box)
        """
        updated_kf_box = assigned_tracklet.kalman_update(assigned_det)
        assigned_tracklet.add_box(updated_kf_box, frame_index, self.frame_size, assigned_det)
        assigned_tracklet.miss_streak = 0
        assigned_tracklet.hits += 1
       
       
    def orm_matrix(self, detections):
        mat = np.zeros((len(self.active_tracklets), len(detections)))
        
        for i in range(len(self.active_tracklets)):
            tracklet = self.active_tracklets[i]
            
            #no penality if tracklet is not long enough
            if len(tracklet.boxes) < self.orm_t + 1: continue
            
            last_box = tracklet.boxes[-1]
            target_box = tracklet.boxes[-1 - self.orm_t]
            tracklet_angle = np.arctan((last_box[1] - target_box[1])/(last_box[0] - target_box[0] + 1E-10))
            
            for j in range(len(detections)):
                det = detections[j]
                    
                target_box = tracklet.boxes[-self.orm_t]
                mat[i, j] = np.abs(np.arctan((det[1] - target_box[1])/(det[0] - target_box[0] + 1E-10)) - tracklet_angle)
                
        return self.orm_factor * mat

        
    def det_tracklet_matches(self, tracklet_preds, detections, with_orm=True):
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
            
            if with_orm:
                cost_matrix = iou_mat + self.orm_matrix(detections)
            else:
                cost_matrix = iou_mat
            
            tracklet_indices, detection_indices = linear_sum_assignment(cost_matrix, True)
            tracklet_indices, detection_indices = list(tracklet_indices), list(detection_indices)
            
            #remove any matches that are below the iou threshold
            for track_i, det_i in zip(tracklet_indices[::-1], detection_indices[::-1]):
                if iou_mat[track_i, det_i] < self.iou_min:
                    tracklet_indices.remove(track_i)
                    detection_indices.remove(det_i)
                    unassigned_track_indices.append(track_i)
                    unassigned_det_indices.append(det_i)

        unassigned_track_indices = [j for j in range(num_tracklets) if j not in tracklet_indices]
        unassigned_det_indices = [j for j in range(num_detections) if j not in detection_indices]
            
        return tracklet_indices, detection_indices, unassigned_track_indices, unassigned_det_indices
    
    
    def track(self):
        """Runs the SORT algorithm for every frame in the video"""
        print(f"Starting {self.name}")
        for i in tqdm(range(self.video.num_of_frames), bar_format="{l_bar}{bar:30}{r_bar}"):
            tracklet_predictions, detections = self.get_preds(i)
            
            tracklet_indices, detection_indices, unassigned_track_indices, \
            unassigned_det_indices = self.det_tracklet_matches(tracklet_predictions, detections)
          
            #Perform observation centric recovery 
            unassigned_tracklet_last_obs = [self.active_tracklets[j].last_observation for j in unassigned_track_indices]
            unassigned_dets = [detections[j] for j in unassigned_det_indices]
            
            orc_det_indices, orc_tracklet_indices, orc_unassigned_track_indices, \
            orc_unassigned_det_indices = self.det_tracklet_matches(unassigned_tracklet_last_obs, unassigned_dets, with_orm=False)
            
            orc_det_map = lambda x: unassigned_det_indices[x]
            orc_tracklet_map = lambda x: unassigned_track_indices[x]
            
            tracklet_indices = tracklet_indices + list(map(orc_tracklet_map, orc_det_indices))
            detection_indices = detection_indices + list(map(orc_det_map, orc_tracklet_indices))
            unassigned_track_indices = list(map(orc_tracklet_map, orc_unassigned_track_indices))
            unassigned_det_indices = list(map(orc_det_map, orc_unassigned_det_indices))
            
            self.process_matches(tracklet_indices, detection_indices, unassigned_track_indices, unassigned_det_indices, tracklet_predictions, detections, i)
            
            self.cleanup_dead_tracklets(unassigned_track_indices)
            
            self.cleanup_off_screen()
           
        combined_tracklets = self.deceased_tracklets + self.active_tracklets
        self.cleanup_min_hits(combined_tracklets) 
        return self.save_and_return(combined_tracklets)
    
    
@silence
def OC_SORT(model, video, iou_min = 0.5, t_lost = 1, probation_timer = 3, min_hits = 5, orm_factor = 0.2, orm_t = 3, no_save = False):
    """Create and run the SORT tracker with a single function"""
    tracker = OC_SORT_Tracker(model, video, iou_min, t_lost, probation_timer, min_hits, orm_factor, orm_t, no_save)
    return tracker()