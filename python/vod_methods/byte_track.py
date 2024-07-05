from tqdm import tqdm

from python.vod_methods.sort import SORT_Tracker
from python.utils.VOD_utils import silence


class ByteTrack_Tracker(SORT_Tracker):
    def __init__(self, 
                 model, 
                 video, 
                 iou_min=0.2, 
                 t_lost=1, 
                 probation_timer=3, 
                 min_hits=5,
                 low_conf_th = 0.6, 
                 greedy_assoc=False, 
                 no_save=False
                 ):
        
        super().__init__(model, video, iou_min, t_lost, probation_timer, min_hits, greedy_assoc, no_save)
        self.low_conf_th = low_conf_th
        self.orignal_conf = model.conf
        model.conf = 0.001
        
    def track(self):
        """Runs the SORT algorithm for every frame in the video"""
        print(f"Starting {self.name}")
        for i in tqdm(range(self.video.num_of_frames), bar_format="{l_bar}{bar:30}{r_bar}"):
            tracklet_predictions, detections = self.get_preds(i)
              
            #associate high conf dets with tracklet preds
            high_conf_indices = [j for j in range(len(detections)) if detections[j][4] > self.low_conf_th]
            high_conf_dets = [detections[j] for j in high_conf_indices]
            
            hc_tracklet_indices, hc_detection_indices, hc_unassigned_track_indices, \
            hc_unassigned_det_indices = self.det_tracklet_matches(tracklet_predictions, high_conf_dets)      
   
            #associate low conf dets with remaining tracklet preds
            low_conf_indices = [j for j in range(len(detections)) if detections[j][4] <= self.low_conf_th]
            low_conf_dets = [detections[j] for j in low_conf_indices]
            remaining_tracklets = [tracklet_predictions[j] for j in hc_unassigned_track_indices]
            
            lc_tracklet_indices, lc_detection_indices, lc_unassigned_track_indices, \
            lc_unassigned_det_indices = self.det_tracklet_matches(remaining_tracklets, low_conf_dets)
            
            #index lists returned from track_det_match are relative to the shortened input list so the indices need to be translated
            hc_det_map = lambda x: high_conf_indices[x]
            lc_det_map = lambda x: low_conf_indices[x]
            lc_track_map = lambda x: hc_unassigned_track_indices[x]
            
            tracklet_indices = hc_tracklet_indices + list(map(lc_track_map, lc_tracklet_indices))
            detection_indices = list(map(hc_det_map, hc_detection_indices)) + list(map(lc_det_map, lc_detection_indices))
            unassigned_track_indices = list(map(lc_track_map, lc_unassigned_track_indices))
            unassigned_det_indices = list(map(hc_det_map, hc_unassigned_det_indices)) + list(map(lc_det_map, lc_unassigned_det_indices))
                
            self.process_matches(tracklet_indices, detection_indices, unassigned_track_indices, unassigned_det_indices, tracklet_predictions, detections, i)
            
            self.cleanup_dead_tracklets(unassigned_track_indices)
            
            self.cleanup_off_screen()
           
        combined_tracklets = self.deceased_tracklets + self.active_tracklets
        self.cleanup_min_hits(combined_tracklets) 
        
        self.model.conf = self.orignal_conf
        return self.save_and_return(combined_tracklets)
        
        
@silence
def ByteTrack(model, video, iou_min = 0.5, t_lost = 1, probation_timer = 3, min_hits = 5, low_conf_th = 0.6, greedy_assoc = False, no_save = False):
    """Create and run the ByteTrack tracker with a single function"""
    tracker = ByteTrack_Tracker(model, video, iou_min, t_lost, probation_timer, min_hits, low_conf_th, greedy_assoc, no_save)
    return tracker()
