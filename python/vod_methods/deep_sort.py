import torch
from scipy.optimize import linear_sum_assignment
import numpy as np

from mv_utils.VOD_utils import silence, iou_matrix, box_off_screen
from reid.reid_data_utils import extract_bbox_image
from .sort import SORT_Tracker, SortTracklet, KalmanTracker


class DeepSortTracklet(SortTracklet):
    def __init__(self, track_id, initial_box, initial_frame_index, timer):
        super().__init__(track_id, initial_box, initial_frame_index, timer)
        self.app_vecs = []

        #self.smoothing_factor = 0.2
        #self.app_vec = None


    def add_app_vec(self, app_vec):
        self.app_vecs.append(app_vec)
        self.app_vecs = self.app_vecs[-100:]

        #if self.app_vec is None:
        #    self.app_vec = app_vec
        #else:
        #    self.app_vec = self.smoothing_factor * self.app_vec + (1 - self.smoothing_factor) * app_vec 

    def tracklet_similarity(self, det_app_vec):
        dists = [det_app_vec.T @ app_vec for app_vec in self.app_vecs]
        return max(dists)

        #return det_app_vec.T @ self.app_vec
    

class DeepSORT_Tracker(SORT_Tracker):
    def __init__(self, 
                 model, 
                 video, 
                 iou_min=0.2, 
                 t_lost=1, 
                 probation_timer=3, 
                 min_hits=5, 
                 greedy_assoc=False, 
                 no_save=False,
                 lambda_iou = 0.98,
                 sim_min = 0.8,
                 reid_model = None
                 ):
        
        super().__init__(model, video, iou_min, t_lost, probation_timer, min_hits, greedy_assoc, no_save)

        self.name = "Deep SORT"
        self.lambda_iou = lambda_iou
        self.sim_min = sim_min
        self.current_frame_index = 0
        self.reid_model = reid_model
        self.tracklet_type = DeepSortTracklet
        self.det_app_vecs = None

        self.kf_est_for_unmatched = False
        

    def get_preds(self, frame_index):
        """Get the model and tracklet predictions for the current frame"""
        frame_pred = self.model.xywhcl(self.video.frames[frame_index])
        tracklet_predictions = [t.kalman_predict() for t in self.active_tracklets]
        self.current_frame_index = frame_index
      
        return tracklet_predictions, frame_pred

    def set_det_app_vecs(self, detections):
        detection_images = torch.zeros((len(detections), 3, *self.reid_model.size))

        for i, d in enumerate(detections):
            extracted_im = extract_bbox_image(d[:4], self.video.frames[self.current_frame_index])
            detection_images[i] = self.reid_model.format_img(extracted_im)

        detection_vectors = self.reid_model.extract_feature(detection_images)
        self.det_app_vecs = detection_vectors.cpu().detach().numpy()

    def appearance_sim_mat(self):
        sim_funcs = [t.tracklet_similarity for t in self.active_tracklets]
        appearance_matrix = np.zeros((len(self.active_tracklets), len(self.det_app_vecs)))

        for i, s_func in enumerate(sim_funcs):
            for j, d_vec in enumerate(self.det_app_vecs):
                appearance_matrix[i, j] = s_func(d_vec)

        return appearance_matrix


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

        if num_detections: 
            self.set_det_app_vecs(detections)

        if num_tracklets and num_detections:
            #determine tracklet pred and detection matching
            iou_mat = iou_matrix(tracklet_preds, detections)
            app_mat = self.appearance_sim_mat()    
            cost_mat = self.lambda_iou * iou_mat + (1 - self.lambda_iou) * app_mat
            
            tracklet_indices, detection_indices = linear_sum_assignment(cost_mat, True)
            tracklet_indices, detection_indices = list(tracklet_indices), list(detection_indices)

            #remove matches that dont meet the min IOU or similarity threshold
            for track_i, det_i in zip(tracklet_indices[::-1], detection_indices[::-1]):
                if iou_mat[track_i, det_i] < self.iou_min or app_mat[track_i, det_i] < self.sim_min:
                    tracklet_indices.remove(track_i)
                    detection_indices.remove(det_i)
                    unassigned_track_indices.append(track_i)
                    unassigned_det_indices.append(det_i)


        unassigned_track_indices = [j for j in range(num_tracklets) if j not in tracklet_indices]
        unassigned_det_indices = [j for j in range(num_detections) if j not in detection_indices]
            
        return tracklet_indices, detection_indices, unassigned_track_indices, unassigned_det_indices
    

    def handle_successful_match(self, assigned_tracklet, assigned_det, assigned_det_app_vec, frame_index):
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

        #add apperance vector of matched det
        assigned_tracklet.add_app_vec(assigned_det_app_vec)


    def process_matches(self, track_is, det_is, un_track_is, un_det_is, track_preds, detections, frame_i):
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
                det_app_vec = self.det_app_vecs[detect_i]
                self.handle_successful_match(assigned_tracklet, assigned_det, det_app_vec, frame_i)
                    
            #tracklet had no match, update with just kf pred
            for track_i in un_track_is:
                track = self.active_tracklets[track_i]
                if self.kf_est_for_unmatched:
                    track.add_box(track_preds[track_i], frame_i, self.frame_size)
                track.miss_streak += 1

            #detection had no match, create new tracklet
            for det_i in un_det_is:
                new_tracklet = self.tracklet_type(self.id_counter, detections[det_i], frame_i, self.probation_timer)
                new_tracklet.add_app_vec(self.det_app_vecs[det_i]) #add apperance vector of matched det
                self.id_counter += 1
                self.active_tracklets.append(new_tracklet)


@silence
def Deep_SORT(model, video, iou_min = 0.5, t_lost = 1, probation_timer = 3, min_hits = 5, greedy_assoc = False, no_save = False, lambda_iou = 0.98, sim_min = 0.8, reid_model = None):
    """Create and run the Deep SORT tracker with a single function"""
    tracker = DeepSORT_Tracker(model, video, iou_min, t_lost, probation_timer, min_hits, greedy_assoc, no_save, lambda_iou,  sim_min, reid_model)
    return tracker()
