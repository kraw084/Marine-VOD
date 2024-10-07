import torch
from scipy.optimize import linear_sum_assignment

from mv_utils.VOD_utils import silence, iou_matrix, box_off_screen
from reid.reid_data_utils import extract_bbox_image
from .sort import SORT_Tracker



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

        self.kf_est_for_unmatched = False
        
    def get_preds(self, frame_index):
        """Get the model and tracklet predictions for the current frame"""
        frame_pred = self.model.xywhcl(self.video.frames[frame_index])
        tracklet_predictions = [t.kalman_predict() for t in self.active_tracklets]
        self.current_frame_index = frame_index
      
        return tracklet_predictions, frame_pred


    def appearance_sim_mat(self, tracklet_predictions, detections):
        tracklet_images = torch.zeros((len(tracklet_predictions), 3, *self.reid_model.size))
        detection_images = torch.zeros((len(detections), 3, *self.reid_model.size))

        off_screen_tracklet_preds = []

        for i, t in enumerate(tracklet_predictions):
            if box_off_screen(self.video.frames[self.current_frame_index].shape, t):
                off_screen_tracklet_preds.append(i)
                continue

            extracted_im = extract_bbox_image(t[:4], self.video.frames[self.current_frame_index])
            tracklet_images[i] = self.reid_model.format_img(extracted_im)

        for i, d in enumerate(detections):
            extracted_im = extract_bbox_image(d[:4], self.video.frames[self.current_frame_index])
            detection_images[i] = self.reid_model.format_img(extracted_im)

        tracklet_vectors = self.reid_model.extract_feature(tracklet_images)
        for i in off_screen_tracklet_preds:
            tracklet_vectors[i] = torch.zeros_like(tracklet_vectors[i])

        detection_vectors = self.reid_model.extract_feature(detection_images)

        appearance_matrix = self.reid_model.batch_vector_similarity(tracklet_vectors, detection_vectors, numpy=True)
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
         
        if num_tracklets and num_detections:
            #determine tracklet pred and detection matching
            iou_mat = iou_matrix(tracklet_preds, detections)
            app_mat = self.appearance_sim_mat(tracklet_preds, detections)            
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
    
@silence
def Deep_SORT(model, video, iou_min = 0.5, t_lost = 1, probation_timer = 3, min_hits = 5, greedy_assoc = False, no_save = False, lambda_iou = 0.98, sim_min = 0.8, reid_model = None):
    """Create and run the Deep SORT tracker with a single function"""
    tracker = DeepSORT_Tracker(model, video, iou_min, t_lost, probation_timer, min_hits, greedy_assoc, no_save, lambda_iou,  sim_min, reid_model)
    return tracker()
