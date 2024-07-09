import numpy as np
from tqdm import tqdm
import time

from mv_utils.VOD_utils import iou_matrix, Tracklet, TrackletSet, save_VOD, silence


"""Han, W., Khorrami, P., Paine, T. L., Ramachandran, P., Babaeizadeh, M., Shi, H., ... & Huang, T. S. (2016). 
   Seq-nms for video object detection. arXiv preprint arXiv:1602.08465.
   """

class SeqNmsTracklet(Tracklet):
    """Tracklet subclass for use with seqNMS. Keeps track of box indicies and sequence score and confidences"""
    def __init__(self, id):
        super().__init__(id)
        self.box_indexes = []
        self.sequence_conf = 0
        self.sequence_score = 0

    def add_box(self, box, box_index, frame_index):
        self.boxes.append(box)
        self.box_indexes.append(box_index)
        self.frame_indexes.append(frame_index)

        if self.start_frame is None: self.start_frame = frame_index

        if frame_index < self.start_frame: self.start_frame = frame_index
        if frame_index > self.end_frame: self.end_frame = frame_index

    def tracklet_conf(self, type):
        """Calculates tracklet confidence"""
        conf_values = [box[4] for box in self.boxes]
        if type == "avg":
            return sum(conf_values)/len(conf_values)
        if type == "max":
            return max(conf_values)
        
    def set_conf(self, type, update_boxes=False):
        """Calculates and sets the tracklet confidence, optionally update box conf for drawing"""
        conf = self.tracklet_conf(type)
        self.sequence_conf = conf

        if update_boxes: 
            for box in self.boxes: box[4] = conf
        

def select_sequence(frame_preds, id):
    """Selects the sequence of linked boxes with the highest confidence sum using a dynamic programming algorithm"""
    best_score = 0
    best_sequence = []

    previous_scores = [box[4] for box in frame_preds[-1]]
    previous_sequences = [[(len(frame_preds) - 1, i)] for i in range(len(previous_scores))]

    for frame_index in tqdm(range(len(frame_preds) - 2, -1, -1), bar_format="{l_bar}{bar:20}{r_bar}"):
        current_scores = []
        current_sequences = []

        boxes = frame_preds[frame_index]

        #if there are no boxes in this frame, skip it
        if not boxes: continue

        #If there are no boxes in the previous frame
        if not frame_preds[frame_index + 1]:
            for box_index in range(len(boxes)):
                current_scores.append(boxes[box_index][4])
                current_sequences.append([(frame_index, box_index)])
                previous_scores = current_scores
                previous_sequences = current_sequences

            continue

        #Computes the iou between every box in this frame and the next frame
        iou_mat = iou_matrix(boxes, frame_preds[frame_index + 1])

        for box_index in range(len(boxes)):
            conf = boxes[box_index][4]
            label = boxes[box_index][5]

            #find linked boxes in previous frame with the same class            
            linked_box_indexes = np.where((iou_mat[box_index] >= 0.5) & 
                                          (label == np.array(frame_preds[frame_index + 1])[:,5]))[0]
            
            if linked_box_indexes.shape[0]:
                #choose the linked box with the highest score
                selected_index = linked_box_indexes[np.argmax(np.array(previous_scores)[linked_box_indexes])]
                selected_score = previous_scores[selected_index]

                #set the current score and sequence
                current_score = conf + selected_score
                current_scores.append(current_score)

                current_sequence = [(frame_index, box_index)] + previous_sequences[selected_index]
                current_sequences.append(current_sequence)
            else:
                #if there are no linked boxes
                current_score = conf
                current_scores.append(current_score)
                current_sequence = [(frame_index, box_index)]
                current_sequences.append(current_sequence)

            #if this is the new best score
            if current_score >= best_score:
                best_score = current_score
                best_sequence = current_sequence

        previous_scores = current_scores
        previous_sequences = current_sequences

    #Create trackelt object
    tracklet = SeqNmsTracklet(id)
    tracklet.sequence_score = best_score

    for (frame_index, box_index) in best_sequence:
        tracklet.add_box(frame_preds[frame_index][box_index], box_index, frame_index)

    print(f"Tracklet found - Score: {best_score}, length: {len(best_sequence)}")
    return tracklet


@silence
def Seq_nms(model, video, nms_iou = 0.6, avg_conf_th = 0.2, early_stopping_score_th = 0.8 ,no_save=False):
    """Implements Seq_nms from Han, W. et al (2016)"""
    start_time = time.time()
    original_conf = model.conf
    original_iou = model.iou
    model.update_parameters(conf=0.01, iou=0.85) #update parameters to effectively skip NMS

    print(f"Frames: {video.num_of_frames}")
    frame_predictions = [model.xywhcl(frame) for frame in video]
    remaining_boxes = sum([len(frame_pred) for frame_pred in frame_predictions])
    print(f"Total box predictions: {remaining_boxes}")
    print(f"Avg predictions per frame: {remaining_boxes/video.num_of_frames}")

    #reset model parameters
    model.update_parameters(conf=original_conf, iou=original_iou)

    detected_tracklets = []
    id_counter = 0
    print("Begining SeqNMS")
    while remaining_boxes > 0:
        print(f"----- Selecting sequence {id_counter} -----")
        print(f"Total remaining boxes: {remaining_boxes}")

        #detected the sequence with the max score
        tracklet = select_sequence(frame_predictions, id_counter)

        #early stopping if scores get low
        if tracklet.sequence_score < early_stopping_score_th: 
            print(f"Ending SeqNMS early - best tracklet has score less than {early_stopping_score_th}")
            break

        #only keep tracklets with good average confidence
        tracklet.set_conf("avg")
        if tracklet.sequence_conf >= avg_conf_th:
            detected_tracklets.append(tracklet)
            id_counter += 1
        else:
            print(f"Sequence avg confidence too low ({tracklet.sequence_conf} < {avg_conf_th}) - rejecting tracklet")

        #Non maximal supression
        boxes_removed = 0
        for frame_index, box_index, target_box in zip(tracklet.frame_indexes, tracklet.box_indexes, tracklet.boxes):
            boxes = frame_predictions[frame_index]
            boxes.pop(box_index)

            if len(boxes) == 0:
                boxes_removed += 1
                continue

            label = target_box[5]
            iou_mat = iou_matrix([target_box], boxes)[0]
            overlapping_box_indexes = [i for i in range(len(iou_mat)) 
                                  if iou_mat[i] >= nms_iou and 
                                  label == frame_predictions[frame_index][i][5]]
            
            for i in sorted(overlapping_box_indexes, reverse=True): 
                boxes.pop(i)

            boxes_removed += len(overlapping_box_indexes) + 1

        print(f"NMS complete - {boxes_removed} boxes removed")
        remaining_boxes = sum([len(frame_pred) for frame_pred in frame_predictions])
    
    ts = TrackletSet(video, detected_tracklets, model.num_to_class)

    duration = round((time.time() - start_time)/60, 2)
    print(f"Finished SeqNMS in {duration}mins ({round(duration/video.num_of_frames, 4)}mins per frame)")
    if not no_save: save_VOD(ts, "seqNMS")
    return ts

    
#def seqNMS_grid_search(videos, gt_tracklets, detector, nms_ious, avg_conf_ths, es_score_ths):
#    best_metrics = None
#    best_mota = 0
#    best_parameters = None
#    counter = 0
#
#    for a, b, c in product(nms_ious, avg_conf_ths, es_score_ths):
#        components = np.zeros((10,))
#        for i, vid in enumerate(videos):
#            pred_tracklets = frame_skipping(vid, Seq_nms, detector, 2, nms_iou=a, avg_conf_th=b, early_stopping_score_th=c, silence=True)
#            components += single_vid_metrics(gt_tracklets[i], pred_tracklets, return_components=True)
#
#        metrics = metrics_from_components(components)
#
#        print(f"Finished search {counter} - {(a, b, c)}")
#        counter += 1
#
#        if metrics[2] > best_mota:
#            best_mota = metrics[2]
#           best_metrics = metrics
#            best_parameters = (a, b, c)
#            print("Found new best:")
#            print(best_parameters)
#            print(best_metrics)
#
#     
#
#    print(f"Grid search finished  - searched over {counter} combinations")
#    print(f"Best parameters:")
#    print(f"NMS_iou = {best_parameters[0]}, avg_conf_th = {best_parameters[1]}, early_stopping_score_th = {best_parameters[2]}")
#    print_metrics(*best_metrics)