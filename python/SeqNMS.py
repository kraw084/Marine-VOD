from Video_utils import video_to_frames
from VOD_utils import iou_matrix
import numpy as np

class SeqNmsTracklet:
    def __init__(self, id):
        self.boxes = []
        self.box_index = []
        self.frame_indexes = []
        self.id = id

    def add_box(self, box, box_index, frame_index):
        self.boxes.append(box)
        self.box_index.append(box_index)
        self.frame_indexes.append(frame_index)

    def tracklet_conf(self, type):
        conf_values = [box[4] for box in self.boxes]
        if type == "avg":
            return sum(conf_values)/len(conf_values)
        if type == "max":
            return max(conf_values)
        
    def tracklet_length(self):
        return len(self.boxes)
    
    def id(self):
        return self.id
    
    def __iter__(self):
        self.__i = 0
        return self
    
    def __next__(self):
        val = (self.frame_indexes[self.__i], self.box_index[self.__i], self.boxes[self.__i])
        self.__i += 1
        return val


def select_sequence(frame_preds, id):
    best_score = 0
    best_sequence = []

    previous_scores = [box[4] for box in frame_index[-1]]
    previous_sequences = [[(len(frame_index) - 1, prev_score)] for prev_score in previous_scores]

    for frame_index in range(len(frame_preds) - 2, -1, -1):
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

        iou_mat = iou_matrix(boxes, frame_preds[frame_index + 1])

        for box_index in range(len(boxes)):
            conf = boxes[box_index][4]
            label = boxes[box_index][5]

            #find linked boxes in previous frame with the same class            
            linked_box_indexes = np.where((iou_mat[box_index] >= 0.5) & 
                                          (label == np.array(frame_preds[frame_index + 1])[:,5]))[0]
            
            if linked_box_indexes:
                #choose the linked box with the highest score
                selected_index = linked_box_indexes[np.argmax(np.array(previous_scores)[linked_box_indexes])]
                selected_score = previous_scores[selected_index]

                #set the current score and sequence
                current_score = conf + selected_score
                current_scores.append(current_score)

                current_sequence = [(frame_index, box_index)] + previous_sequences[selected_index]
                current_sequences.append(current_sequences)
            else:
                #if there are no linked boxes
                current_score = conf
                current_scores.append(current_score)
                current_sequence = [(frame_index, box_index)]
                current_sequences.append(current_sequences)

            #if this is the new best score
            if current_score >= best_score:
                best_score = current_score
                best_sequence = current_sequence

        previous_scores = current_scores
        previous_sequences = current_sequences

    #Create trackelt object
    tracklet = SeqNmsTracklet(id)
    for (frame_index, box_index) in best_sequence:
        tracklet.add_box(frame_preds[frame_index][box_index], box_index, frame_index)

    print(f"Tracklet {id} found - Score: {best_score}, length: {len(best_sequence)}")
    return tracklet


def Seq_nms(model, video_path, nms_iou = 0.6):
    """Implements Seq_nms from Han, W. et al (2016)"""
    model.update_parameters(conf=0.001, iou=1) #update parameters to effectively skip NMS

    frames = video_to_frames(video_path)[0]
    print(f"Frames: {len(frames)}")
    frame_predictions = [model.xywhcl(frame) for frame in frames]
    print(f"Total box predictions: {len(frame_predictions)}")
    print(f"Avg predictions per frame: {len(frame_predictions)/len(frames)}")

    detected_tracklets = []
    id_counter = 0
    remaining_boxes = sum([len(frame_pred) for frame_pred in frame_predictions])
    while remaining_boxes > 0:
        print(f"Total remaining boxes: {remaining_boxes}")

        #detected the sequence with the max score
        tracklet = select_sequence(frame_predictions, id_counter)
        detected_tracklets.append(tracklet)
        id_counter += 1

        #Non maximal supression
        boxes_removed = 0
        for frame_index, box_index, target_box in tracklet:
            boxes = frame_predictions[frame_index]
            boxes.pop(box_index)

            label = target_box[5]
            iou_mat = iou_matrix([target_box], boxes)[0]
            overlapping_box_indexes = [i for i in range(len(iou_mat)) 
                                  if iou_mat[i] >= nms_iou and 
                                  label == frame_predictions[frame_index][i][5]]
            
            for i in overlapping_box_indexes: boxes.pop(i)
            boxes_removed += len(overlapping_box_indexes) + 1
        print(f"NMS complete - {boxes_removed} boxes removed")

        remaining_boxes = sum([len(frame_pred) for frame_pred in frame_predictions])

    return detected_tracklets
