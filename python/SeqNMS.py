from Video_utils import video_to_frames
from VOD_utils import iou_matrix

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


def select_sequence(frame_preds, id):
    best_score = 0
    best_sequence = []

    previous_scores = [box[4] for box in frame_index[-1]]
    previous_sequences = [[(len(frame_index) - 1, prev_score)] for prev_score in previous_scores]

    for frame_index in range(len(frame_preds) - 2, -1, -1):
        current_scores = []
        current_sequences = []
        boxes = frame_preds[frame_index]

        #If there are no boxes in the previous frame
        if not frame_preds[frame_index + 1]:
            for box_index in range(len(boxes)):
                current_scores.append(boxes[box_index][4])
                current_sequences.append([(frame_index, box_index)])

        iou_mat = iou_matrix(boxes, frame_preds[frame_index + 1])

        for box_index in range(len(boxes)):
            conf = boxes[box_index][4]
            label = boxes[box_index][5]

            #find linked boxes in previous frame with the same class
            linked_box_indexes = [i for i in range(len(iou_mat[box_index])) 
                                  if iou_mat[box_index][i] >= 0.5 and 
                                  label == frame_preds[frame_index + 1][i][5]]
            
            #choose the linked box with the highest score
            selected_index = linked_box_indexes[0]
            selected_score = previous_scores[selected_index]
            for i in linked_box_indexes[1:]:
                if previous_scores[i] > selected_score:
                    selected_index = i
                    selected_score = previous_scores[i]
            
            #set the current score and sequence
            current_score = conf + selected_score
            current_scores.append(current_score)

            current_sequence = [(frame_index, box_index)] + previous_sequences[selected_index]
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

    print(f"Tracklet found - Score: {best_score}, length: {len(best_sequence)}")
    return tracklet

def Seq_nms(model, video_path):
    """Implements Seq_nms from Han, W. et al (2016)"""
    model.update_parameters(conf=0.001, iou=1) #update parameters to effectively skip NMS

    frames = video_to_frames(video_path)[0]
    print(f"Frames: {len(frames)}")
    frame_predictions = [model.xywhcl(frame) for frame in frames]
    print(f"Total box predictions: {len(frame_predictions)}")
    print(f"Avg predictions per frame: {len(frame_predictions)/len(frames)}")


