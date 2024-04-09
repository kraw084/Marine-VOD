from Video_utils import video_to_frames

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


def select_sequence(frame_predictions):
    best_frame_index = len(frame_predictions) - 1
    best_box_index = 0
    previous_scores = [box[4] for box in frame_index[-1]]

    for frame_index in range(len(frame_predictions) - 2, -1, -1):
        pass

def Seq_nms(model, video_path):
    """Implements Seq_nms from Han, W. et al (2016)"""
    model.update_parameters(conf=0.001, iou=1)
    frames = video_to_frames(video_path)[0]
    print(f"Frames: {len(frames)}")
    frame_predictions = [model.xywhcl(frame) for frame in frames]
    print(f"Total box predictions: {len(frame_predictions)}")
    print(f"Avg predictions per frame: {len(frame_predictions)/len(frames)}")


