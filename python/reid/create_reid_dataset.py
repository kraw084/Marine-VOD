import os
import sys
import math

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2

from mv_utils.Config import Config
from mv_utils.Video_utils import Video
from mv_utils.Detectors import create_urchin_model
from vod_methods.byte_track import ByteTrack

urchin_bot = create_urchin_model(Config.cuda)
video_folder = Config.urchin_vid_path

padding = 0.3

dataset_folder = "C:/Users/kraw084/OneDrive - The University of Auckland/Desktop/reid_dataset_v2"
os.mkdir(dataset_folder)

#video_names = os.listdir(video_folder)
video_names = [
"DSC_2168.MP4", 
"DSC_4778.MP4",  
"DSC_4952.MP4",    
"DSC_5814.MP4",  
"DSC_7840.MP4",  
"DSC_7848.MP4"
]

global_id = 0
for name in video_names:
    vid = Video(video_folder + "/" + name)

    tracklets = ByteTrack(urchin_bot, vid, iou_min=0.3, t_lost=30, probation_timer=5, min_hits=10, no_save=True, silence=False)
    for tracklet in tracklets:
        os.mkdir(dataset_folder + "/" + str(global_id))

        local_id = 0    
        for frame_index, box in tracklet:
            frame = vid.frames[frame_index]
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
            x_center, y_center, w, h = box[0], box[1], box[2], box[3]
            w = math.ceil(w * (1 + padding))
            h = math.ceil(h * (1 + padding))

            urchin_image = frame[int(y_center - h / 2):int(y_center + h / 2), int(x_center - w / 2):int(x_center + w / 2)]
            cv2.imwrite(dataset_folder + "/" + str(global_id) + "/" + str(local_id) + ".jpg", urchin_image)
            local_id += 1

        global_id += 1