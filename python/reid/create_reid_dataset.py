import os
import sys
import math
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2

from mv_utils.Config import Config
from mv_utils.Video_utils import Video
from mv_utils.Detectors import create_urchin_model
from vod_methods.byte_track import ByteTrack
from datasets.urchin_videos import urchin_gt_generator

urchin_bot = create_urchin_model(Config.cuda)
video_folder = Config.urchin_vid_path

padding = 0.3
split_to_make = "test"


if split_to_make == "train":
    dataset_folder = "C:/Users/kraw084/OneDrive - The University of Auckland/Desktop/reid_dataset_train"

    video_names = ["DSC_1802.MP4",  
                "DSC_1876.MP4",   
                "DSC_2168.MP4", 
                "DSC_2172.MP4",  
                "DSC_4231.MP4",   
                "DSC_4232.MP4",  
                "DSC_4778.MP4",  
                "DSC_4779.MP4",   
                "DSC_4780.MP4",   
                "DSC_4951.MP4",   
                "DSC_4952.MP4",    
                "DSC_5639.MP4",   
                "DSC_5808.MP4",  
                "DSC_5814.MP4",  
                "DSC_6014.MP4",   
                "DSC_6225.MP4",   
                "DSC_7840.MP4",  
                "DSC_7848.MP4",  
                "DSC_7915.MP4"
                ]

else:
    dataset_folder = f"C:/Users/kraw084/OneDrive - The University of Auckland/Desktop/reid_dataset_{split_to_make}"
    gen = urchin_gt_generator(split_to_make)
    video_names = list(range(5))


os.mkdir(dataset_folder)
data = []

global_id = 0
for name in video_names:
    ids = []

    if split_to_make == "train":
        vid = Video(video_folder + "/" + name)
        tracklets = ByteTrack(urchin_bot, vid, iou_min=0.3, t_lost=30, probation_timer=5, min_hits=10, no_save=True, silence=False)
        video_data = {"video_name": name} 
    else:
        vid, tracklets = next(gen)
        video_data = {"video_name": vid.full_name}

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

        ids.append(global_id)
        global_id += 1
        
    video_data["count"] = len(ids)
    video_data["ids"] = ids
    data.append(video_data)

with open(dataset_folder + "/video_data.json", "w") as f:
    json.dump(data, f)