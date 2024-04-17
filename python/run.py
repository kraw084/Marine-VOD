import torch
import os
from Detectors import create_urchin_model
from VOD_utils import frame_by_frame_VOD
from Video_utils import Video


if __name__ == "__main__":
    cuda = torch.cuda.is_available()
    urchin_bot = create_urchin_model(cuda)

    urchin_video_folder = "E:/urchin video/All/"

    for vid_name in os.listdir(urchin_video_folder):
        vid = Video(urchin_video_folder + vid_name)
        print("Finished loading video")
        
        frame_by_frame_VOD(urchin_bot, vid, 0, 1500)