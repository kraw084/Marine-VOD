import torch
import os

from Detectors import create_urchin_model
from Video_utils import Video

from VOD_utils import frame_by_frame_VOD
from SeqNMS import Seq_nms


if __name__ == "__main__":
    cuda = torch.cuda.is_available()
    urchin_bot = create_urchin_model(cuda)

    urchin_video_folder = "E:/urchin video/All/"

    for vid_name in os.listdir(urchin_video_folder):
        vid = Video(urchin_video_folder + vid_name)
        print("Finished loading video")

        #frame_by_frame_VOD(urchin_bot, vid)
        #vid.play(1500)

        seqNMSTracklets = Seq_nms(urchin_bot, vid)
        seqNMSTracklets.video.play(1500, start_paused=True)

        break
