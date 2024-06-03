import os
os.environ["TQDM_DISABLE"] = "1"

from Config import Config
from Detectors import create_urchin_model
from Video_utils import Video, stitch_video
from VOD_utils import (frame_by_frame_VOD, frame_by_frame_VOD_with_tracklets, 
                       TrackletSet, frame_skipping, single_vid_metrics, print_metrics, 
                       save_VOD, metrics_from_components)

from SeqNMS import Seq_nms
from sort import SORT, play_sort_with_kf

if __name__ == "__main__":
    count = 0

    urchin_bot =  create_urchin_model(Config.cuda)
    urchin_video_folder = f"{Config.drive}:/urchin video/All/"

    for vid_name in os.listdir(urchin_video_folder):
        vid = Video(urchin_video_folder + vid_name)
        print("Finished loading video")
        #vid.play()

        #frame_by_frame_VOD(urchin_bot, vid)
        #vid.play(1500)

        #seqNMSTracklets = Seq_nms(urchin_bot, vid)
        #seqNMSTracklets.video.play(1500, start_paused=True)

        sortTracklets = SORT(urchin_bot, vid, iou_min=0.3, t_lost=8, probation_timer=3, min_hits=10, greedy_assoc=False, no_save=True)
        #sortTracklets.draw_tracklets()
        #sortTracklets.video.play(1500, start_paused=True)

        play_sort_with_kf(sortTracklets)

        count += 1
        if count > 5: break