import os
#os.environ["TQDM_DISABLE"] = "1"

from Config import Config
from Detectors import create_urchin_model
from Video_utils import Video, stitch_video
from VOD_utils import (frame_by_frame_VOD, frame_by_frame_VOD_with_tracklets, 
                       TrackletSet, frame_skipping, single_vid_metrics, print_metrics, 
                       save_VOD, metrics_from_components)

from SeqNMS import Seq_nms
from sort import SORT, play_sort_with_kf
from bot_sort import BoT_SORT
from cmc import show_flow

if __name__ == "__main__":

    urchin_bot =  create_urchin_model(Config.cuda)
    urchin_video_folder = f"{Config.drive}:/urchin video/All/"

    start = 7
    count = 0
    for vid_name in os.listdir(urchin_video_folder):
        count += 1
        if count - 1 <= start:
            continue


        vid = Video(urchin_video_folder + vid_name)
        print("Finished loading video")


        #show_flow(vid)

        sort_tracklets = SORT(urchin_bot, vid, iou_min=0.3, t_lost=8, probation_timer=3, min_hits=5, no_save=True, silence=False)

        vid1 = Video(urchin_video_folder + vid_name)
        bot_sort_tracklets = BoT_SORT(urchin_bot, vid1, iou_min=0.3, t_lost=8, probation_timer=3, min_hits=5, no_save=True, silence=False)

        
        sort_tracklets.draw_tracklets()
        bot_sort_tracklets.draw_tracklets()

        stitched_video = stitch_video(sort_tracklets.video, bot_sort_tracklets.video, "sort_vs_bot-sort.mp4")
        stitched_video.play(1800, start_paused = True)


     