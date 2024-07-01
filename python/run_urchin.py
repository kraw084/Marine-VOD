import os

from utils.Config import Config
from utils.Detectors import create_urchin_model
from utils.Video_utils import Video, stitch_video
from utils.VOD_utils import TrackletSet, frame_skipping

from vod_methods.fbf import frame_by_frame_VOD, frame_by_frame_VOD_with_tracklets
from vod_methods.SeqNMS import Seq_nms
from vod_methods.sort import SORT, play_sort_with_kf
from vod_methods.bot_sort import BoT_SORT

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


        sort_tracklets = SORT(urchin_bot, vid, iou_min=0.3, t_lost=8, probation_timer=3, min_hits=5, no_save=True, silence=False)

        vid1 = Video(urchin_video_folder + vid_name)
        bot_sort_tracklets = BoT_SORT(urchin_bot, vid1, iou_min=0.3, t_lost=8, probation_timer=3, min_hits=5, no_save=True, silence=False)

        
        sort_tracklets.draw_tracklets()
        bot_sort_tracklets.draw_tracklets()

        stitched_video = stitch_video(sort_tracklets.video, bot_sort_tracklets.video, "sort_vs_bot-sort.mp4")
        stitched_video.play(1800, start_paused = True)


     