import os

from mv_utils.Config import Config
from mv_utils.Detectors import create_urchin_model
from mv_utils.Video_utils import Video, stitch_video
from mv_utils.VOD_utils import TrackletSet, frame_skipping

from vod_methods.fbf import frame_by_frame_VOD_with_tracklets
from vod_methods.SeqNMS import Seq_nms
from vod_methods.sort import SORT
from vod_methods.bot_sort import BoT_SORT
from vod_methods.byte_track import ByteTrack

if __name__ == "__main__":

    urchin_bot =  create_urchin_model(Config.cuda)
    urchin_video_folder = f"{Config.drive}:/urchin video/All/"

    start = 0
    count = 0
    for vid_name in os.listdir(urchin_video_folder):
        if count < start:
            count += 1
            continue

        count += 1
        
        vid = Video(urchin_video_folder + vid_name)
        print("Finished loading video")

        #sort_tracklets = SORT(urchin_bot, vid, iou_min=0.3, t_lost=8, probation_timer=3, min_hits=5, no_save=True, silence=False)
        
        bot_sort_tracklets = BoT_SORT(urchin_bot, vid, iou_min=0.3, t_lost=8, probation_timer=3, min_hits=5, no_save=True, silence=False)
        bot_sort_tracklets.draw_tracklets()
        bot_sort_tracklets.video.play(1500, start_paused=True)
     
        #byte_tracklets = ByteTrack(urchin_bot, vid, iou_min=0.2, t_lost=15, probation_timer=3, min_hits=5, low_conf_th=0.45, no_save=True, silence=False)
        #byte_tracklets.draw_tracklets()
        #byte_tracklets.video.play(1500, start_paused=True)
     