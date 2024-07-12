import os

from mv_utils import Detectors, Video_utils, VOD_utils, Eval_utils, Cmc, Config
from vod_methods import fbf, SeqNMS, sort, bot_sort, byte_track, oc_sort


if __name__ == "__main__":

    urchin_bot =  Detectors.create_urchin_model(Config.Config.cuda)
    urchin_video_folder = f"{Config.Config.drive}:/urchin video/All/"

    start = 0
    count = 0
    for vid_name in os.listdir(urchin_video_folder):
        if count < start:
            count += 1
            continue

        count += 1
        
        vid = Video_utils.Video(urchin_video_folder + vid_name)
        print("Finished loading video")
        
        #Cmc.show_flow(vid)
        #vid.play(start_paused=True)

        #sort_tracklets = SORT(urchin_bot, vid, iou_min=0.3, t_lost=8, probation_timer=3, min_hits=5, no_save=True, silence=False)
        
        bot_sort_tracklets = bot_sort.BoT_SORT(urchin_bot, vid, iou_min=0.3, t_lost=30, probation_timer=3, min_hits=5, no_save=True, silence=False)
        VOD_utils.interpoalte_tracklet_set(bot_sort_tracklets)
        
        bot_sort_tracklets.draw_tracklets()
        bot_sort_tracklets.video.play(1500, start_paused=True)
     
        #byte_tracklets = ByteTrack(urchin_bot, vid, iou_min=0.2, t_lost=15, probation_timer=3, min_hits=5, low_conf_th=0.45, no_save=True, silence=False)
        #byte_tracklets.draw_tracklets()
        #byte_tracklets.video.play(1500, start_paused=True)
     