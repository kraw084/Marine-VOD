import os
import cv2
import numpy as np

from mv_utils import Detectors, Video_utils, VOD_utils, Eval_utils, Cmc, Config
from vod_methods import fbf, SeqNMS, sort, bot_sort, byte_track, oc_sort


if __name__ == "__main__":

    #urchin_bot =  Detectors.create_urchin_model(Config.Config.cuda)
    urchin_video_folder = Config.Config.urchin_vid_path

    enable_fbf = False
    enable_seqNMS = False
    enable_SORT = False
    enable_BoTSORT = True
    enable_ByteTrack = False
    enable_OCSORT = False

    start = 0
    count = 0
    for vid_name in os.listdir(urchin_video_folder):
        if count < start:
            count += 1
            continue

        count += 1
        
        vid = Video_utils.Video(urchin_video_folder + "/" + vid_name)
        if vid.fps > 60: continue

        Video_utils.sample_frames(vid, 5)
        vid.play(start_paused=True)
        Video_utils.save_as_frames(vid, r"C:\Users\kraw084\OneDrive - The University of Auckland\Desktop\sampled_videos")
        print("Finished\n")

        continue

        if enable_fbf:
            fbf_tracklets = fbf.frame_by_frame_VOD_with_tracklets(urchin_bot, vid, True)
            target_tracklets = fbf_tracklets

        if enable_seqNMS:
            seqNMS_tracklets = VOD_utils.frame_skipping(vid, SeqNMS.Seq_nms, urchin_bot, 1, silence=True)
            target_tracklets = seqNMS_tracklets

        if enable_SORT:
            sort_tracklets = sort.SORT(urchin_bot, vid, iou_min=0.3, t_lost=8, probation_timer=3, min_hits=5, no_save=True, silence=False)
            target_tracklets = sort_tracklets

        if enable_BoTSORT:
            bot_sort_tracklets = bot_sort.BoT_SORT(urchin_bot, vid, iou_min=0.3, t_lost=30, probation_timer=3, min_hits=5, no_save=True, silence=False)
            VOD_utils.interpoalte_tracklet_set(bot_sort_tracklets)
            target_tracklets = bot_sort_tracklets
            
        if enable_ByteTrack:
            byte_track_tracklets = byte_track.ByteTrack(urchin_bot, vid, iou_min=0.3, t_lost=30, probation_timer=5, min_hits=10, no_save=True, silence=False)
            VOD_utils.interpoalte_tracklet_set(byte_track_tracklets)
            target_tracklets = byte_track_tracklets
            
        if enable_OCSORT:
            oc_sort_tracklets = oc_sort.OC_SORT(urchin_bot, vid, iou_min=0.3, t_lost=30, probation_timer=3, min_hits=5, no_save=True, silence=False)
            VOD_utils.interpoalte_tracklet_set(oc_sort_tracklets)
            target_tracklets = oc_sort_tracklets      


        target_tracklets.video.play(1800, start_paused=True)  
        
        