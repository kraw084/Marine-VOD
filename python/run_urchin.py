import os

import matplotlib.pyplot as plt
import matplotlib
import cv2
import numpy as np

from mv_utils import Detectors, Video_utils, VOD_utils, Eval_utils, Cmc, Config
from vod_methods import fbf, SeqNMS, sort, bot_sort, byte_track, oc_sort, deep_sort
from reid.reid import create_reid_model
from reid.eval_reid import random_view_similarity

if __name__ == "__main__":

    urchin_bot =  Detectors.create_urchin_model(Config.Config.cuda)
    urchin_video_folder = Config.Config.urchin_vid_path

    urchin_reid_model = create_reid_model()

    enable_fbf = False
    enable_seqNMS = False
    enable_SORT = False
    enable_BoTSORT = False
    enable_ByteTrack = False
    enable_OCSORT = False
    enable_DeepSORT = True

    start = 0
    count = 0
    for vid_name in os.listdir(urchin_video_folder):
        if count < start:
            count += 1
            continue

        count += 1
        
        vid = Video_utils.Video(urchin_video_folder + "/" + vid_name)
        #random_view_similarity(vid, urchin_bot, urchin_reid_model)
        
        if enable_fbf:
            fbf_tracklets = fbf.frame_by_frame_VOD_with_tracklets(urchin_bot, vid, True) 
            target_tracklets = fbf_tracklets

        if enable_seqNMS:
            seqNMS_tracklets = VOD_utils.frame_skipping(vid, SeqNMS.Seq_nms, urchin_bot, 1, silence=True)
            target_tracklets = seqNMS_tracklets

        if enable_SORT:
            #sort_tracklets = sort.SORT(urchin_bot, vid, iou_min=0.3, t_lost=8, probation_timer=3, min_hits=5, no_save=True, silence=False)
            vid1 = Video_utils.Video(urchin_video_folder + "/" + vid_name)
            sort_tracklets = sort.SORT(urchin_bot, vid1, iou_min=0.3, t_lost=30, probation_timer=3, min_hits=5, no_save=True, silence=False, kf_est_for_unmatched=False)
            VOD_utils.interpoalte_tracklet_set(sort_tracklets)
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

        if enable_DeepSORT:
            deep_sort_tracklets = deep_sort.Deep_SORT(urchin_bot, vid, iou_min=0.0, t_lost=20, probation_timer=5, min_hits=10, no_save=True, silence=False,
                                                      lambda_iou=0.8, reid_model=urchin_reid_model, sim_min=0.5)
            VOD_utils.interpoalte_tracklet_set(deep_sort_tracklets)
            target_tracklets = deep_sort_tracklets

        target_tracklets.draw_tracklets()
        target_tracklets.video.play(1300, start_paused=True)

        #sort_tracklets.draw_tracklets()
        #deep_sort_tracklets.draw_tracklets()
        #stitched_video = Video_utils.stitch_video(sort_tracklets.video, deep_sort_tracklets.video, "sort_vs_deep_sort.mp4")
        #stitched_video.play(1500, start_paused = True)