import os
os.environ["TQDM_DISABLE"] = "1"

import torch
import numpy as np

from Detectors import create_brackish_model
from Video_utils import Video, stitch_video
from VOD_utils import (frame_by_frame_VOD, frame_by_frame_VOD_with_tracklets, 
                       TrackletSet, frame_skipping, single_vid_metrics, print_metrics, 
                       save_VOD, metrics_from_components)

from SeqNMS import Seq_nms
from sort import SORT

from BrackishMOT import brackishMOT_tracklet, id_by_set


if __name__ == "__main__":
    cuda = torch.cuda.is_available()

    brackish_bot = create_brackish_model(cuda)
    brackish_video_folder = "E:/Marine-VOD/BrackishMOT/videos/"

    enable_gt = True
    enable_fbf = False
    enable_seqNMS = False
    enable_SORT = True

    data_set = "val"
    ids = id_by_set(data_set)
    print(f"{len(ids)} ids in set {data_set}")

    components = np.zeros((10,))

    for id in ids:
        vid_name = f"brackishMOT-{id:02}.mp4"
        #get ground truth tracklets
        if enable_gt:
            vid = Video(brackish_video_folder + vid_name)
            gt_tracklets = TrackletSet(vid, brackishMOT_tracklet(id), brackish_bot.num_to_class)

        #get fbf tracklets
        if enable_fbf:
            vid2 = Video(brackish_video_folder + vid_name)
            fbf_tracklets = frame_by_frame_VOD_with_tracklets(brackish_bot, vid2, True)

        #get seqNMS tracklets
        if enable_seqNMS:
            vid3 = Video(brackish_video_folder + vid_name)
            seqNMS_tracklet_set = frame_skipping(vid3, Seq_nms, brackish_bot, 1, nms_iou=0.4, avg_conf_th=0.3, early_stopping_score_th=0.5, silence=True)

        #get sort tracklets
        if enable_SORT:
            vid4 = Video(brackish_video_folder + vid_name)
            #sort_tracklet_set = frame_skipping(vid4, SORT, brackish_bot, 1, iou_min=0.3, t_lost=5, min_hits=12, greedy_assoc=True, silence=True)
            sort_tracklet_set = SORT(brackish_bot, vid4, 0.3, 5, 12, True, True, silence=True)


        target_tracklets = sort_tracklet_set
        gt_ids, pred_ids = single_vid_metrics(gt_tracklets, target_tracklets, True)[-2:]
        gt_tracklets.draw_tracklets(gt_ids)
        target_tracklets.draw_tracklets(pred_ids)

        stitched_vid = stitch_video(gt_tracklets.video, target_tracklets.video, "gt_SORT.mp4")
        stitched_vid.play(1500, start_paused=True)
        
        comp = single_vid_metrics(gt_tracklets, target_tracklets, return_components=True)  
        print_metrics(*metrics_from_components(comp))
        components += comp
        
    print("Overall metrics:")
    print_metrics(*metrics_from_components(components))