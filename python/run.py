import os
os.environ["TQDM_DISABLE"] = "0"

import torch
import numpy as np

from Detectors import create_urchin_model, create_brackish_model
from Video_utils import Video, stitch_video
from VOD_utils import (frame_by_frame_VOD, frame_by_frame_VOD_with_tracklets, 
                       TrackletSet, frame_skipping, single_vid_metrics, print_metrics, 
                       save_VOD, metrics_from_components)

from SeqNMS import Seq_nms, seqNMS_grid_search
from sort import SORT
from BrackishMOT import brackishMOT_tracklet, id_by_set


if __name__ == "__main__":
    cuda = torch.cuda.is_available()

    count = 0
    if False:
        urchin_bot =  create_urchin_model(cuda)
        urchin_video_folder = "E:/urchin video/All/"

        for vid_name in os.listdir(urchin_video_folder):
            vid = Video(urchin_video_folder + vid_name)
            print("Finished loading video")
            #vid.play()

            #frame_by_frame_VOD(urchin_bot, vid)
            #vid.play(1500)

            #seqNMSTracklets = Seq_nms(urchin_bot, vid)
            #seqNMSTracklets.video.play(1500, start_paused=True)

            sortTracklets = frame_skipping(vid, SORT, urchin_bot, 1, iou_min=0.5, t_lost=5, min_hits=10)
            sortTracklets.draw_tracklets()
            sortTracklets.video.play(1500, start_paused=True)
            #save_VOD(sortTracklets, "SORT")

            count += 1
            if count > 5: break

    if True:
        brackish_bot = create_brackish_model(cuda)
        brackish_video_folder = "E:/Marine-VOD/BrackishMOT/videos/"

        enable_gt = True
        enable_fbf = True
        enable_seqNMS = False
        enable_SORT = False

        data_set = "val"
        ids = id_by_set(data_set)
        print(f"{len(ids)} ids in set {data_set}")

        components = np.zeros((10,))

        #videos = [Video(brackish_video_folder + vid_name) for vid_name in [f"brackishMOT-{id:02}.mp4" for id in ids]]
        #gt_tracklets = [TrackletSet(vid, brackishMOT_tracklet(id), brackish_bot.num_to_class) for id, vid in zip(ids, videos)]
        #values = [0.2 * i for i in range(1, 5)]
        #seqNMS_grid_search(videos, gt_tracklets, brackish_bot, list(values), list(values), list(values))

        if True:
            for id in ids:
                vid_name = f"brackishMOT-{id:02}.mp4"
                #get ground truth tracklets
                if enable_gt:
                    vid = Video(brackish_video_folder + vid_name)
                    gt_tracklets = TrackletSet(vid, brackishMOT_tracklet(id), brackish_bot.num_to_class)
                    #gt_tracklets.draw_tracklets()

                #get fbf tracklets
                if enable_fbf:
                    vid2 = Video(brackish_video_folder + vid_name)
                    fbf_tracklets = frame_by_frame_VOD_with_tracklets(brackish_bot, vid2, True)
                    #fbf_tracklets.draw_tracklets()

                #get seqNMS tracklets
                if enable_seqNMS:
                    vid3 = Video(brackish_video_folder + vid_name)
                    seqNMS_tracklet_set = frame_skipping(vid3, Seq_nms, brackish_bot, 1, nms_iou=0.4, avg_conf_th=0.3, early_stopping_score_th=0.5, silence=True)
                    seqNMS_tracklet_set.draw_tracklets()

                #get sort tracklets
                if enable_SORT:
                    vid4 = Video(brackish_video_folder + vid_name)
                    sort_tracklet_set = frame_skipping(vid4, SORT, brackish_bot, 1, iou_min=0.5, t_lost=3, min_hits=5)
                    sort_tracklet_set.draw_tracklets()


                target_tracklets = sort_tracklet_set
                gt_ids, pred_ids = single_vid_metrics(gt_tracklets, target_tracklets, True)[-2:]
                gt_tracklets.draw_tracklets(gt_ids)
                target_tracklets.draw_tracklets(pred_ids)
                stitched_vid = stitch_video(gt_tracklets.video, target_tracklets.video, "gt_SORT.mp4")
                stitched_vid.play(1500, start_paused=True)
                
                comp = single_vid_metrics(gt_tracklets, target_tracklets, return_components=True)  
                print_metrics(*metrics_from_components(comp))
                components += comp
                
            print("Overall metrics")
            print_metrics(*metrics_from_components(components))