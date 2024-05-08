import torch
import os

from Detectors import create_urchin_model, create_brackish_model
from Video_utils import Video, stitch_video
from VOD_utils import (frame_by_frame_VOD, frame_by_frame_VOD_with_tracklets, 
                       TrackletSet, frame_skipping, single_vid_metrics, mutiple_vid_metrics, save_VOD)

from SeqNMS import Seq_nms
from sort import SORT
from BrackishMOT import brackishMOT_tracklet, id_by_set


if __name__ == "__main__":
    cuda = torch.cuda.is_available()

    count = 0
    if False:
        urchin_bot = create_urchin_model(cuda)
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
            #sortTracklets.draw_tracklets()
            #sortTracklets.video.play(1500, start_paused=True)
            save_VOD(sortTracklets, "SORT")

            count += 1
            if count > 5: break

    if True:
        brackish_bot = create_brackish_model(cuda)
        brackish_video_folder = "d:/Marine-VOD/BrackishMOT/videos/"

        enable_gt = False
        enable_fbf = False
        enable_seqNMS = False
        enable_SORT = True

        data_set = "val"
        ids = id_by_set(data_set)

        for id in ids:
            count+= 1
            if count <= 4:
                continue

            vid_name = f"brackishMOT-{id:02}.mp4"
            print(vid_name)

            #get ground truth tracklets
            if enable_gt:
                vid = Video(brackish_video_folder + vid_name)
                gt_tracklets = TrackletSet(vid, brackishMOT_tracklet(id), brackish_bot.num_to_class)
                gt_tracklets.draw_tracklets()

            #get fbf tracklets
            if enable_fbf:
                vid2 = Video(brackish_video_folder + vid_name)
                fbf_tracklets = frame_skipping(vid2, frame_by_frame_VOD_with_tracklets, brackish_bot, 1)
                fbf_tracklets.draw_tracklets()

            #get seqNMS tracklets
            if enable_seqNMS:
                vid3 = Video(brackish_video_folder + vid_name)
                seqNMS_tracklet_set = frame_skipping(vid3, Seq_nms, brackish_bot, 1, nms_iou=0.4, avg_conf_th=0.3, early_stopping_score_th=0.5)
                seqNMS_tracklet_set.draw_tracklets()

            #get sort tracklets
            if enable_SORT:
                vid4 = Video(brackish_video_folder + vid_name)
                sort_tracklet_set = frame_skipping(vid4, SORT, brackish_bot, 1, iou_min=0.5, t_lost=3, min_hits=5)
                sort_tracklet_set.draw_tracklets()
                sort_tracklet_set.video.play(1300, start_paused=True)

            #new_vid = stitch_video(vid2, vid4, "fbf_SORT.mp4")
            #new_vid.play(1300, start_paused=True)

           
