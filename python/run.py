import torch
import os

from Detectors import create_urchin_model, create_brackish_model
from Video_utils import Video, stitch_video
from VOD_utils import frame_by_frame_VOD, frame_by_frame_VOD_with_tracklets, TrackletSet, frame_skipping
from SeqNMS import Seq_nms
from BrackishMOT import brackishMOT_tracklet


if __name__ == "__main__":
    cuda = torch.cuda.is_available()


    count = 0
    if False:
        urchin_bot = create_urchin_model(cuda)
        urchin_video_folder = "E:/urchin video/All/"

        for vid_name in os.listdir(urchin_video_folder):
            vid = Video(urchin_video_folder + vid_name)
            print("Finished loading video")
            vid.play()

            #frame_by_frame_VOD(urchin_bot, vid)
            #vid.play(1500)

            seqNMSTracklets = Seq_nms(urchin_bot, vid)
            #seqNMSTracklets.video.play(1500, start_paused=True)
            count += 1
            if count > 5: break

    if True:
        brackish_bot = create_brackish_model(cuda)
        brackish_video_folder = "d:/Marine-VOD/BrackishMOT/videos/"
        for vid_name in os.listdir(brackish_video_folder):
            #get ground truth tracklets
            vid = Video(brackish_video_folder + vid_name)
            vid_num = int(vid.name[-2:])
            gt_tracklets = TrackletSet(vid, brackishMOT_tracklet(vid_num), brackish_bot.num_to_class)
            gt_tracklets.draw_tracklets()

            #get seqNMS tracklets
            vid2 = Video(brackish_video_folder + vid_name)
            seqNMS_tracklet_set = frame_skipping(vid2, Seq_nms, brackish_bot, 1, nms_iou=0.4, avg_conf_th=0.2, early_stopping_score_th=0.5)
            seqNMS_tracklet_set.draw_tracklets()

            combined_vid = stitch_video(vid, vid2, f"gt_seqNMS_{vid_name}")
            combined_vid.play(1600)

        
