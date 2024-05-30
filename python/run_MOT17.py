import os
#os.environ["TQDM_DISABLE"] = "1"

from Detectors import create_MOT_model
from Video_utils import Video, stitch_video
from VOD_utils import (frame_by_frame_VOD, frame_by_frame_VOD_with_tracklets, 
                       TrackletSet, frame_skipping, single_vid_metrics, print_metrics, 
                       save_VOD, metrics_from_components)

from SeqNMS import Seq_nms
from sort import SORT

from MOT17 import load_MOT17_video, vid_names_by_set, MOT17_gt_tracklet

if __name__ == "__main__":
    data_set = "train"
    names = sorted(vid_names_by_set(data_set))

    enable_gt = True
    enable_fbf = False
    enable_seqNMS = False
    enable_SORT = True

    compare_to_gt = True

    for vid_name in names:
        MOT17_bot = create_MOT_model(vid_name)

        if enable_gt:
            vid1 = load_MOT17_video(vid_name, data_set)
            gt_tracklets = MOT17_gt_tracklet(vid1)

        if enable_fbf:
            vid2 = load_MOT17_video(vid_name, data_set)
            fbf_tracklets = frame_by_frame_VOD_with_tracklets(MOT17_bot, vid2, True)
            target_tracklets = fbf_tracklets

        if enable_seqNMS:
            vid3 = load_MOT17_video(vid_name, data_set)
            seqNMS_tracklets = frame_skipping(vid3, Seq_nms, MOT17_bot, 1, silence=True)
            target_tracklets = seqNMS_tracklets

        if enable_SORT:
            vid4 = load_MOT17_video(vid_name, data_set)
            sort_tracklets = SORT(MOT17_bot, vid4, iou_min=0.5, t_lost=5, min_hits=5, no_save=True, silence=False)
            target_tracklets = sort_tracklets


        if enable_gt and compare_to_gt:
            *metrics, gt_ids, pred_ids = single_vid_metrics(gt_tracklets, target_tracklets, True)

            gt_tracklets.draw_tracklets(gt_ids)
            target_tracklets.draw_tracklets(pred_ids)

            print_metrics(*metrics)

            stitched_video = stitch_video(gt_tracklets.video, target_tracklets.video, "gt_vs_tracking.mp4")
            stitched_video.play(1500, start_paused = True)
        else:
            target_tracklets.draw_tracklets()
            target_tracklets.video.play(1080, start_paused = True)