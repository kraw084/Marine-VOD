import numpy as np

from mv_utils.Config import Config
from mv_utils.Detectors import create_brackish_model
from mv_utils.Video_utils import Video, stitch_video
from mv_utils.VOD_utils import TrackletSet, frame_skipping, save_VOD, metrics_from_components
from mv_utils.Cmc import show_flow
from datasets.BrackishMOT import brackishMOT_tracklet, id_by_set

from vod_methods.fbf import frame_by_frame_VOD, frame_by_frame_VOD_with_tracklets
from vod_methods.SeqNMS import Seq_nms
from vod_methods.sort import SORT
from vod_methods.bot_sort import BoT_SORT


if __name__ == "__main__":
    brackish_bot = create_brackish_model(Config.cuda)
    brackish_video_folder = f"{Config.drive}:/Marine-VOD/BrackishMOT/videos/"

    enable_gt = False
    enable_fbf = False
    enable_seqNMS = False
    enable_SORT = False
    enable_bot_sort = True

    data_set = "val"
    ids = id_by_set(data_set)
    print(f"{len(ids)} ids in set {data_set}")

    components = np.zeros((11,))

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
            sort_tracklet_set = SORT(brackish_bot, vid4, iou_min=0.3, t_lost=5, probation_timer=3, 
                                     min_hits=12, greedy_assoc=True, no_save=True, silence=True)
            
        #get bot sort tracklets
        if enable_bot_sort:
            vid5 = Video(brackish_video_folder + vid_name)
            bot_sort_tracklet_set = BoT_SORT(brackish_bot, vid5, iou_min=0.3, t_lost=5, probation_timer=3, 
                                     min_hits=12, greedy_assoc=True, no_save=True, silence=False)


        target_tracklets = bot_sort_tracklet_set
        target_tracklets.draw_tracklets()
        target_tracklets.video.play(1200, start_paused=True)

        #*metrics, gt_ids, pred_ids = single_vid_metrics(gt_tracklets, target_tracklets, match_iou=0.3, return_correct_ids=True)
        #print_metrics(*metrics)

        #gt_tracklets.draw_tracklets(gt_ids)
        #target_tracklets.draw_tracklets(pred_ids)

        #stitched_vid = stitch_video(gt_tracklets.video, target_tracklets.video, "gt_SORT.mp4")
        #stitched_vid.play(1500, start_paused=True)
        
        #comp = single_vid_metrics(gt_tracklets, target_tracklets, return_components=True)  
        #components += comp
        
    #print("Overall metrics:")
    #print_metrics(*metrics_from_components(components))