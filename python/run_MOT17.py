from mv_utils import Detectors, Video_utils, VOD_utils, Eval_utils, Cmc
from datasets import MOT17
from vod_methods import fbf, SeqNMS, sort, bot_sort, byte_track, oc_sort


if __name__ == "__main__":
    data_set = "train"
    half = 2
    names = sorted(MOT17.vid_names_by_set(data_set))

    enable_gt = False
    
    enable_fbf = False
    enable_seqNMS = False
    enable_SORT = False
    enable_BoTSORT = False
    enable_ByteTrack = True
    enable_OCSORT = False

    compare_to_gt = False
    overall_metrics = False

    start = 0
    end = len(names)
    count = 0
    
    MOT17_bot = Detectors.create_MOT_YOLOX_model()

    for vid_name in names:
        if count < start:
            count += 1
            continue

        if count >= end: break
        count += 1

        print(vid_name)
        
        #MOT17_bot = create_MOT_model(vid_name, half=half)

        if enable_gt:
            vid1 = MOT17.load_MOT17_video(vid_name, half)
            gt_tracklets = MOT17.MOT17_gt_tracklet(vid1, conf_threshold=0.5, half=half)

        if enable_fbf:
            vid2 = MOT17.load_MOT17_video(vid_name, half)
            fbf_tracklets = fbf.frame_by_frame_VOD_with_tracklets(MOT17_bot, vid2, True)
            target_tracklets = fbf_tracklets

        if enable_seqNMS:
            vid3 = MOT17.load_MOT17_video(vid_name, half)
            seqNMS_tracklets = VOD_utils.frame_skipping(vid3, SeqNMS.Seq_nms, MOT17_bot, 1, silence=True)
            target_tracklets = seqNMS_tracklets

        if enable_SORT:
            vid4 = MOT17.load_MOT17_video(vid_name, half)
            sort_tracklets = sort.SORT(MOT17_bot, vid4, iou_min=0.3, t_lost=8, probation_timer=3, min_hits=5, no_save=True, silence=False)
            target_tracklets = sort_tracklets

        if enable_BoTSORT:
            vid5 = MOT17.load_MOT17_video(vid_name, half)
            bot_sort_tracklets = bot_sort.BoT_SORT(MOT17_bot, vid5, iou_min=0.3, t_lost=8, probation_timer=3, min_hits=5, no_save=True, silence=False)
            target_tracklets = bot_sort_tracklets
            
        if enable_ByteTrack:
            vid6 = MOT17.load_MOT17_video(vid_name, half)
            byte_track_tracklets = byte_track.ByteTrack(MOT17_bot, vid6, iou_min=0.3, t_lost=8, probation_timer=3, min_hits=5, no_save=True, silence=False)
            
            VOD_utils.interpoalte_tracklet_set(byte_track_tracklets)

            target_tracklets = byte_track_tracklets
            
        if enable_OCSORT:
            vid7 = MOT17.load_MOT17_video(vid_name, half)
            oc_sort_tracklets = oc_sort.OC_SORT(MOT17_bot, vid7, iou_min=0.3, t_lost=8, probation_timer=3, min_hits=5, no_save=True, silence=False)
            target_tracklets = oc_sort_tracklets

        if enable_gt and compare_to_gt:
            gt_ids, pred_ids = Eval_utils.correct_ids(gt_tracklets, target_tracklets)
            gt_tracklets.draw_tracklets(gt_ids)
            target_tracklets.draw_tracklets(pred_ids)

            stitched_video = Video_utils.stitch_video(gt_tracklets.video, target_tracklets.video, "gt_vs_tracking.mp4")
            stitched_video.play(1500, start_paused = True)

        elif enable_gt and overall_metrics:
            eval = Eval_utils.Evaluator("SORT", 0.5)
            eval.set_tracklets(gt_tracklets, target_tracklets)
            eval.eval_video(loading_bar=False)
            eval.print_metrics(True)
            
            #mota = [results[2] for results in eval.metrics_fbf()]
            #metric_by_frame_graph(target_tracklets.video, "MOTA", mota)
            
        else:

            Eval_utils.save_track_result(target_tracklets, vid_name, "ByteTrack", "MOT17-half-val", "Exp2")
        
            #target_tracklets.draw_tracklets()
            #target_tracklets.video.play(1080, start_paused = True)
            
            #sort_tracklets.draw_tracklets()
            #show_flow(bot_sort_tracklets.video)
            #bot_sort_tracklets.draw_tracklets()
            #stitched_video = stitch_video(sort_tracklets.video, bot_sort_tracklets.video, "sort_vs_bot.mp4")
            #stitched_video.play(1500, start_paused = True)


    if enable_gt and overall_metrics:
        print("Overall metrics:")
        eval.print_metrics()