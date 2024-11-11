
from mv_utils import Detectors, Video_utils, VOD_utils, Eval_utils, Cmc, Config
from vod_methods import fbf, SeqNMS, sort, bot_sort, byte_track, oc_sort, deep_sort
from datasets.urchin_videos import urchin_gt_generator, only_matched_tracklets

if __name__ == "__main__":
    urchin_bot =  Detectors.create_urchin_model(Config.Config.cuda)

    start = 0

    display = True
    show_correct = False

    if False:
        Eval_utils.track_eval(tracker_name="OC-SORT",
                            sub_name=f"Final",
                            dataset_name="UrchinNZ",
                            split="val",
                            iou_th=0.3)

    if True:
        for i, (vid1, gt) in enumerate(urchin_gt_generator("val")):
            if i < start:
                continue

            print(vid1.name)

            vid2 = vid1.copy()

            fbf_tracklets = fbf.frame_by_frame_VOD_with_tracklets(urchin_bot, vid2, True) 
            #sort_tracklets = sort.SORT(urchin_bot, vid2, iou_min=0.3, t_lost=8, probation_timer=3, min_hits=5, no_save=True, silence=False, kf_est_for_unmatched=False)
            #bot_sort_tracklets = bot_sort.BoT_SORT(urchin_bot, vid2, iou_min=0.3, t_lost=30, probation_timer=3, min_hits=5, no_save=True, silence=False)
            #byte_track_tracklets = byte_track.ByteTrack(urchin_bot, vid2, iou_min=0.3, t_lost=30, probation_timer=3, min_hits=5, low_conf_th=0.05, no_save=True, silence=False)
            #byte_track_tracklets = byte_track.ByteTrack(urchin_bot, vid2, iou_min=0.2, t_lost=30, probation_timer=0, min_hits=1, low_conf_th=0.01, no_save=True, silence=False)
            #oc_sort_tracklets = oc_sort.OC_SORT(urchin_bot, vid2, iou_min=0.3, t_lost=30, probation_timer=3, min_hits=5, no_save=True, silence=False)
            
            target_tracklets = fbf_tracklets

            #VOD_utils.interpoalte_tracklet_set(target_tracklets)

            #only_matched_tracklets(target_tracklets, gt)


            if display:
                if show_correct:
                    gt_ids, pred_ids = Eval_utils.correct_ids(gt, target_tracklets, match_iou=0.3)
                    gt.draw_tracklets(gt_ids)
                    target_tracklets.draw_tracklets(pred_ids)
                else:
                    gt.draw_tracklets()
                    target_tracklets.draw_tracklets()

                stitched_video = Video_utils.stitch_video(gt.video, target_tracklets.video, "gt_vs_tracking.mp4")
                stitched_video.play(1900, start_paused = True)


            Eval_utils.save_track_result(target_tracklets, vid1.name, "FBF", "UrchinsNZ-val", "first")

            #evaluator = Eval_utils.Evaluator("SORT", 0.5)
            #evaluator.set_tracklets(gt, target_tracklets)
            #evaluator.eval_video(loading_bar=True)
            #evaluator.print_metrics(True)
