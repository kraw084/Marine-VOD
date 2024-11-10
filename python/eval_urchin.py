
from mv_utils import Detectors, Video_utils, VOD_utils, Eval_utils, Cmc, Config
from vod_methods import fbf, SeqNMS, sort, bot_sort, byte_track, oc_sort, deep_sort
from datasets.urchin_videos import urchin_gt_generator, only_matched_tracklets

if __name__ == "__main__":
    urchin_bot =  Detectors.create_urchin_model(Config.Config.cuda)

    start = 0

    display = False


    #Eval_utils.track_eval(tracker_name="SORT",
    #                      sub_name="first_test",
    #                      dataset_name="UrchinNZ",
    #                      split="val",
    #                      iou_th=0.5)

    if True:
        for i, (vid1, gt) in enumerate(urchin_gt_generator("val")):
            if i < start:
                continue

            print(vid1.name)

            vid2 = vid1.copy()

            sort_tracklets = sort.SORT(urchin_bot, vid2, iou_min=0.3, t_lost=8, probation_timer=3, min_hits=5, no_save=True, silence=False, kf_est_for_unmatched=True)
            target_tracklets = sort_tracklets

            only_matched_tracklets(target_tracklets, gt)

            if display:
                gt.draw_tracklets()
                target_tracklets.draw_tracklets()
                stitched_video = Video_utils.stitch_video(gt.video, target_tracklets.video, "gt_vs_tracking.mp4")
                stitched_video.play(1500, start_paused = True)


            Eval_utils.save_track_result(target_tracklets, vid1.name, "SORT", "UrchinsNZ-val", "first_test")

            #evaluator = Eval_utils.Evaluator("SORT", 0.5)
            #evaluator.set_tracklets(gt, target_tracklets)
            #evaluator.eval_video(loading_bar=True)
            #evaluator.print_metrics(True)
