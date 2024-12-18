import os
os.environ["TQDM_DISABLE"] = "True"

import time

from mv_utils import Detectors, Video_utils, VOD_utils, Eval_utils, Config
from vod_methods import sort, bot_sort, byte_track, oc_sort
from datasets.urchin_videos import urchin_gt_generator, only_matched_tracklets

if __name__ == "__main__":
    urchin_bot =  Detectors.create_urchin_model(Config.Config.cuda)


    if True:
        Eval_utils.track_eval(tracker_name="OC-SORT",
                              sub_name="test-2",
                              dataset_name="UrchinNZ",
                              split="test",
                              iou_th=0.3)


    if False:
        mpe = open("mpe.txt", "a")
        speeds = open("speeds.txt", "a")
        speeds.close()
        mpe.close()
        
        names = ["SORT", "SORTwithLerp", "BoT-SORT", "ByteTrack", "OC-SORT"]
        methods = [sort.SORT, sort.SORT, bot_sort.BoT_SORT, byte_track.ByteTrack, oc_sort.OC_SORT]
        #values = [(8, 5, 3, 0.3), (30, 5, 3, 0.3), (30, 5, 3, 0.3), (30, 5, 3, 0.3), (30, 5, 3, 0.3)]
        values = [(8, 10, 0, 0.5), (30, 5, 3, 0.5), (30, 5, 0, 0.3), (30, 10, 3, 0.3), (30, 5, 3, 0.5)]

        for n,m,v in zip(names, methods, values):

            num_of_frames = 0
            duration = 0
            pe = 0
            print(n)

            for i, (vid1, gt) in enumerate(urchin_gt_generator("test")):
                
                num_of_frames += vid1.num_of_frames
                s = time.time()

                if n == "SORTwithLerp":
                    target_tracklets = m(urchin_bot, vid1, iou_min=v[3], t_lost=v[0], probation_timer=v[2], min_hits=v[1], no_save=True, silence=True, kf_est_for_unmatched=False)
                else:
                    target_tracklets = m(urchin_bot, vid1, iou_min=v[3], t_lost=v[0], probation_timer=v[2], min_hits=v[1], no_save=True, silence=True)
                    
                if n != "SORT":
                    VOD_utils.interpoalte_tracklet_set(target_tracklets)
            
                duration += time.time() - s

                VOD_utils.interpoalte_tracklet_set(target_tracklets)

                only_matched_tracklets(target_tracklets, gt, silence=True)

                Eval_utils.save_track_result(target_tracklets, vid1.name, n, "UrchinsNZ-test", "test-2", silence=True)

                pe += abs(len(target_tracklets) - len(gt))/len(gt)

                #print(vid1.name)
                #vid_copy = vid1.copy()
                #gt.video = vid_copy
                #gt.draw_tracklets()
                #target_tracklets.draw_tracklets()
                #stitched_video = Video_utils.stitch_video(gt.video, target_tracklets.video, "gt_vs_tracking.mp4")
                #stitched_video.play(1500, start_paused = True)
            
            print(f"Average speed: {num_of_frames/duration}")
            print(f"MPE: {pe/(i + 1)}")

            with open("speeds.txt", "a") as speeds, open("mpe.txt", "a") as mpe:
                speeds.write(f"{n} {num_of_frames/duration}\n")
                mpe.write(f"{n} {pe/(i + 1)}\n")

            print()

