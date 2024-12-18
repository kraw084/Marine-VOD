import os
os.environ["TQDM_DISABLE"] = "True"

from mv_utils import Detectors, VOD_utils, Eval_utils, Config
from vod_methods import sort, bot_sort, byte_track, oc_sort
from datasets.urchin_videos import urchin_gt_generator, only_matched_tracklets

if __name__ == "__main__":
    urchin_bot =  Detectors.create_urchin_model(Config.Config.cuda)

    display = False
    show_correct = False
    
    t_vals = [1, 8, 30]
    mh_vals = [1, 5, 10]
    p_vals = [0, 3, 15]
    iou_vals = [0.3, 0.5, 0.7]

    values = [(t, mh, p, iou) for t in t_vals for mh in mh_vals for p in p_vals for iou in iou_vals]

    names = ["SORT", "SORTwithLerp",  "BoT-SORT", "ByteTrack", "OC-SORT"]
    methods = [sort.SORT, sort.SORT, bot_sort.BoT_SORT, byte_track.ByteTrack, oc_sort.OC_SORT]

    if True:
        for n,m in zip(names, methods):
            for v in values:
                name = f"{n}_t{v[0]}_mh{v[1]}_p{v[2]}_iou{v[3]}"
                print(name)

                for i, (vid1, gt) in enumerate(urchin_gt_generator("val")):         
                    if n == "SORTwithLerp":
                        target_tracklets = m(urchin_bot, vid1, iou_min=v[3], t_lost=v[0], probation_timer=v[2], min_hits=v[1], no_save=True, silence=True, kf_est_for_unmatched=False)
                    else:
                        target_tracklets = m(urchin_bot, vid1, iou_min=v[3], t_lost=v[0], probation_timer=v[2], min_hits=v[1], no_save=True, silence=True)
                    
                    if n != "SORT":
                        VOD_utils.interpoalte_tracklet_set(target_tracklets)

                    only_matched_tracklets(target_tracklets, gt, silence=True)

                    Eval_utils.save_track_result(target_tracklets, vid1.name, name, "UrchinsNZ-val", "val", silence=True)

