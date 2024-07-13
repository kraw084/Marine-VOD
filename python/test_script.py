import numpy as np

from mv_utils import Detectors, Video_utils, VOD_utils, Eval_utils, Cmc, Config, Plotting
from datasets import MOT17, BrackishMOT
from vod_methods import fbf, SeqNMS, sort, bot_sort, byte_track, oc_sort


#Eval_utils.track_eval(tracker_name="SORT", sub_name="Exp2", dataset_name="MOT17-half", split='val')

vid_name = sorted(MOT17.vid_names_by_set("train"))[-1]

vid = MOT17.load_MOT17_video(vid_name)
MOT17_bot = Detectors.create_MOT_model(vid_name)

gt_tracklets = MOT17.MOT17_gt_tracklet(vid, conf_threshold=0.5)

sort_tracklets = sort.SORT(MOT17_bot, vid, iou_min=0.3, t_lost=8, probation_timer=3, min_hits=5, no_save=True, silence=False)
bot_sort_tracklets = bot_sort.BoT_SORT(MOT17_bot, vid, iou_min=0.3, t_lost=30, probation_timer=3, min_hits=5, no_save=True, silence=False)
byte_track_tracklets = byte_track.ByteTrack(MOT17_bot, vid, iou_min=0.3, t_lost=30, probation_timer=5, min_hits=10, no_save=True, silence=False)
oc_sort_tracklets = oc_sort.OC_SORT(MOT17_bot, vid, iou_min=0.3, t_lost=30, probation_timer=3, min_hits=5, no_save=True, silence=False)

tss = [oc_sort_tracklets, bot_sort_tracklets, byte_track_tracklets, sort_tracklets]
t_names = ["OC-SORT", "BoT-SORT", "ByteTrack", "SORT"]

Plotting.mt_heatmap(vid, gt_tracklets, tss, t_names)