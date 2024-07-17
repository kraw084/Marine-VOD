import numpy as np
import cv2
import os
from tqdm import tqdm

from mv_utils import Detectors, Video_utils, VOD_utils, Eval_utils, Cmc, Config, Plotting
from datasets import MOT17, BrackishMOT
from vod_methods import fbf, SeqNMS, sort, bot_sort, byte_track, oc_sort

#Eval_utils.track_eval(tracker_name="SORT", sub_name="Exp2", dataset_name="MOT17-half", split='val')


""""
urchin_bot =  Detectors.create_urchin_model(Config.Config.cuda)
urchin_vid_name = os.listdir(Config.Config.urchin_vid_path)[1]
vid = Video_utils.Video(Config.Config.urchin_vid_path + "/" + urchin_vid_name)

clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(32, 32))
new_frames = []
for i in tqdm(range(vid.num_of_frames)):
    frame = vid.frames[i]
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HLS)
    h, l, s =cv2.split(frame)
    #new_r, new_g, new_b = map(clahe.apply, [r, g, b])
    #new_frame = cv2.merge([new_r, new_g, new_b])
    new_frame = cv2.merge([h, clahe.apply(l), s])
    new_frame = cv2.cvtColor(new_frame, cv2.COLOR_HLS2RGB)
    new_frames.append(new_frame)

vid.set_frames(new_frames, vid.fps)

sort_tracklets = sort.SORT(urchin_bot, vid, iou_min=0.3, t_lost=8, probation_timer=3, min_hits=5, no_save=True, silence=False)

orig_vid = Video_utils.Video(Config.Config.urchin_vid_path + "/" + urchin_vid_name)
orig_sort_tracklets = sort.SORT(urchin_bot, orig_vid, iou_min=0.3, t_lost=8, probation_timer=3, min_hits=5, no_save=True, silence=False)

orig_sort_tracklets.draw_tracklets()
sort_tracklets.draw_tracklets()

stitched_video = Video_utils.stitch_video(orig_vid, vid, "orig_vs_CLAHE")
stitched_video.play(1800, start_paused=True)
"""


"""
vid_name = sorted(MOT17.vid_names_by_set("train"))[-1]

vid = MOT17.load_MOT17_video(vid_name)
MOT17_bot = Detectors.create_MOT_YOLOX_model(True)

gt_tracklets = MOT17.MOT17_gt_tracklet(vid, conf_threshold=0.0)

sort_tracklets = sort.SORT(MOT17_bot, vid, iou_min=0.3, t_lost=8, probation_timer=3, min_hits=5, no_save=True, silence=False)
bot_sort_tracklets = bot_sort.BoT_SORT(MOT17_bot, vid, iou_min=0.3, t_lost=30, probation_timer=3, min_hits=5, no_save=True, silence=False)
byte_track_tracklets = byte_track.ByteTrack(MOT17_bot, vid, iou_min=0.3, t_lost=30, probation_timer=5, min_hits=10, no_save=True, silence=False)
oc_sort_tracklets = oc_sort.OC_SORT(MOT17_bot, vid, iou_min=0.3, t_lost=30, probation_timer=3, min_hits=5, no_save=True, silence=False)


tss = [sort_tracklets, bot_sort_tracklets, byte_track_tracklets, oc_sort_tracklets]
t_names = ["SORT", "BoT-SORT", "ByteTrack", "OC-SORT"]

gt_index = 0
gt_tracklet = gt_tracklets.tracklets[gt_index]
matched_tracklets = []
for ts in tss:
    evaluator = Eval_utils.Evaluator("")
    evaluator.set_tracklets(gt_tracklets, ts)
    evaluator.eval_video()
    best_matches = evaluator.id_of_best_match()
    pred_id = best_matches[gt_index]
    matched_tracklets.append(ts.tracklets[ts.id_to_index[pred_id]])

Plotting.tracklet_trail_graph(gt_tracklet, matched_tracklets, t_names, vid)

#Plotting.mt_heatmap(vid, gt_tracklets, tss, t_names)
"""