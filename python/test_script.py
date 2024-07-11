import numpy as np

from mv_utils import Detectors, Video_utils, VOD_utils, Eval_utils, Cmc, Config
from datasets import MOT17, BrackishMOT
from vod_methods import fbf, SeqNMS, sort, bot_sort, byte_track, oc_sort


#Eval_utils.track_eval(tracker_name="ByteTrack", sub_name="Exp3", dataset_name="MOT17-half", split='val')

class test_model:
    def __init__(self) -> None:
        self.num_to_class = []

    def xywhcl(self, num):
        if type(num) != int:
            num = 0

        boxes = [
            [np.array([100, 0, 50, 50, 0.9, 1])],
            [np.array([125, 0, 50, 50, 0.9, 1])],
            [np.array([150, 0, 50, 50, 0.9, 1])],
            [np.array([175, 0, 50, 50, 0.9, 1])],
            [],
            [],
            [],
            [],
            [np.array([200, 0, 50, 50, 0.9, 1])],
            []
        ]

        return boxes[num]


vid = Video_utils.Video("test", True)
vid.set_frames([np.zeros((1000, 1000, 3)), 1, 2, 3, 4, 5, 6, 7, 8, 9], 1)

tracker = oc_sort.OC_SORT_Tracker(test_model(), vid, no_save=True, min_hits=0, probation_timer=0, iou_min=0.1, orm_t=2, t_lost=10)
ts = tracker.track()

for t in ts:
    print(t.id)
    for i, b in t:
        print(i, b)
    print("")


tracker = sort.SORT_Tracker(test_model(), vid, no_save=True, min_hits=0, probation_timer=0, iou_min=0.1, t_lost=10)
ts = tracker.track()

for t in ts:
    print(t.id)
    for i, b in t:
        print(i, b)
    print("")

