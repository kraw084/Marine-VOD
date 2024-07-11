import numpy as np

from mv_utils import Detectors, Video_utils, VOD_utils, Eval_utils, Cmc, Config
from datasets import MOT17, BrackishMOT
from vod_methods import fbf, SeqNMS, sort, bot_sort, byte_track, oc_sort


Eval_utils.track_eval(tracker_name="ByteTrack", sub_name="Exp2", dataset_name="MOT17-half", split='val')