import os 

from mv_utils import Detectors, Video_utils, VOD_utils, Eval_utils, Cmc, Config


groups = ["SORT", "SORTwithLerp", "BoT-SORT", "ByteTrack", "OC-SORT"]
group_paths = {name: [] for name in groups}

for name in os.listdir("TrackEval_results/UrchinsNZ-val"):
    group_paths[name.split("_")[0]].append(name)

    if False:
        Eval_utils.track_eval(tracker_name=name,
                        sub_name="val",
                        dataset_name="UrchinNZ",
                        split="val",
                        iou_th=0.3)



for k in group_paths.keys():
    scores = []
    for path in group_paths[k]:
        met_file = f"TrackEval_results/UrchinsNZ-val/{path}/val/pedestrian_summary.txt"

        with open(met_file, "r") as f:
            lines = f.readlines()
            hota = float(lines[1].split(" ")[0])

            scores.append((hota, path))

    scores.sort(reverse=True)
    print(scores[0])
    print()

    #Eval_utils.track_eval(tracker_name=scores[0][1],
    #                    sub_name="val",
    #                    dataset_name="UrchinNZ",
    #                    split="val",
    #                    iou_th=0.3)
    
    pause = input()