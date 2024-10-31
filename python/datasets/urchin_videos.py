import ast
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

from mv_utils import Video_utils, Config
from mv_utils.VOD_utils import Tracklet, TrackletSet, round_box


desktop = "C:/Users/kraw084/OneDrive - The University of Auckland/Desktop/"
squidle_annotations = pd.read_csv(f"{desktop}squidle_vid_annots.csv")

name_to_file = pd.read_csv(f"{desktop}Video_Files.csv")
name_to_file = dict(zip(name_to_file["File"], name_to_file["Video"]))

folder = "UrchinsNZ"

num_to_class = ["Evechinus chloroticus","Centrostephanus rodgersii"]
class_to_num = dict(zip(num_to_class, range(len(num_to_class))))


def process_image_name(name):
    segments = name.split("_")
    name = segments[0] + "_" + segments[1]
    frame = int(segments[-1])

    video = name_to_file[name].split(".")[0]
    return video, frame


def get_frame_size(name):
    im = cv2.imread(f"{desktop}GPS things/sampled_videos/{name}/0.jpg")
    h, w, _ = im.shape
    return w, h


def format_annotations():
    ids = squidle_annotations["object_id"].unique()
    vid_id_counts = {}

    for id in tqdm(ids):
        rows = squidle_annotations[squidle_annotations["object_id"] == id]
        
        for index, row in rows.iterrows():
            name = row["point.media.key"]
            vid_name, frame = process_image_name(name)

            if not f"{vid_name}.txt" in os.listdir(f"{folder}/raw"): 
                f = open(f"{folder}/raw/{vid_name}.txt", "w")
                vid_id_counts[vid_name] = 0
            else:
                f = open(f"{folder}/raw/{vid_name}.txt", "a")


            frame_w, frame_h = get_frame_size(vid_name)

            x_center, y_center = float(row["point.x"]), float(row["point.y"])
            unformatted_box =  ast.literal_eval(row["point.polygon"])

            width = unformatted_box[3][0] * frame_w * 2
            height = unformatted_box[3][1] * frame_h * 2

            top_left_x = x_center * frame_w - width/2
            top_left_y = y_center * frame_h - height/2

            id = vid_id_counts[vid_name]
            conf = float(row["likelihood"])
            vis = 1
            label = class_to_num[row["label.name"]]

            if conf == 1: conf = int(conf)
            f.write(f"{frame},{id},{int(top_left_x)},{int(top_left_y)},{int(width)},{int(height)},{conf},{label},{vis}\n")
            f.close()

        vid_id_counts[vid_name] += 1

    print("FINISHED")
    print(f"Unique urchins: {len(ids)}")
    print(f"Videos: {len(vid_id_counts)}")


def format_txts():
    for txt in tqdm(os.listdir(f"{folder}/raw")):   
        with open(f"{folder}/raw/{txt}", "r") as f:
            lines = f.readlines()
            lines = [line.strip().split(",") for line in lines]
            lines = [[int(x) for x in line] for line in lines] 

            ids = list(set([line[1] for line in lines]))
            
            new_boxes = np.zeros((0, len(lines[0])))
            for id in ids:
                boxes = [line for line in lines if line[1] == id]
                updated_boxes = interpolate(boxes)
                new_boxes = np.vstack([new_boxes, updated_boxes])

        np.savetxt(f"{folder}/lerped/{txt}", new_boxes, delimiter=",", fmt="%d")

    print("FINISHED")


def interpolate(boxes, factor=5):
    boxes = np.array(boxes)
    boxes = boxes[boxes[:, 0].argsort()]

    full_frames = np.arange(boxes[:, 0].min() * factor, boxes[:, 0].max() * factor + 1)
    box_frames = boxes[:, 0] * factor

    lerped_boxes = np.hstack([full_frames.reshape(-1, 1)] + 
                             [np.interp(full_frames, box_frames, boxes[:, i]).reshape(-1, 1) 
                              for i in range(1, boxes.shape[1])])
    
    lerped_boxes = np.rint(lerped_boxes).astype(int)
    lerped_boxes[:, 6] = 1

    return lerped_boxes


def urchin_gt_tracklet(vid_name, vid):
    f = open(f"{folder}/lerped/{vid_name}.txt")
    gts = f.readlines()
    gts = [line.strip().split(",") for line in gts]
    gts = [[int(x) for x in line] for line in gts] 
    f.close()

    tracklets = {}
    for frame, id, top_left_x, top_left_y, width, height, conf, class_number, vis in gts:
        center_x = top_left_x + width/2
        center_y = top_left_y + height/2
        box = round_box(np.array([center_x, center_y, width, height, conf, class_number]))

        if id in tracklets:
            tracklets[id].add_box(box, int(frame))
        else:
            new_tracklet = Tracklet(id)
            new_tracklet.add_box(box, int(frame))
            tracklets[id] = new_tracklet

    return TrackletSet(vid, list(tracklets.values()), num_to_class)


#frame
#id
#top left x
#top left y
#width
#height
#confidene
#class
#vis (1)

if __name__ == "__main__":
    format_annotations()
    format_txts()

    urchin_video_folder = Config.Config.urchin_vid_path
    for vid_name in os.listdir(f"{folder}/lerped"):
        name = vid_name.split(".")[0]
        vid = Video_utils.Video(urchin_video_folder + "/" + name + ".mp4")

        print("Getting gts")
        gt = urchin_gt_tracklet(name, vid)
        gt.draw_tracklets()

        vid.play(1200, start_paused = True)