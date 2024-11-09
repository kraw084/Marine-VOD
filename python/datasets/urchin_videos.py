import ast
import os
import sys
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

from mv_utils import Video_utils, Config
from mv_utils.VOD_utils import Tracklet, TrackletSet, round_box


desktop = "C:/Users/kraw084/OneDrive - The University of Auckland/Desktop/"
squidle_annotations = pd.read_csv(f"{desktop}squidle_vid_annots_V4.csv")

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


def urchin_gt_tracklet(vid_name, vid, format="trimmed"):
    f = open(f"{folder}/{format}/{vid_name}.txt")
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


def trim_vid_to_tracklets(vid, tracklets):
    min_frame = vid.num_of_frames
    max_frame = 0
    for tracklet in tracklets:
        min_frame = min(min_frame, tracklet.start_frame)
        max_frame = max(max_frame, tracklet.end_frame)

    vid.set_frames(vid.frames[min_frame:max_frame+1], vid.fps)

    return min_frame, max_frame


def offset_tracklets(tracklets, offset):
    for tracklet in tracklets:
        tracklet.start_frame += offset
        tracklet.end_frame += offset
        tracklet.frame_indexes = [x + offset for x in tracklet.frame_indexes]


def split_vid_and_tracklets(vid, tracklet_set, points):
    new_tracklet_sets = []
    new_vids = []

    points = [0, *points, vid.num_of_frames]

    for i in range(len(points) - 1):
        start = points[i]
        end = points[i+1]
        tracklets = [t for t in tracklet_set if t.start_frame >= start and t.end_frame <= end]

        new_vid = Video_utils.Video(vid.name + f"_part_{i}" + vid.file_type, True)
        new_vid.set_frames(vid.frames, vid.fps)
        #new_min, new_max = trim_vid_to_tracklets(new_vid, tracklets)
        #offset_tracklets(tracklets, -new_min)
        new_vids.append(new_vid)
        new_tracklet_sets.append(TrackletSet(new_vid, tracklets, tracklet_set.num_to_label))


    return new_vids, new_tracklet_sets


def save_trimmed_tracklets():
    urchin_video_folder = Config.Config.urchin_vid_path
    for vid_name in tqdm(os.listdir(f"{folder}/lerped")):
        name = vid_name.split(".")[0]
        vid = Video_utils.Video(urchin_video_folder + "/" + name + ".mp4")
        gt = urchin_gt_tracklet(name, vid, format="lerped")


        if name == "DSC_6238": 
            vids, tss = split_vid_and_tracklets(vid, gt, [190, 550])
        else:
            vids = [vid]
            tss = [gt]


        for i in range(len(vids)):
            vid = vids[i]
            gt = tss[i]

            ids = [t.id for t in gt.tracklets]
            min_frame, max_frame = trim_vid_to_tracklets(vid, gt.tracklets)

            with open(f"{folder}/lerped/{name}.txt", "r") as f:
                lines = f.readlines()
                lines = [line.strip().split(",") for line in lines]
                lines = [[int(x) for x in line] for line in lines]

                lines = [line for line in lines if line[1] in ids]
                for j in range(len(lines)):
                    lines[j][0] -= min_frame

            with open(f"{folder}/trimmed/{name if len(vids) == 1 else name + f'_part_{i}'}.txt", "w") as f:
                for line in lines:
                    f.write(",".join([str(x) for x in line]) + "\n")


            #vid.save(f"{folder}/{name if len(vids) == 1 else name + f'_part_{i}'}.mp4")


def read_txt(path):
    with open(path, "r") as f:
        lines = f.readlines()
        lines = [line.strip().split(",") for line in lines]
        lines = [[int(x) for x in line] for line in lines]

        return np.array(lines)


def data_summary(path=f"{folder}/trimmed"):
    total_frames = 0
    total_urchins = 0
    total_counts = 0

    if path.endswith(".txt"):
        with open(path, "r") as f:
            txts = f.readlines()
            txts = [txt.strip() for txt in txts]
        path = f"{folder}/trimmed"
    else:
        txts = os.listdir(path)

    for txt in txts:
        data = read_txt(f"{path}/{txt}")

        frames = len(np.unique(data[:, 0]))
        urchins = len(np.unique(data[:, 1]))
        counts = np.bincount(data[:, 7])

        if counts.shape[0] != 2:
            counts = np.array([counts[0], 0])

        total_frames += frames
        total_urchins += urchins
        total_counts += counts

        print(txt)
        print(f"Num of frames: {frames}")
        print(f"Num of tracklets: {urchins}")
        print(f"Class counts: {counts}")
        print("-----------------------------")

    print(f"Total num of frames: {total_frames}")
    print(f"Total num of tracklets: {total_urchins}")
    print(f"Total class counts: {total_counts}")


def val_test_splits():
    names = os.listdir(f"{folder}/trimmed")
    random.shuffle(names)

    with open(f"{folder}/val.txt", "w") as f:
        for name in names[:len(names)//2]:
            f.write(name + "\n")

    with open(f"{folder}/test.txt", "w") as f:
        for name in names[len(names)//2:]:
            f.write(name + "\n")


def urchin_gt_generator(val_or_test = "val"):
    with open(f"{folder}/{val_or_test}.txt", "r") as f:
        txts = f.readlines()
        names = [txt.strip().split(".")[0] for txt in txts]

    for name in names:
        vid = Video_utils.Video(f"{Config.Config.urchin_vid_trimmed_path}/{name}.mp4")
        tracklet_set = urchin_gt_tracklet(name, vid)

        yield vid, tracklet_set


def only_matched_tracklets(tracklet_set, gt_tracklet_set):
    

#frame 0
#id 1
#top left x 2
#top left y 3
#width 4
#height 5
#confidene 6
#class 7
#vis 8

if __name__ == "__main__":
    format_annotations()
    format_txts()

    save_trimmed_tracklets()

    #data_summary()
    #val_test_splits()

    #data_summary(f"{folder}/val.txt")
    #print()
    #data_summary(f"{folder}/test.txt")

    if False:
        urchin_video_folder = Config.Config.urchin_vid_path
        for vid_name in os.listdir(f"{folder}/lerped"):
            name = vid_name.split(".")[0]
            vid = Video_utils.Video(urchin_video_folder + "/" + name + ".mp4")
            gt = urchin_gt_tracklet(name, vid)

            min_frame, max_frame = trim_vid_to_tracklets(vid, gt.tracklets)
            print(min_frame, max_frame)
            offset_tracklets(gt.tracklets, -min_frame)

            gt.draw_tracklets()
            vid.play(1200, start_paused = True)

    if False:
        urchin_video_folder = Config.Config.urchin_vid_trimmed_path
        for vid_name in os.listdir(f"{folder}/trimmed"):
            name = vid_name.split(".")[0]
            vid = Video_utils.Video(urchin_video_folder + "/" + name + ".mp4")
            gt = urchin_gt_tracklet(name, vid)

            gt.draw_tracklets()
            vid.play(1200, start_paused = True)