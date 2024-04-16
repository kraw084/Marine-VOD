#Malte Pedersen, Joakim Bruslund Haurum, Daniel Lehotsky, and Ivan Nikolov. (2023). BrackishMOT [Data set]. Kaggle.

import os
import cv2
import numpy as np
import configparser
import shutil
import random
import math
import sys
from Video_utils import Video
from VOD_utils import Tracklet, TrackletSet

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_dir)
import yolov5.train

NUM_TO_LABEL = ["Jellyfish", "Fish", "Crab", "Shrimp", "Starfish", "Smallfish", ""]


def create_bMOT_videos():
    config = configparser.ConfigParser()

    for set in ("train", "test"):
        path = "BrackishMOT/" + set
        video_folders = os.listdir(path)
        for video_folder in video_folders:
            frames = []
            for im in os.listdir(path + "/" + video_folder + "/" + "img1"):
                im = cv2.imread(path + "/" + video_folder + "/" + "img1/" + im)
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                frames.append(im)

            size = (frames[0].shape[1], frames[0].shape[0])

            config.read(path + "/" + video_folder + "/seqinfo.ini")
            fps = int(config["Sequence"]["frameRate"])

            frames_to_videos(frames, "BrackishMOT/videos/" + video_folder + ".mp4", fps, None, size)
            

def brackishMOT_tracklet(video_number):
    video_folder_name = f"brackishMOT-{video_number:02}"
    set = "train" if os.path.isdir("BrackishMOT/train/" + video_folder_name) else "test"
    txt = open(f"BrackishMOT/{set}/{video_folder_name}/gt/gt.txt")

    lines = txt.readlines()
    lines = [tuple([int(num.split(".")[0]) for num in line.strip("/n").split(",")][:-1]) for line in lines]

    tracklets = {}

    for frame, id, x_topleft, y_topleft, width, height, conf, label in lines:
        #convert box to correct
        x = x_topleft + width//2
        y = y_topleft + height//2

        box = np.array([x, y, width, height, conf, label])
        if id in tracklets:
            tracklets[id].add_box(box, frame - 1)
        else:
            new_tracklet = Tracklet(id)
            new_tracklet.add_box(box, frame - 1)
            tracklets[id] = new_tracklet
    
    return list(tracklets.values())


def play_brackish_video(video_number):
    vid = Video(f"BrackishMOT/videos/brackishMOT-{video_number:02}.mp4")
    tracklets = TrackletSet(vid, brackishMOT_tracklet(video_number), NUM_TO_LABEL)

    tracklets.play_video(30)


def create_yolo_images_and_labels():
    for set in ("train", "test"):
        path = "BrackishMOT/" + set
        video_folders = os.listdir(path)
        for video_folder in video_folders:
            video_number = video_folder[-2:]

            for im in os.listdir(path + "/" + video_folder + "/" + "img1"):
                im_full_path = path + "/" + video_folder + "/img1/" + im
                shutil.copyfile(im_full_path, "BrackishMOT/images/vid" + video_number + "-" + im)
                
            txt = open(f"BrackishMOT/{set}/{video_folder}/gt/gt.txt")
            lines = txt.readlines()
            lines = [tuple([int(num.split(".")[0]) for num in line.strip("/n").split(",")][:-1]) for line in lines]

            for frame, id, x_topleft, y_topleft, width, height, conf, label in lines:
                if label > 5: continue

                x = x_topleft + width//2
                y = y_topleft + height//2

                box = f"{label}, {x}, {y}, {width}, {height}"

                label_path = f"BrackishMOT/labels/vid{video_number}-{frame:06}.txt"
                if os.path.isfile(label_path):
                    f = open(label_path, "a")
                    f.write("\n" + box)
                    f.close()
                else:
                    f = open(label_path, "w")
                    f.write(box)
                    f.close()


def train_test_split():
    train = []
    val = []
    test = []

    indices = list(range(1, 99))
    random.shuffle(indices)

    target_set = [0] * len(indices)
    cutoff__1 = math.floor(len(indices)*0.8)
    cutoff__2 = cutoff__1 + math.ceil(len(indices)*0.1)

    for i, vid_index in enumerate(indices):
        if i <= cutoff__1 and i <= cutoff__2: continue

        if i <= cutoff__2: 
            target_set[vid_index - 1] = 1
        else:
            target_set[vid_index - 1] = 2

    for im in os.listdir("BrackishMOT/images"):
        vid_number = int(im[3:5])
        set_code = target_set[vid_number - 1]

        if set_code == 0: train.append("images/" + im)
        if set_code == 1: val.append("images/" + im)
        if set_code == 2: test.append("images/" + im)

    total = len(train + val + test)
    print(len(train)/total, len(val)/total, len(test)/total)

    for data, name in zip([train, val, test], ("train", "val", "test")):
        f = open(f"BrackishMOT/{name}.txt", "w")
        f.write("\n".join(data))
        f.close()


def train_brackish_detector():
    yolov5.train.run(imgsz = 640, 
                        epochs = 200, 
                        data = "BrackishMOT/brackishMOT.yaml", 
                        weights = "yolov5s.pt", 
                        save_period = 10,
                        batch_size = -1,
                        cache = "ram",
                        patience = 50,
                        )

if __name__ == "__main__":
    #create_bMOT_videos()

    #brackishMOT_tracklet(1)
    #for i in range(10, 100):
    #    play_brackish_video(i)

    #create_yolo_images_and_labels()
    
    #train_test_split()

    train_brackish_detector()