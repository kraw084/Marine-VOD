#Malte Pedersen, Joakim Bruslund Haurum, Daniel Lehotsky, and Ivan Nikolov. (2023). BrackishMOT [Data set]. Kaggle.

import os
import cv2
import numpy as np
import configparser
import shutil
from VOD_utils import Tracklet, display_VOD
from Video_utils import frames_to_videos, video_to_frames


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
    frames = video_to_frames(f"BrackishMOT/videos/brackishMOT-{video_number:02}.mp4")[0]
    tracklets = brackishMOT_tracklet(video_number)

    display_VOD(frames, tracklets, NUM_TO_LABEL, 1080, 20)


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


if __name__ == "__main__":
    #create_bMOT_videos()

    #brackishMOT_tracklet(1)
    #for i in range(10, 100):
    #    play_brackish_video(i)

    create_yolo_images_and_labels()