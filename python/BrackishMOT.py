#Malte Pedersen, Joakim Bruslund Haurum, Daniel Lehotsky, and Ivan Nikolov. (2023). BrackishMOT [Data set]. Kaggle.

import os
import cv2
import configparser
from Video_utils import frames_to_videos


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
            



create_bMOT_videos()