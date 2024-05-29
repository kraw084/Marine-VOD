import configparser
import os
import cv2
import numpy as np

from Video_utils import Video, frames_to_video
from VOD_utils import Tracklet, TrackletSet, annotate_image
from Detectors import PublicDetectionsDetector


def create_MOT17_videos():
    """Turn the frames from the dataset into .mp4 videos"""
    config = configparser.ConfigParser()

    for set in ("train", "test"):
        path = "MOT17/" + set
        video_folders = os.listdir(path)
        video_folders = list({name[:8] for name in video_folders})

        for video_folder in video_folders:
            frames = []
            for im in sorted(os.listdir(path + "/" + video_folder + "-FRCNN/" + "img1")):
                im = cv2.imread(path + "/" + video_folder + "-FRCNN/" + "img1/" + im)
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                frames.append(im)

            size = (frames[0].shape[1], frames[0].shape[0])

            config.read(path + "/" + video_folder + "-FRCNN/seqinfo.ini")
            fps = int(config["Sequence"]["frameRate"])

            frames_to_video(frames, "MOT17/videos/" + video_folder + ".mp4", fps, size)


def test_mot_detector():
    detector = PublicDetectionsDetector("MOT17-02", ["Pedestrian"], [(255, 0, 0)], conf=0.6)
    vid = Video("MOT17/videos/MOT17-02.mp4")
    for frame in vid:
        pred = detector.xywhcl(frame)
        annotate_image(frame, pred, detector.num_to_class, detector.num_to_colour)

    vid.play()


if __name__ == "__main__":
    test_mot_detector()