import configparser
import os
import cv2
import numpy as np

from Video_utils import Video, frames_to_video
from VOD_utils import Tracklet, TrackletSet, annotate_image, round_box
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
    detector = PublicDetectionsDetector("MOT17-02", ["Person"], [(255, 0, 0)], conf=0.6)
    vid = Video("MOT17/videos/MOT17-02.mp4")
    for frame in vid:
        pred = detector.xywhcl(frame)
        annotate_image(frame, pred, detector.num_to_class, detector.num_to_colour)

    vid.play()


def MOT17_gt_tracklet(vid, data_set="train", conf_threshold=0.5):
    vid_name = vid.name
    set_folder = "train" if os.path.isdir(f"MOT17/train/{vid_name}-FRCNN") else "test"
    gt_file = open(f"MOT17/{set_folder}/{vid_name}-FRCNN/gt/gt.txt")
    gts = [tuple([float(num) for num in line.strip("/n").split(",")]) for line in gt_file.readlines()]
    gt_file.close()

    config = configparser.ConfigParser()
    config.read("MOT17/" + set_folder + "/" + vid_name + "-FRCNN/seqinfo.ini")
    im_shape = (int(config["Sequence"]["imHeight"]), int(config["Sequence"]["imWidth"]))

    tracklets = {}
    for frame, id, top_left_x, top_left_y, width, height, is_person, class_number, conf in gts:
        if is_person == 0: continue
        if conf < conf_threshold: continue 
        if data_set == "train" and frame > vid.num_of_frames: continue
        if data_set == "val" and frame <= vid.num_of_frames: continue

        center_x = top_left_x + width/2
        center_y = top_left_y + height/2
        box = round_box(np.array([center_x, center_y, width, height, conf, 0]))

        id = int(id)
        if id in tracklets:
            tracklets[id].add_box(box, int(frame) - 1, im_shape)
        else:
            new_tracklet = Tracklet(id)
            new_tracklet.add_box(box, int(frame) - 1, im_shape)
            tracklets[id] = new_tracklet

    return TrackletSet(vid, list(tracklets.values()), ["Person"])


def load_MOT17_video(vid_name, data_set="train"):
    vid = Video(f"MOT17/videos/{vid_name}.mp4")
    if data_set == "train":
        vid.set_frames(vid.frames[:vid.num_of_frames//2], vid.fps)
    elif data_set == "val":
        vid.set_frames(vid.frames[vid.num_of_frames//2:], vid.fps)

    return vid


def vid_names_by_set(data_set = "train"):
    if data_set == "train" or data_set == "val":
        names = list({name[:8] for name in os.listdir("MOT17/train")})
    else:
        names = list({name[:8] for name in os.listdir("MOT17/test")})

    return names


if __name__ == "__main__":
    #create_MOT17_videos()
    
    #test_mot_detector()

    #vid = Video("MOT17/videos/MOT17-02.mp4")
    #ts = MOT17_gt_tracklet(vid)
    #ts.draw_tracklets()
    #ts.video.play()

    pass