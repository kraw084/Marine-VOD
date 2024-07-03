import os

from python.utils.VOD_utils import annotate_image, draw_data, Tracklet, TrackletSet, save_VOD


def frame_by_frame_VOD(model, video, no_save=False):
    """Perfoms VOD by running the detector on each frame independently"""
    frame_predictions = [model.xywhcl(frame) for frame in video]
    print("Finished predicting")

    total = 0
    for frame, frame_pred in zip(video, frame_predictions):
        count = len(frame_pred)
        total += count

        annotate_image(frame, frame_pred, model.num_to_class, model.num_to_colour)
        draw_data(frame, {"Objects":count, "Total":total})

    print("Finished drawing")
    if not no_save:
        print("Saving result . . .")
        count = len([name for name in os.listdir("results") if name[:name.rfind("_")] == f"{video.name}_fbf"])
        video.save(f"results/{video.name}_fbf_{count}.mp4")


def frame_by_frame_VOD_with_tracklets(model, video, no_save=False):
    """Same as fbf_VOD but returns results as a TrackletSet rather than directly drawing on the video"""
    frame_predictions = [model.xywhcl(frame) for frame in video]
    print("Finished predicting")

    tracklets = []
    id_counter = 0
    for i, frame_pred in enumerate(frame_predictions):
        for box in frame_pred:
            new_tracklet = Tracklet(id_counter)
            new_tracklet.add_box(box, i)
            tracklets.append(new_tracklet)
            id_counter += 1

    ts = TrackletSet(video, tracklets, model.num_to_class)    

    if not no_save: save_VOD(ts, "fbf")
    return ts
