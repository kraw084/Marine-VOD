import cv2
import math

def video_to_frames(video_file_path):
    """Returns a list of image frames, size of frames, fps and fourcc code"""

    frames = []
    cap = cv2.VideoCapture(video_file_path)

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) #(w, h) of the image
    fps = cap.get(cv2.CAP_PROP_FPS) #frame rate of the video
    fourcc = cap.get(cv2.CAP_PROP_FOURCC) #Fourcc video format code

    return frames, size, fps, fourcc


def frames_to_videos(frames, video_file_path, fps, fourcc, size):
    """Saves a list of frames as a video"""
    video = cv2.VideoWriter(video_file_path, fourcc, fps, size)

    for frame in frames:
        video.write(frame)

    video.release()


def annotate_image(im, prediction, num_to_label, num_to_colour, draw_labels=True):
        label_data = []
        for pred in prediction:
            top_left = (round(pred[0]), round(pred[1]))
            bottom_right = (round(pred[2]), round(pred[3]))

            label = num_to_label[int(pred[5])]
            label = f"{label[0]}. {label.split()[1]}"

            colour = num_to_colour[pred[5]]

            font_size = max(im.shape) / 1900
            thickness = max(int(math.ceil(font_size)) - 1, 1)
            if not draw_labels: thickness = 2 * thickness

            #Draw boudning box
            im = cv2.rectangle(im, top_left, bottom_right, colour, 3 * thickness)

            label_data.append((f"{label} - {pred[4]:.2f}", top_left, font_size, thickness, colour))
        
        #Draw text over boxes
        if draw_labels:
            for data in label_data:
                text_size = cv2.getTextSize(data[0], cv2.FONT_HERSHEY_SIMPLEX, data[2], data[3])[0]
                text_box_top_left = (data[1][0] - data[3] - 1, data[1][1] - text_size[1] - data[3] - 8 * math.ceil(data[2]))
                text_box_bottom_right = (data[1][0] + text_size[0] + data[3], data[1][1])
                im = cv2.rectangle(im, text_box_top_left, text_box_bottom_right, data[4], -1)

                im = cv2.putText(im, data[0], (data[1][0], data[1][1] - 6 * math.ceil(data[2])), 
                                cv2.FONT_HERSHEY_SIMPLEX, data[2], (0, 0, 0), data[3], cv2.LINE_AA)