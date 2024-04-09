import cv2

def video_to_frames(video_file_path):
    """Returns a list of image frames, size of frames, fps and fourcc code"""

    frames = []
    cap = cv2.VideoCapture(video_file_path)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) #(w, h) of the image
    fps = cap.get(cv2.CAP_PROP_FPS) #frame rate of the video
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC)) #Fourcc video format code

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    return frames, size, fps, fourcc


def frames_to_videos(frames, video_file_path, fps, fourcc, size):
    """Saves a list of frames as a video"""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(video_file_path, fourcc, fps, size)

    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video.write(frame)

    video.release()


def video_frame_by_frame(video_file_path, delay=0, delay_is_fps=False, size=640):
    """Play a video frame by frame with delay millisconds beetween frame"""
    cap = cv2.VideoCapture(video_file_path)

    if delay_is_fps: delay = round(1000 * (1/delay))       

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        frame = resize_image(frame, new_width=size) if frame.shape[1] > frame.shape[0] else resize_image(frame, new_height=size)
        cv2.imshow(f"{video_file_path}", frame)
        cv2.waitKey(delay)

    cap.release()
    cv2.destroyAllWindows()


def annotate_image(im, prediction, num_to_label, num_to_colour):
        """Draws xywhcl boxes onto a single image. Colours are BGR"""
        thickness = 2
        font_size = 1

        label_data = []
        for pred in prediction:
            top_left = (pred[0], pred[1])
            bottom_right = (top_left[0] + pred[2], top_left[1] + pred[3])
            label = num_to_label[int(pred[5])]
            label = f"{label[0]}. {label.split()[1]}"

            colour = num_to_colour[pred[5]]

            #Draw boudning box
            im = cv2.rectangle(im, top_left, bottom_right, colour, thickness)

            label_data.append((f"{label} - {pred[4]:.2f}", top_left, colour))
        
        #Draw text over boxes
        for data in label_data:
            text_size = cv2.getTextSize(data[0], cv2.FONT_HERSHEY_SIMPLEX, font_size, thickness)[0]
            text_box_top_left = (data[1][0], data[1][1] - text_size[1])
            text_box_bottom_right = (data[1][0] + text_size[0], data[1][1])
            im = cv2.rectangle(im, text_box_top_left, text_box_bottom_right, data[4], -1)

            im = cv2.putText(im, data[0], data[1][0], cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), thickness, cv2.LINE_AA)


def resize_image(im, new_width=None, new_height=None):
    """Resizes an image while maintaining aspect ratio"""
    h = im.shape[0]
    w = im.shape[1]

    if new_width is None:
        ratio = new_height/h
        new_shape = (int(ratio * w), new_height)
    else:
        ratio = new_width/w
        new_shape = (new_width, int(ratio * h))

    return cv2.resize(im, new_shape, cv2.INTER_AREA)