import cv2

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