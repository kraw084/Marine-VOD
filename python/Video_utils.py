import cv2

def frames_to_video(frames, video_file_path, fps, size):
    """Saves a list of frames as a video"""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(video_file_path, fourcc, fps, size)

    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video.write(frame)

    video.release()


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


class Video:
    def __init__(self, video_file_path):
        self.name = video_file_path
        self.frames = []

        cap = cv2.VideoCapture(video_file_path)
        self.size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) #(w, h) of the image
        self.fps = cap.get(cv2.CAP_PROP_FPS) #frame rate of the video
        self.fourcc = int(cap.get(cv2.CAP_PROP_FOURCC)) #Fourcc video format code

        while cap.isOpened():
            success, frame = cap.read()
            if not success: break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.frames.append(frame)

        cap.release()

        self.num_of_frames = len(self.frames)

    def play(self, resize=1080, fps=None, start_frame=None, end_frame=None):
        if fps is None: fps = self.fps
        delay = round(1000/fps) if fps != 0 else 0

        for i, frame in enumerate(self.frames):
            if not start_frame is None and i < start_frame: continue
            if not end_frame is None and i > end_frame: break

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = resize_image(frame, new_width=resize) if frame.shape[1] > frame.shape[0] else resize_image(frame, new_height=resize)
            cv2.imshow(self.name, frame)
            cv2.waitKey(delay)

        cv2.destroyAllWindows()

    def save(self, video_file_path):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video = cv2.VideoWriter(video_file_path, fourcc, self.fps, self.size)

        for frame in self:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video.write(frame)

        video.release()

    def __repr__(self):
        return f"Video({self.name})"
    
    def __iter__(self):
        self.i = 0
        return self
    
    def __next__(self):
        if self.i >= self.num_of_frames: raise StopIteration
        frame = self.frames[self.i]
        self.i += 1
        return frame