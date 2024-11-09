import cv2
import math
import os

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
    def __init__(self, video_file_path, init_empty=False):
        self.path = os.path.dirname(video_file_path)
        self.name = os.path.basename(video_file_path).split(".")[0]
        self.file_type = video_file_path.split(".")[-1]
        self.full_name = self.name + "." + self.file_type
        self.frames = []
        self.size = (0, 0)
        self.fps = 0
        self.fourcc = None
        self.num_of_frames = 0

        if not init_empty:
            if not os.path.isfile(video_file_path): raise ValueError(f"{video_file_path} not found")

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

    def set_frames(self, frames, fps):
        """Replaces the frames and fps of a video - use after initialising an empty video"""
        self.frames = frames
        self.size = (frames[0].shape[1], frames[0].shape[0])
        self.fps = fps
        self.num_of_frames = len(frames)

    def play(self, resize=1080, fps=None, start_frame=None, end_frame=None, start_paused=False):
        """Play video with controls"""
        if fps is None: fps = self.fps
        delay = round(1000/fps) if fps != 0 else 0
        first_index = 0 if start_frame is None else start_frame
        final_index = len(self.frames) - 1 if end_frame is None else end_frame
        is_paused = start_paused
        temp_delay = delay

        i = first_index
        while True:
            frame = self.frames[i]
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = resize_image(frame, new_width=resize) if frame.shape[1] > frame.shape[0] else resize_image(frame, new_height=resize)
            cv2.imshow(self.name + "." + self.file_type, frame)
            key = cv2.waitKey(temp_delay if not is_paused else 0)

            if key == ord('q'): #close the video
                break
            elif key == ord(" "): #pause
                is_paused = not is_paused
            elif key == ord('a'): #previous frame
                i = max(first_index, i - 1)
            elif key == ord('d'): #next frame
                i = min(i + 1, final_index)
            elif key == ord('w'): #restart
                i = first_index
            elif key == ord('s'): #end
                i = final_index
            elif key == ord('z'): #reduce fps
                temp_delay *= 2
            elif key == ord('x'): #reset fps
                temp_delay = delay
            elif key == ord('c'): #increase fps
                temp_delay = math.ceil(temp_delay/2)
            elif key == ord('v'): #save video
                count = len([name for name in os.listdir("results") if name[:name.rfind("_")] == f"{self.name}"])
                self.save(f"results/{self.name}_{count}.{self.file_type}")
            else:
                i = min(i + 1, final_index)
            
        cv2.destroyAllWindows()

    def save(self, video_file_path):
        """Save the this video at video_file_path"""
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video = cv2.VideoWriter(video_file_path, fourcc, self.fps, self.size)

        for frame in self:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video.write(frame)

        video.release()
        print(f"Saved as {video_file_path}")

    def copy(self):
        new_vid = Video(self.full_name, init_empty=True)
        new_vid.set_frames([f.copy() for f in self], self.fps)

        new_vid.path = self.path
        new_vid.name = self.name
        new_vid.file_type = self.file_type
        new_vid.full_name = self.full_name

        return new_vid

    def __repr__(self):
        return f"Video({self.full_name})"
    
    def __iter__(self):
        self.i = 0
        return self
    
    def __next__(self):
        if self.i >= self.num_of_frames: raise StopIteration
        frame = self.frames[self.i]
        self.i += 1
        return frame
    
    def __len__(self):
        return self.num_of_frames
    

def stitch_video(left_vid, right_vid, stitched_vid_name=None):
    """Concatenate two videos horizontally, returns a new video"""
    if left_vid.num_of_frames != right_vid.num_of_frames:
        raise ValueError(f"Left video has {left_vid.num_of_frames} while right video has {right_vid.num_of_frames}")
    if left_vid.size != right_vid.size:
        raise ValueError(f"Left video has shape {left_vid.size} while right vid has shape {right_vid.size}")
    
    new_frames = []
    for left_frame, right_frame in zip(left_vid, right_vid):
        new_frames.append(cv2.hconcat([left_frame, right_frame]))

    if stitched_vid_name is None: stitched_vid_name = "Stitched_vid.mp4"
    new_vid = Video(stitched_vid_name, True)
    new_vid.set_frames(new_frames, left_vid.fps)
    
    return new_vid


def sample_frames(video, n):
    new_frames = []
    for i in range(0, len(video), n):
        new_frames.append(video.frames[i])

    print(f"Name: {video.full_name}")
    print(f"Original num of frames: {len(video)}")
    print(f"New num of frames: {len(new_frames)}")

    video.set_frames(new_frames, video.fps)


def save_as_frames(video, save_dir):
    target_dir = save_dir + "/" + video.name
    os.mkdir(target_dir)
    for i, frame in enumerate(video):
        cv2.imwrite(target_dir + f"/{i}.jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))