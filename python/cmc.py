import cv2
import numpy as np
from tqdm import tqdm
import colorsys

from Video_utils import resize_image, Video


class CameraMotionCompensation:
    """Implements CMC based on approach in BoT-SORT: Robust Associations Multi-Pedestrian Tracking, 2022"""
    def __init__(self, affine=True):
        self.previous_points = None
        self.prev_frame_grey = None
        self.affine = affine

    def find_transform(self, new_frame):
        new_frame_grey = cv2.cvtColor(new_frame, cv2.COLOR_RGB2GRAY)
        new_points = cv2.goodFeaturesToTrack(new_frame_grey, maxCorners=1000, qualityLevel=0.01, 
                                             minDistance=1, blockSize=3, useHarrisDetector=False, k=0.04)
        
        #set up varaibles if this is the first time this func has been called
        if self.previous_points is None:
            self.previous_points = new_points
            self.prev_frame_grey = new_frame_grey

        #use optical flow to find points in new frame
        moved_points, status, err = cv2.calcOpticalFlowPyrLK(self.prev_frame_grey, new_frame_grey, self.previous_points, None)
       

        indicies_to_keep = [i for i in range(status.shape[0]) if status[i] == 1]
        prev_points_keep = self.previous_points[indicies_to_keep]
        moved_points_keep = moved_points[indicies_to_keep]

        #estimate the transformation
        if self.affine:
            mat, _ = cv2.estimateAffine2D(prev_points_keep, moved_points_keep, method=cv2.RANSAC)
        else:
            mat, _ = cv2.findHomography(prev_points_keep, moved_points_keep, method=cv2.RANSAC)

        self.previous_points = new_points
        self.prev_frame_grey = new_frame_grey

        return mat
    

def cmc_registration(vid_path, affine=True):
    """Usese cmc on every frame to stitch them together like a panorama"""
    vid = Video(vid_path)

    GMC = CameraMotionCompensation(affine)
    GMC.find_transform(vid.frames[0])

    #set up initial canvas
    w = vid.frames[0].shape[1]
    h = vid.frames[0].shape[0]
    canvas_size = (3 *h, 3 * w, 3)
    prev = np.zeros(canvas_size, dtype=np.uint8)
    prev[h:2*h, w:2*w] = vid.frames[0]

    prev_mat = np.eye(3)

    new_frames = []
    new_frames.append(np.copy(prev))

    for i in tqdm(range(1, vid.num_of_frames//3), bar_format="{l_bar}{bar:30}{r_bar}"):
        #find tranformation to new frame 
        mat = GMC.find_transform(vid.frames[i])

        if affine:
            mat = np.array([*mat, [0, 0, 1]])

        mat = np.linalg.inv(mat)
        mat = mat @ prev_mat
        prev_mat = np.copy(mat)
        mat[:, 2] += np.array([w, h, 0])

        #apply transformation
        if affine:
            aligned = cv2.warpAffine(vid.frames[i], mat[:2, :], canvas_size[::-1][1:])
        else:
            aligned = cv2.warpPerspective(vid.frames[i], mat, canvas_size[::-1][1:])

        #paste transformed frame onto the canvas
        new_im_mask = np.any(aligned != [0, 0, 0], axis=-1).astype(np.uint8)
        prev -= np.multiply(np.dstack([new_im_mask] * 3), prev)
        prev += aligned
        new_frames.append(np.copy(prev))
        
    #combine frames and play as a video
    new_vid = Video("test.mp4", True)
    new_vid.set_frames(new_frames, vid.fps)

    new_vid.play(1500, start_paused=True)


def draw_flow(im, start_points, end_points):
    for p1, p2 in zip(start_points, end_points):
        arrow_dir = np.array(p2) - np.array(p1)
        length = np.linalg.norm(arrow_dir)
        angle = 2*np.pi - np.arccos(arrow_dir[0]/length)
        if arrow_dir[1] > 0: angle = 2 * np.pi - angle
        angle /= 2 * np.pi

        col = colorsys.hsv_to_rgb(1 - angle, 1, 1)
        col = (round(255 * col[2]), round(255 * col[1]), round(255 * col[0]))

        cv2.arrowedLine(im, p1, p2, col, 5)

    #image = np.zeros((500, 500, 3), dtype=np.uint8)
    #center = (250, 250)
    #circle_points = [(round(150 * np.cos(theta)) + 250, round(150 * np.sin(theta)) + 250) for theta in np.linspace(0, 2*np.pi, 2**4 + 1)]

    #draw_flow(image, [center] * len(circle_points), circle_points)

    #cv2.imshow("test", image)
    #cv2.waitKey(0)