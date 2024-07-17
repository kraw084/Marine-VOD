import cv2
import numpy as np
from tqdm import tqdm
import colorsys

from .Video_utils import resize_image, Video


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
            mat, inliers = cv2.estimateAffine2D(prev_points_keep, moved_points_keep, method=cv2.RANSAC)

            #if the proportion of inliers is < 40% use no transformation
            inlier_prop = [j for i in inliers for j in i]
            inlier_prop = sum(inlier_prop)/len(inlier_prop)
            if inlier_prop <= 0.4:
                mat = np.eye(2)
                mat = np.append(mat, np.zeros((2, 1)), axis=1)
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


def draw_flow_arrows(im, start_points, end_points, thickness=2, length_scalar=1):
    """Takes a set of feature points and moved points (from flow calculations) and draws the vectors onto the image"""
    for p1, p2 in zip(start_points, end_points):
        if len(start_points[0].shape) == 2:
            p1 = p1[0]
            p2 = p2[0]

        #use the angle of the vector to determine its colour
        arrow_dir = (p2 - p1)
        length = np.linalg.norm(arrow_dir)

        if length == 0:
            angle = 0
        else:
            angle = 2*np.pi - np.arccos(arrow_dir[0]/length)
            if arrow_dir[1] > 0: angle = 2 * np.pi - angle
            angle /= 2 * np.pi

        col = colorsys.hsv_to_rgb(1 - angle, 1, 1)
        col = (round(255 * col[0]), round(255 * col[1]), round(255 * col[2]))

        if length_scalar != 1:
            p2 = p1 + arrow_dir * length_scalar

        #draw the arrow
        p1 = p1.astype(int)
        p2 = p2.astype(int)
        cv2.arrowedLine(im, (p1[0], p1[1]), (p2[0], p2[1]), col, thickness, line_type=cv2.LINE_AA)


def draw_flow_colour_indicator(im, center, length, num):
    """Draw a set of vectors in a circle to indicate what colours corrospond to what angles"""
    #Draw black rectangle to place indicator on
    cv2.rectangle(im, (round(center[0] - length * 1.1), round(center[1] - length * 1.1)), 
                      (round(center[0] + length * 1.1), round(center[1] + length * 1.1)),
                      (0, 0, 0), -1)
    
    #compute points in a circle around the center
    circle_points = [[round(length * np.cos(theta)) + center[0], round(length * np.sin(theta)) + center[1]] for theta in np.linspace(0, 2*np.pi, num)]
    
    #draw arrows
    draw_flow_arrows(im, np.array([list(center)] * num), np.array(circle_points))


def show_flow(vid):
    """Calculates flow at every frames and draws it onto the video then plays it"""
    prev_im = cv2.cvtColor(vid.frames[0], cv2.COLOR_RGB2GRAY)
    prev_points = cv2.goodFeaturesToTrack(prev_im, maxCorners=1000, qualityLevel=0.01, 
                                             minDistance=1, blockSize=3, useHarrisDetector=False, k=0.04)
    
    for i in tqdm(range(1, vid.num_of_frames)):
        #Calculate flow in new frame using previous frame
        new_im = cv2.cvtColor(vid.frames[i], cv2.COLOR_RGB2GRAY)
        moved_points, status, err = cv2.calcOpticalFlowPyrLK(prev_im, new_im, prev_points, None)
       
        indicies_to_keep = [i for i in range(status.shape[0]) if status[i] == 1]
        prev_points_keep = prev_points[indicies_to_keep]
        moved_points_keep = moved_points[indicies_to_keep]

        #draw flow vectors and angle colour indicator
        draw_flow_arrows(vid.frames[i], prev_points_keep, moved_points_keep)
        draw_flow_colour_indicator(vid.frames[i], (100, 100), 75, 33)

        #set prev varaibles for next loop
        prev_im = new_im
        prev_points = cv2.goodFeaturesToTrack(prev_im, maxCorners=1000, qualityLevel=0.01, 
                                             minDistance=1, blockSize=3, useHarrisDetector=False, k=0.04)


def show_transformation(video, matrices, rows=4, cols=5):
    w, h = video.size
    padding_size = 0.2
    w_pad = w * padding_size
    h_pad = h * padding_size

    x_values = np.linspace(w_pad, w - w_pad, cols)
    y_values = np.linspace(h_pad, h - h_pad, rows)

    x_coord, y_coord = np.meshgrid(x_values, y_values)

    points = np.vstack([x_coord.ravel(), y_coord.ravel()]).T
    
    for frame, mat in zip(video, matrices):
        apply_tr = lambda x: (mat[:2, :2] @ x) + mat[:2, 2]
        transformed_points = np.apply_along_axis(apply_tr, axis=1, arr=points)
        draw_flow_arrows(frame, points, transformed_points, thickness=16, length_scalar=5)
