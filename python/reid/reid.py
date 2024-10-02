import os
import math
import itertools
import random

import cv2
import torch
import torchvision
import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

from reid.reid_data_utils import resize_with_aspect_ratio, ReID_dataset


def load_resnet():
    return torch.hub.load("pytorch/vision:v0.10.0", "resnet50", num_classes=128, weights=None)


def save_model(model, path, name):
    """Save model weights to path/name"""
    torch.save(model.state_dict(), f"{path}/{name}")


def load_model(model, path, name):
    """Load model weights from path/name"""
    model.load_state_dict(torch.load(f"{path}/{name}", weights_only=True))


def cosine_distance(x1, x2):
    return 1 - torch.nn.functional.cosine_similarity(x1, x2, dim = -1)


class ReIDModel:
    def __init__(self, model, size=(224, 224), padding=0.3):
        self.model = model
        self.model.cuda()
        self.model.eval()

        self.size = size
        self.padding = padding

    def format_img(self, img):
        img = torch.tensor(img).float() / 255
        img = img.permute(2, 0, 1)
        return resize_with_aspect_ratio(img, self.size)

    def extract_feature(self, img, is_batch=True, numpy=False):
        if not is_batch: img = img.unsqueeze(0)
        img_vec = self.model(img.cuda())

        if numpy:
            return img_vec.cpu().detach().numpy()
        else:
            return img_vec
        
    def vector_similarity(self, vec1, vec2):
        return torch.nn.functional.cosine_similarity(vec1, vec2, dim=-1).item()
    
    def batch_vector_similarity(self, vecs1, vecs2, numpy=False):
        vecs1 = torch.nn.functional.normalize(vecs1, p=2, dim=1)
        vecs2 = torch.nn.functional.normalize(vecs2, p=2, dim=1)

        cosine_sim_matrix = torch.matmul(vecs1, vecs2.T)

        if numpy:
            return cosine_sim_matrix.cpu().detach().numpy()
        else:
            return cosine_sim_matrix
    

def create_reid_model():
    model = load_resnet()
    load_model(model, "runs/siamese_triplet_resnet_lowMargin", "models/Epoch_99.pt")
    return ReIDModel(model, (224, 224), 0.3)
    

def plot_embeddings(reid_model, dataset, tb, step, embedding_size=128):
    """Plot embeddings in tensorboard"""
    embeddings = np.zeros((0, embedding_size))
    labels = np.zeros((0, 1))

    for global_id in tqdm(dataset.global_ids, desc ="Calculating embeddings", bar_format = "{l_bar}{bar:20}{r_bar}"):
        local_ids = os.listdir(dataset.dir + f"/{global_id}")

        for i, local_id in enumerate(local_ids):
            image = torchvision.io.read_image(dataset.dir + f"/{global_id}/{local_id}").float() / 255
            image = resize_with_aspect_ratio(image, reid_model.size)
            local_embedding = reid_model.extract_feature(image, is_batch=False, numpy=True)

            embeddings = np.concatenate((embeddings, local_embedding), axis=0)
            labels = np.concatenate((labels, np.array([[global_id]])), axis=0)

            if i >= 10: break
    
    tb.add_embedding(embeddings, metadata=labels, global_step=step, tag="Feature Embedding")


def extract_bbox_image(box, image, padding=0.3):
    """Takes a box [x, y, w, h] and extracts the image"""
    x_center, y_center, w, h = box[0], box[1], box[2], box[3]
    w = math.ceil(w * (1 + padding))
    h = math.ceil(h * (1 + padding))

    top, bottom = max(0, int(y_center - h / 2)), min(image.shape[0], int(y_center + h / 2))
    left, right = max(0, int(x_center - w / 2)), min(image.shape[1], int(x_center + w / 2))

    extracted_img = image[top:bottom, left:right]
    return extracted_img


def view_similarity(target_box, frame, reid_model, sub_sample_factor=60):
    urchin_image = extract_bbox_image(target_box, frame, padding=reid_model.padding)
    urchin_vector = reid_model.extract_feature(reid_model.format_img(urchin_image), is_batch=False)
    window_h, window_w = urchin_image.shape[:2]

    step = sub_sample_factor
    frame_sample_w = round(frame.shape[1] / step)
    frame_sample_h = round(frame.shape[0] / step)
    sim_mat = np.zeros((frame_sample_h, frame_sample_w))


    for i, j in tqdm(itertools.product(range(0, frame.shape[0], step), range(0, frame.shape[1], step)), 
                     desc="Calculating similarity", bar_format="{l_bar}{bar:20}{r_bar}", 
                     total=sim_mat.shape[0]*sim_mat.shape[1]):
        
        window = extract_bbox_image([j, i, window_w, window_h], frame, padding=0)
        frame_vector = reid_model.extract_feature(reid_model.format_img(window), is_batch=False)
        sim_mat[i//step, j//step] = reid_model.vector_similarity(urchin_vector, frame_vector)

    sim_mat = cv2.resize(sim_mat, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_CUBIC)

    fig, axes = plt.subplots(1, 2)
    
    top_left = (int(target_box[0]) - int(target_box[2])//2, int(target_box[1]) - int(target_box[3])//2)
    bottom_right = (top_left[0] + int(target_box[2]), top_left[1] + int(target_box[3]))
    cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 15)

    axes[0].imshow(frame)
    axes[0].set_title("Frame")
    axes[0].axis("off")

    axes[1].imshow(frame)
    heatmap = axes[1].imshow(sim_mat, alpha=0.7, cmap="plasma")
    axes[1].set_title("Similarity")
    axes[1].axis("off")
    fig.colorbar(heatmap, ax=axes[1], fraction=0.05)
    
    plt.show()


def random_view_similarity(video, detector, reid_model):
    matplotlib.use('TkAgg')
    frame = random.choice(video.frames)
    detections = detector.xywhcl(frame)

    if len(detections) > 0:
        target_box = random.choice(detections)
        view_similarity(target_box, frame, reid_model)



if __name__ == "__main__":
    reid_model = create_reid_model()
    dataset = ReID_dataset("C:/Users/kraw084/OneDrive - The University of Auckland/Desktop/reid_dataset_v2")
    
    #from torch.utils.tensorboard import SummaryWriter
    #writer = SummaryWriter("runs/temp")
    #plot_embeddings(reid_model, dataset, writer, 0)

