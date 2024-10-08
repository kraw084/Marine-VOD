import functools
import os
import itertools
import random

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision
import cv2
import matplotlib.pyplot as plt
import matplotlib

try:
    from reid import create_reid_model
    from reid_data_utils import extract_bbox_image, resize_with_aspect_ratio, ReIDRandomTripletDataset, QueryGalleryDataset
except:
    from reid.reid import create_reid_model
    from reid.reid_data_utils import extract_bbox_image, resize_with_aspect_ratio, ReIDRandomTripletDataset, QueryGalleryDataset


def display_scores(reid_model, dataset):
    for i in range(len(dataset)):
        anc, pos, neg = dataset[i]

        fig, axes = plt.subplots(1, 3)
        axes[0].imshow(anc.permute(1, 2, 0))
        axes[1].imshow(pos.permute(1, 2, 0))
        axes[2].imshow(neg.permute(1, 2, 0))

        axes[0].set_title("Anchor")
        axes[1].set_title("Positive")
        axes[2].set_title("Negative")

        axes[0].axis("off")
        axes[1].axis("off")
        axes[2].axis("off")

        anc_vec = reid_model.extract_feature(anc, is_batch=False)
        pos_vec = reid_model.extract_feature(pos, is_batch=False)
        neg_vec = reid_model.extract_feature(neg, is_batch=False)

        pos_sim = reid_model.vector_similarity(anc_vec, pos_vec)
        neg_sim = reid_model.vector_similarity(anc_vec, neg_vec)

        fig.suptitle(f"Positive similarity: {pos_sim:.3f}, Negative similarity: {neg_sim:.3f}")

        plt.show()


def pr_curve(reid_model, dataset, thresh_increments=0.01):
    pos_scores = []
    neg_scores = []

    for i in tqdm(range(len(dataset)), desc = f"Computing similarities", bar_format = "{l_bar}{bar:20}{r_bar}"):
        anc, pos, neg = dataset[i]

        anc_vec = reid_model.extract_feature(anc, is_batch=False)
        pos_vec = reid_model.extract_feature(pos, is_batch=False)
        neg_vec = reid_model.extract_feature(neg, is_batch=False)

        pos_sim = reid_model.vector_similarity(anc_vec, pos_vec)
        neg_sim = reid_model.vector_similarity(anc_vec, neg_vec)

        pos_scores.append(pos_sim)
        neg_scores.append(neg_sim)

    p = []
    r = []

    thresholds = np.arange(-1.0, 1.0, thresh_increments)
    for t in tqdm(thresholds, desc = f"Calculating P and R", bar_format = "{l_bar}{bar:20}{r_bar}"):
        tp = len([s for s in pos_scores if s >= t])
        fp = len([s for s in neg_scores if s >= t])
        fn = len([s for s in pos_scores if s < t])

        p.append(tp / (tp + fp))
        r.append(tp / (tp + fn))

    fig, axes = plt.subplots(1, 3)

    axes[0].plot(thresholds, p)
    axes[0].set_xlabel("Threshold")
    axes[0].set_ylabel("Precision")
    axes[0].set_title("Precision Curve")
    axes[0].set_xlim(-1.0, 1.0)
    axes[0].set_ylim(0.0, 1.0)

    axes[1].plot(thresholds, r)
    axes[1].set_xlabel("Threshold")
    axes[1].set_ylabel("Recall")
    axes[1].set_title("Recall Curve")
    axes[1].set_xlim(-1.0, 1.0)
    axes[1].set_ylim(0.0, 1.0)

    axes[2].plot(r, p)
    axes[2].set_xlabel("Recall")
    axes[2].set_ylabel("Precision")
    axes[2].set_title("PR Curve")
    axes[2].set_xlim(0.0, 1.0)
    axes[2].set_ylim(0.0, 1.0)


    
    plt.show()


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


def query_gallery(reid_model, dataset, k, show_plots=True):
    hits = 0

    if not show_plots:
        iterator = tqdm(range(len(dataset)), total=len(dataset), bar_format="{l_bar}{bar:20}{r_bar}")
    else:
        iterator = range(len(dataset))

    for i in iterator:
        images, ids = dataset[i]
        embeddings = reid_model.extract_feature(images, is_batch=True)
        query_vec = embeddings[0]
        scores = np.array([reid_model.vector_similarity(query_vec, emb) for emb in embeddings[1:]])
        ranking = np.argsort(scores)[::-1]

        if show_plots:
            fig, axes = plt.subplots(1, images.shape[0], figsize=(16, 4))

            axes[0].imshow(images[0].permute(1, 2, 0))
            axes[0].axis("off")
            axes[0].set_title("Query")

            for i, im_index in enumerate(ranking):
                label = "P" if im_index == 0 else "N"
                id = ids[im_index + 1]
                axes[i+1].imshow(images[im_index + 1].permute(1, 2, 0))
                axes[i+1].axis("off")
                axes[i+1].set_title(f"{label}({id}) - {scores[im_index]:.3f}")

            plt.tight_layout()
            plt.show()

        if 0 in ranking[:k]: hits += 1

    print(f"Top {k} accuracy: {hits / len(dataset):.3f}")     
    


if __name__ == "__main__":
    #load model
    model = create_reid_model()
    resize_and_pad = functools.partial(resize_with_aspect_ratio, target_size=(224, 224))

    #dataset = ReIDRandomTripletDataset("C:/Users/kraw084/OneDrive - The University of Auckland/Desktop/reid_dataset_val", 
    #                                   min_track_length=50, 
    #                                   transform=resize_and_pad)

    dataset = QueryGalleryDataset("C:/Users/kraw084/OneDrive - The University of Auckland/Desktop/reid_dataset_val",
                                  min_track_length=50, 
                                  num_negatives=9,
                                  transform=resize_and_pad)

    #display_scores(model, dataset)
    #pr_curve(model, dataset)

    query_gallery(model, dataset, 1, show_plots=False)