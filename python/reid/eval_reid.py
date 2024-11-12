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
import seaborn as sns

try:
    from reid import create_reid_model
    from reid_data_utils import extract_bbox_image, resize_with_aspect_ratio, ReIDRandomTripletDataset, QueryGalleryDataset, AllTripletsInVideo
except:
    from reid.reid import create_reid_model
    from reid.reid_data_utils import extract_bbox_image, resize_with_aspect_ratio, ReIDRandomTripletDataset, QueryGalleryDataset, AllTripletsInVideo


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


def plot_qg_row(images, ranking, scores, ids, axes):
    axes[0].imshow(images[0].permute(1, 2, 0))
    axes[0].axis("off")
    axes[0].set_title("Query")

    for i, im_index in enumerate(ranking):
        label = "P" if im_index == 0 else "N"
        id = ids[im_index + 1]
        axes[i+1].imshow(images[im_index + 1].permute(1, 2, 0))
        axes[i+1].axis("off")
        axes[i+1].set_title(f"{label} - {scores[im_index]:.3f}")


def query_gallery(reid_model, dataset, k, show_plots=True, stack_plots=False, show_wrong=False, exclude_ties=False):
    hits = 0

    if not show_plots:
        iterator = tqdm(range(len(dataset)), total=len(dataset), bar_format="{l_bar}{bar:20}{r_bar}")
    else:
        iterator = range(len(dataset))

    row_images = []
    row_rankings = []
    row_scores = []
    row_ids = []

    for i in iterator:
        images, ids = dataset[i]
        embeddings = reid_model.extract_feature(images, is_batch=True)
        query_vec = embeddings[0]
        scores = np.array([reid_model.vector_similarity(query_vec, emb) for emb in embeddings[1:]])
        ranking = np.argsort(scores)[::-1]

        in_topk = 0 in ranking[:k]

        if exclude_ties:
            tie = np.any(scores[[ranking[i] for i in ranking if i != 0]] + 0.01 >= scores[ranking[0]])
        else:
            tie = False

        if in_topk and not tie: hits += 1

        if (show_plots and not stack_plots) or (show_wrong and (not in_topk or tie)):
            fig, axes = plt.subplots(1, images.shape[0], figsize=(16, 4))
            plot_qg_row(images, ranking, scores, ids, axes)
            plt.tight_layout()
            plt.show()

        if stack_plots:
            row_images.append(images)
            row_rankings.append(ranking)
            row_scores.append(scores)
            row_ids.append(ids)

            if i%stack_plots == 0 and i > 0:
                fig, axes = plt.subplots(stack_plots, len(row_images[0]), figsize=(16, 9))
                for j in range(stack_plots):
                    plot_qg_row(row_images[j], row_rankings[j], row_scores[j], row_ids[j], axes[j])
                plt.tight_layout()
                plt.show()
                row_images = []
                row_rankings = []
                row_scores = []
                row_ids = []
        
    print(f"Top {k} accuracy: {hits / len(dataset):.3f}")     


def all_triplets_histogram(reid_model, dataset):
    positives = []
    negatives = []

    for i in tqdm(range(len(dataset)), desc = f"Computing similarities", bar_format = "{l_bar}{bar:20}{r_bar}"):
        global_id, positive_local_ids, negative_ids = dataset[i]
        
        positve_images = torch.stack([dataset.get_img(global_id, id) for id in positive_local_ids])
        positive_embeddings = reid_model.extract_feature(positve_images, is_batch=True)

        pos_sim_mat = reid_model.batch_vector_similarity(positive_embeddings, positive_embeddings)
        pairs = torch.triu(pos_sim_mat, diagonal=1).nonzero()
        positives += [pos_sim_mat[i, j].item() for i, j in pairs]

        for neg_global_id, neg_local_ids in negative_ids:
            neg_img = dataset.get_img(neg_global_id, neg_local_ids)
            negative_embedding = reid_model.extract_feature(neg_img, is_batch=False)

            neg_sim_mat = reid_model.batch_vector_similarity(positive_embeddings, negative_embedding)
            negatives += [neg_sim_mat[i, 0].item() for i in range(neg_sim_mat.shape[0])]

    plt.figure(figsize=(8, 6))

    bin_width = 0.01
    bins = np.arange(start=-1, stop=1 + bin_width, step=bin_width)
    plt.hist(positives, bins=bins, color='skyblue', label='positive', density=True, alpha=0.7)
    plt.hist(negatives, bins=bins, color='coral', label='negative', density=True, alpha=0.7)
 
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    for name in os.listdir("runs"):
        random.seed(42)

        
        #load model
        print(name)
        model = create_reid_model(name, 90)
        resize_and_pad = functools.partial(resize_with_aspect_ratio, target_size=(224, 224))

        #dataset = ReIDRandomTripletDataset("C:/Users/kraw084/OneDrive - The University of Auckland/Desktop/reid_dataset_val", 
        #                                   min_track_length=50, 
        #                                   transform=resize_and_pad)

        #display_scores(model, dataset)
        #pr_curve(model, dataset)

        dataset = QueryGalleryDataset("C:/Users/kraw084/OneDrive - The University of Auckland/Desktop/reid_dataset_train",
                                        min_track_length=30, 
                                        num_negatives=9,
                                        transform=resize_and_pad,
                                        same_video=True)


        query_gallery(model, dataset, 1, show_plots=False, stack_plots=False, show_wrong=False)
        print()
        

        #dataset = AllTripletsInVideo("C:/Users/kraw084/OneDrive - The University of Auckland/Desktop/reid_dataset_val_2", 
        #                            min_track_length=50, 
        #                            transform=resize_and_pad)

        #all_triplets_histogram(model, dataset)
