import functools

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from train_reid import load_model, load_resnet, view_dataset, ReID_dataset, resize_with_aspect_ratio

def compute_similarity(model, im1, im2):
    im1 = im1.cuda().unsqueeze(0)
    im2 = im2.cuda().unsqueeze(0)
    im1_vec = model(im1)
    im2_vec = model(im2)

    return torch.nn.functional.cosine_similarity(im1_vec, im2_vec, dim=-1).item()

def display_scores(dataset):
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

        pos_sim = compute_similarity(model, anc, pos)
        neg_sim = compute_similarity(model, anc, neg)

        fig.suptitle(f"Positive similarity: {pos_sim:.3f}, Negative similarity: {neg_sim:.3f}")

        plt.show()


def eval(model, dataset, thresh_increments=0.01):
    pos_scores = []
    neg_scores = []

    for i in tqdm(range(len(dataset)), desc = f"Computing similarities", bar_format = "{l_bar}{bar:20}{r_bar}"):
        anc, pos, neg = dataset[i]

        pos_sim = compute_similarity(model, anc, pos)
        neg_sim = compute_similarity(model, anc, neg)

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

if __name__ == "__main__":
    #load model
    model = load_resnet()
    load_model(model, "runs/siamese_triplet_resnet_lowMargin", "models/Epoch_99.pt")
    model.cuda()
    model.eval()

    #load dataset
    resize_and_pad = functools.partial(resize_with_aspect_ratio, target_size=(256, 256))
    dataset = ReID_dataset("C:/Users/kraw084/OneDrive - The University of Auckland/Desktop/reid_dataset_v2", transform=resize_and_pad)

    #display_scores(dataset)
    eval(model, dataset)