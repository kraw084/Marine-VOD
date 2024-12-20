import os
import random
import math
import json

import torch
import torchvision
import torchvision.transforms.v2 as v2
import matplotlib.pyplot as plt
import numpy as np


class ReIDRandomTripletDataset(torch.utils.data.Dataset):
    def __init__(self, dir, min_track_length=15, transform=None, same_video=False):
        self.dir = dir
        self.transform = transform

        self.global_ids = os.listdir(self.dir)
        self.global_ids = [int(i) for i in self.global_ids if i.isnumeric()]
        self.global_ids.sort()

        for i in range(len(self.global_ids) - 1, -1, -1):
            items = os.listdir(self.dir + f"/{self.global_ids[i]}")
            if len(items) <= min_track_length: 
                self.global_ids.pop(i)

        if same_video:
            with open(dir + "/video_data.json") as f:
                self.video_data = json.load(f)

            self.video_ids = [v["ids"] for v in self.video_data]
        else:
            self.video_ids = None


    def random_global(self, ignore=None):
        if ignore:
            items = [i for i in self.global_ids if i != ignore]
            if self.video_ids:
                target_item_id = [i for i in range(len(self.video_ids)) if ignore in self.video_ids[i]][0]
                items = [i for i in self.video_ids[target_item_id] if i != ignore and i in self.global_ids]
        else: 
            items = self.global_ids
        return random.choice(items)


    def random_local(self, global_id, ignore=None, min_dist=5):
        items = [int(i.strip(".jpg")) for i in os.listdir(self.dir + f"/{global_id}")]
        if ignore: items = [i for i in items if abs(i - ignore) > min_dist]
        return random.choice(items)
    

    def __len__(self):
        return len(self.global_ids)
    

    def num_of_images(self):
        total_size = 0
        for id in self.global_ids:
            total_size += len(os.listdir(self.dir + f"/{id}"))
        return total_size
    

    def __getitem__(self, index):
        target_global_id = self.global_ids[index]
        anchor_local_id = self.random_local(target_global_id)
        positive_local_id = self.random_local(target_global_id, ignore=anchor_local_id)
        
        negative_global_id = self.random_global(ignore=target_global_id)
        negative_local_id = self.random_local(negative_global_id)

        anchor_im = torchvision.io.read_image(self.dir + f"/{target_global_id}/{anchor_local_id}.jpg").float() / 255
        positive_im = torchvision.io.read_image(self.dir + f"/{target_global_id}/{positive_local_id}.jpg").float() / 255
        negative_im = torchvision.io.read_image(self.dir + f"/{negative_global_id}/{negative_local_id}.jpg").float() / 255

        if self.transform:
            anchor_im = self.transform(anchor_im)
            positive_im = self.transform(positive_im)
            negative_im = self.transform(negative_im)

        return anchor_im, positive_im, negative_im
    

class QueryGalleryDataset(ReIDRandomTripletDataset):
    def __init__(self, dir, min_track_length=15, num_negatives=9, transform=None, same_video=True):
        super().__init__(dir, min_track_length, transform, same_video)
        self.num_negatives = num_negatives

    def __getitem__(self, index):
        target_global_id = self.global_ids[index]
        anchor_local_id = self.random_local(target_global_id, min_dist=0)
        positive_local_id = self.random_local(target_global_id, ignore=anchor_local_id, min_dist=0)
        
        negatives = []
        for i in range(self.num_negatives):
            negative_global_id = self.random_global(ignore=target_global_id)
            negative_local_id = self.random_local(negative_global_id)
            negatives.append((negative_global_id, negative_local_id))

        query = torchvision.io.read_image(self.dir + f"/{target_global_id}/{anchor_local_id}.jpg").float() / 255
        positive = torchvision.io.read_image(self.dir + f"/{target_global_id}/{positive_local_id}.jpg").float() / 255

        if self.transform:
            query = self.transform(query)
            positive = self.transform(positive)

        images = torch.concatenate((query.unsqueeze(0), positive.unsqueeze(0)))
        for i in range(self.num_negatives):
            negative = torchvision.io.read_image(self.dir + f"/{negatives[i][0]}/{negatives[i][1]}.jpg").float() / 255
            if self.transform: negative = self.transform(negative)
            images = torch.concatenate((images, negative.unsqueeze(0)))

        return images, [target_global_id, target_global_id, *[i[0] for i in negatives]]


class ReIDMiningDataset(ReIDRandomTripletDataset):
    def __init__(self, dir, items_per_id=40, min_track_length=40, transform=None):
        super().__init__(dir, min_track_length, transform)
        self.items_per_id = items_per_id
        assert self.items_per_id <= min_track_length, "items_per_id must be less than min_track_length"

   
    def load_and_transform(self, global_id, local_id):
        image = torchvision.io.read_image(self.dir + f"/{global_id}/{local_id}.jpg").float() / 255

        if self.transform:
            image = self.transform(image)

        return image

    def __getitem__(self, index):
        target_global_id = self.global_ids[index]
        anchor_local_ids = np.array([int(i.strip(".jpg")) for i in os.listdir(self.dir + f"/{target_global_id}")])
        sample = np.random.choice(anchor_local_ids, self.items_per_id, replace=False)

        images = torch.stack([self.load_and_transform(target_global_id, i) for i in sample])
        return images, torch.tensor([target_global_id] * self.items_per_id)


class AllTripletsInVideo(ReIDRandomTripletDataset):
    def __init__(self, dir, min_track_length=15, limit=20, transform=None):
        super().__init__(dir, min_track_length, transform, same_video=True)
        self.limit = limit


    def all_local_ids(self, id):
        return [(id, int(i.strip(".jpg"))) for i in os.listdir(self.dir + f"/{id}")]
    

    def get_img(self, global_id, local_id):
        img = torchvision.io.read_image(self.dir + f"/{global_id}/{local_id}.jpg").float() / 255

        if self.transform:
            img = self.transform(img)

        return img

    def __getitem__(self, index):
        target_global_id = self.global_ids[index]
        all_positive_local_ids = [int(i.strip(".jpg")) for i in os.listdir(self.dir + f"/{target_global_id}")]
        all_positive_local_ids = random.sample(all_positive_local_ids, k=self.limit) if (not self.limit is None) and (len(all_positive_local_ids) > self.limit) else all_positive_local_ids

        video_id = [i for i in range(len(self.video_ids)) if target_global_id in self.video_ids[i]][0]

        all_negative_global_ids = [i for i in self.global_ids if i in self.video_ids[video_id] and i != target_global_id]
        all_negative_local_ids = []
        for i in all_negative_global_ids:
            all_negative_local_ids += self.all_local_ids(i)
        all_negative_local_ids = random.sample(all_negative_local_ids, k=self.limit) if (not self.limit is None) and (len(all_negative_local_ids) > self.limit) else all_negative_local_ids

        return target_global_id, all_positive_local_ids, all_negative_local_ids


def get_resized_size(image_size, target_size):
    aspect_ratio = image_size[0] / image_size[1]
    if aspect_ratio > 1:
        new_size = (target_size[0], int(target_size[0] / aspect_ratio))
    else:
        new_size = (int(target_size[1] * aspect_ratio), target_size[1])
    return new_size


def calculate_padding(resized_size, target_size):
    pad_left = (target_size[1] - resized_size[1]) // 2
    pad_right = target_size[1] - resized_size[1] - pad_left
    pad_top = (target_size[0] - resized_size[0]) // 2
    pad_bottom = target_size[0] - resized_size[0] - pad_top
    return (pad_left, pad_top, pad_right, pad_bottom)


def resize_with_aspect_ratio(image, target_size):
    new_size = get_resized_size(image.shape[1:], target_size)
    resize = v2.Resize(new_size)

    padding = calculate_padding(new_size, target_size)
    pad = v2.Pad(padding, fill=0.5)

    return pad(resize(image))


def view_dataset(dataset):
    for i in range(len(dataset)):
        anc, pos, neg = dataset[i]
        print(anc.shape, pos.shape, neg.shape)

        fig, axes = plt.subplots(1, 3)
        axes[0].imshow(anc.permute(1, 2, 0))
        axes[1].imshow(pos.permute(1, 2, 0))
        axes[2].imshow(neg.permute(1, 2, 0))
        plt.show()


def extract_bbox_image(box, image, padding=0.3):
    """Takes a box [x, y, w, h] and extracts the image"""
    x_center, y_center, w, h = box[0], box[1], box[2], box[3]
    w = math.ceil(w * (1 + padding))
    h = math.ceil(h * (1 + padding))

    top, bottom = max(0, int(y_center - h / 2)), min(image.shape[0], int(y_center + h / 2))
    left, right = max(0, int(x_center - w / 2)), min(image.shape[1], int(x_center + w / 2))

    extracted_img = image[top:bottom, left:right]
    return extracted_img


def track_length_histogram(dataset):
    lengths = []
    for i in dataset.global_ids:
        lengths.append(len(os.listdir(dataset.dir + f"/{i}")))
    plt.hist(lengths, bins=100)
    plt.title("Track Length Histogram")
    plt.xlabel("Track Length")
    plt.ylabel("Count")
    plt.show()