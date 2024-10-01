import os
import random
import functools

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import torch
import torchvision
import torchvision.transforms.v2 as v2
import matplotlib.pyplot as plt

class ReID_dataset(torch.utils.data.Dataset):
    def __init__(self, dir, transform=None):
        self.dir = dir
        self.transform = transform

        self.global_ids = os.listdir(self.dir)
        self.global_ids = [int(i) for i in self.global_ids]
        self.global_ids.sort()

        for i in range(len(self.global_ids) - 1, -1, -1):
            items = os.listdir(self.dir + f"/{self.global_ids[i]}")
            if len(items) <= 15: 
                self.global_ids.pop(i)


    def random_global(self, ignore=None):
        if ignore:
            items = [i for i in self.global_ids if i != ignore]
        else: 
            items = self.global_ids
        return random.choice(items)


    def random_local(self, global_id, ignore=None, min_dist=5):
        items = [int(i.strip(".jpg")) for i in os.listdir(self.dir + f"/{global_id}")]
        if ignore: items = [i for i in items if abs(i - ignore) > min_dist]
        return random.choice(items)
    

    def __len__(self):
        return len(self.global_ids)
    

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
    pad = v2.Pad(padding)

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

