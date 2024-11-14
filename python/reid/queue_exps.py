import os
import functools
import itertools
import random

import torch.utils
import torch.utils.data

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import torch
import torchvision.transforms.v2 as v2
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import albumentations as A

from reid import load_model, save_model, load_resnet, cosine_distance, ReIDModel
from reid_data_utils import resize_with_aspect_ratio, track_length_histogram, ReIDRandomTripletDataset, ReIDMiningDataset, view_dataset

from train_reid import ReIDTrainer, AblumentationsWrapper, ReIDTrainerBatchHardMining, ReIDTrainerBatchSemiHardMining, ReIDTrainerBatchAllMining

"""
#Random Triplet with less Aug
try:
    exp_name = "resnet_m05_randomTriplet_lessAug"

    resize_and_pad = functools.partial(resize_with_aspect_ratio, target_size=(224, 224))
    transform = v2.Compose([
                            resize_and_pad,
                            v2.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1), fill=0.5)
                            ])


    dataset = ReIDRandomTripletDataset(dir="C:/Users/kraw084/OneDrive - The University of Auckland/Desktop/reid_dataset_train", 
                                #items_per_id=0, 
                                min_track_length=30, 
                                transform=transform)



    print(f"Dataset size: {len(dataset)}")
    print(f"Number of images: {dataset.num_of_images()}")

    #create experiment folder 
    if not os.path.isdir(f"runs/{exp_name}"):
        os.mkdir(f"runs/{exp_name}")

    #create model
    resnet = load_resnet(use_head=True)
    resnet.cuda()

    #train model
    trainer = ReIDTrainer(model = resnet,
                            dataset = dataset,
                            epochs = 100,
                            batch_size = 16,
                            lr = 1e-4,
                            weight_decay = 5e-4,
                            margin = 0.5,
                            save_path = f"runs/{exp_name}",
                            save_freq = 1,
                            device = "cuda",
                            resume = None)


    trainer.train()

except Exception as e:
    print(e)

#Batch Hard
try:
    exp_name = "resnet_m05_BatchHard"

    resize_and_pad = functools.partial(resize_with_aspect_ratio, target_size=(256, 256))
    transform = v2.Compose([v2.GaussianBlur((3, 3), (0.01, 2.0)),
                            v2.ColorJitter(0.2, 0.1, 0.1, 0.05), #brightness, contrast, saturation, hue
                            resize_and_pad,
                            #v2.RandomHorizontalFlip(),
                            v2.RandomCrop((224, 224)),
                            AblumentationsWrapper(A.augmentations.dropout.CoarseDropout(fill_value=0.5, 
                                                                                        hole_height_range=(0.4, 0.6), 
                                                                                        hole_width_range=(0.4, 0.6),
                                                                                        p=0.2)),
                            v2.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1), fill=0.5)
                            ])


    dataset = ReIDMiningDataset(dir="C:/Users/kraw084/OneDrive - The University of Auckland/Desktop/reid_dataset_train", 
                                items_per_id=10, 
                                min_track_length=30, 
                                transform=transform)



    print(f"Dataset size: {len(dataset)}")
    print(f"Number of images: {dataset.num_of_images()}")

    #create experiment folder 
    if not os.path.isdir(f"runs/{exp_name}"):
        os.mkdir(f"runs/{exp_name}")

    #create model
    resnet = load_resnet(use_head=True)
    resnet.cuda()

    #train model
    trainer = ReIDTrainerBatchHardMining(model = resnet,
                            dataset = dataset,
                            epochs = 100,
                            batch_size = 16,
                            lr = 1e-4,
                            weight_decay = 5e-4,
                            margin = 0.5,
                            save_path = f"runs/{exp_name}",
                            save_freq = 1,
                            device = "cuda",
                            resume = None)


    trainer.train()

except Exception as e:
    print(e)


#Batch Semihard
try:
    exp_name = "resnet_m05_BatchSemiHard"

    resize_and_pad = functools.partial(resize_with_aspect_ratio, target_size=(256, 256))
    transform = v2.Compose([v2.GaussianBlur((3, 3), (0.01, 2.0)),
                            v2.ColorJitter(0.2, 0.1, 0.1, 0.05), #brightness, contrast, saturation, hue
                            resize_and_pad,
                            #v2.RandomHorizontalFlip(),
                            v2.RandomCrop((224, 224)),
                            AblumentationsWrapper(A.augmentations.dropout.CoarseDropout(fill_value=0.5, 
                                                                                        hole_height_range=(0.4, 0.6), 
                                                                                        hole_width_range=(0.4, 0.6),
                                                                                        p=0.2)),
                            v2.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1), fill=0.5)
                            ])


    dataset = ReIDMiningDataset(dir="C:/Users/kraw084/OneDrive - The University of Auckland/Desktop/reid_dataset_train", 
                                items_per_id=10, 
                                min_track_length=30, 
                                transform=transform)



    print(f"Dataset size: {len(dataset)}")
    print(f"Number of images: {dataset.num_of_images()}")

    #create experiment folder 
    if not os.path.isdir(f"runs/{exp_name}"):
        os.mkdir(f"runs/{exp_name}")

    #create model
    resnet = load_resnet(use_head=True)
    resnet.cuda()

    #train model
    trainer = ReIDTrainerBatchSemiHardMining(model = resnet,
                            dataset = dataset,
                            epochs = 100,
                            batch_size = 16,
                            lr = 1e-4,
                            weight_decay = 5e-4,
                            margin = 0.5,
                            save_path = f"runs/{exp_name}",
                            save_freq = 1,
                            device = "cuda",
                            resume = None)


    trainer.train()

except Exception as e:
    print(e)
"""
    
#Batch BatchAll
try:
    exp_name = "resnet_m05_BatchAll_lessAug"

    """
    resize_and_pad = functools.partial(resize_with_aspect_ratio, target_size=(256, 256))
    transform = v2.Compose([v2.GaussianBlur((3, 3), (0.01, 2.0)),
                            v2.ColorJitter(0.2, 0.1, 0.1, 0.05), #brightness, contrast, saturation, hue
                            resize_and_pad,
                            #v2.RandomHorizontalFlip(),
                            v2.RandomCrop((224, 224)),
                            AblumentationsWrapper(A.augmentations.dropout.CoarseDropout(fill_value=0.5, 
                                                                                        hole_height_range=(0.4, 0.6), 
                                                                                        hole_width_range=(0.4, 0.6),
                                                                                        p=0.2)),
                            v2.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1), fill=0.5)
                            ])
    """

    resize_and_pad = functools.partial(resize_with_aspect_ratio, target_size=(224, 224))
    transform = v2.Compose([
                            resize_and_pad,
                            v2.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1), fill=0.5)
                            ])


    dataset = ReIDMiningDataset(dir="C:/Users/kraw084/OneDrive - The University of Auckland/Desktop/reid_dataset_train", 
                                items_per_id=10, 
                                min_track_length=30, 
                                transform=transform)



    print(f"Dataset size: {len(dataset)}")
    print(f"Number of images: {dataset.num_of_images()}")

    #create experiment folder 
    if not os.path.isdir(f"runs/{exp_name}"):
        os.mkdir(f"runs/{exp_name}")

    #create model
    resnet = load_resnet(use_head=True)
    resnet.cuda()

    #train model
    trainer = ReIDTrainerBatchAllMining(model = resnet,
                            dataset = dataset,
                            epochs = 100,
                            batch_size = 16,
                            lr = 1e-4,
                            weight_decay = 5e-4,
                            margin = 0.5,
                            save_path = f"runs/{exp_name}",
                            save_freq = 1,
                            device = "cuda",
                            resume = None)


    trainer.train()

except Exception as e:
    print(e)



