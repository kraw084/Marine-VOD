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

def reid_experiment(exp_name, dataset, trainer_type):
    print(f"Experiment: {exp_name}")    
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of images: {dataset.num_of_images()}")

    resnet = load_resnet(use_head=True)
    resnet.cuda()

    if trainer_type == "randomTriplet":
         trainer_class = ReIDTrainer
    elif trainer_type == "BatchHard":
        trainer_class = ReIDTrainerBatchHardMining
    elif trainer_type == "BatchSemiHard":
        trainer_class = ReIDTrainerBatchSemiHardMining
    elif trainer_type == "BatchAll":
        trainer_class = ReIDTrainerBatchAllMining

    #create experiment folder 
    if not os.path.isdir(f"runs/{exp_name}"):
        os.mkdir(f"runs/{exp_name}")

    #train model
    trainer = trainer_class(model = resnet,
                            dataset = dataset,
                            epochs = 60,
                            batch_size = 16,
                            lr = 1e-4,
                            weight_decay = 5e-4,
                            margin = 0.5,
                            save_path = f"runs/{exp_name}",
                            save_freq = 1,
                            device = "cuda",
                            resume = None)
    
    trainer.train()


def create_dataset(type, split, padding, transform):
    d_path = f"C:/Users/kraw084/OneDrive - The University of Auckland/Desktop/reid_dataset_{split}"

    if padding == 0.0:
        d_path += "_pad00"
    elif padding == 0.6:
        d_path += "_pad06"

    if type == "random":
        dataset = ReIDRandomTripletDataset(dir=d_path, 
                                    min_track_length=30, 
                                    transform=transform)
    elif type == "mining":
            dataset = ReIDMiningDataset(dir=d_path, 
                                items_per_id=10, 
                                min_track_length=30, 
                                transform=transform)
            
    return dataset


resize_and_pad_224 = functools.partial(resize_with_aspect_ratio, target_size=(224, 224))
transform_less_aug = v2.Compose([
                            resize_and_pad_224,
                            v2.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1), fill=0.5)
                            ])


resize_and_pad_256 = functools.partial(resize_with_aspect_ratio, target_size=(256, 256))
transform = v2.Compose([v2.GaussianBlur((3, 3), (0.01, 2.0)),
                        v2.ColorJitter(0.2, 0.1, 0.1, 0.05), #brightness, contrast, saturation, hue
                        resize_and_pad_256,
                        v2.RandomCrop((224, 224)),
                        AblumentationsWrapper(A.augmentations.dropout.CoarseDropout(fill_value=0.5, 
                                                                                    hole_height_range=(0.4, 0.6), 
                                                                                    hole_width_range=(0.4, 0.6),
                                                                                    p=0.2)),
                        v2.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1), fill=0.5)
                        ])



selection_methods = ["randomTriplet", "BatchHard", "BatchSemiHard", "BatchAll"][1:]
dataset_types = ["random", "mining", "mining", "mining"][1:]
paddings = [0.0, 0.3, 0.6]


for sel_method, d_type in zip(selection_methods, dataset_types):
    for padding in paddings:
        padding_string = str(padding).replace(".", "")
        exp_name = f"big_test/resnet50_m05_{sel_method}_{padding_string}"
        dataset = create_dataset(d_type, "train", padding, transform)
        reid_experiment(exp_name, dataset, sel_method)