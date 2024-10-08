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


class ReIDTrainer:
    def __init__(self, model, dataset, epochs, batch_size, lr, margin, weight_decay, save_path, save_freq, device, resume=None):
        self.model = model
        self.dataset = dataset
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.save_path = save_path
        self.save_freq = save_freq
        self.device = device
        self.resume = resume
        self.margin = margin

        if resume:
            load_model(self.model, self.save_path, f"models/Epoch_{self.resume}.pt")
        else:
            os.mkdir(f"{self.save_path}/models")
            os.mkdir(f"{self.save_path}/logs")

        self.loader = torch.utils.data.DataLoader(dataset, batch_size)
        self.opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.loss_func = torch.nn.TripletMarginWithDistanceLoss(distance_function=cosine_distance, margin=margin)

        self.writer = SummaryWriter(f"{save_path}/logs")


    def process_batch(self, e_i, batch_args, progress_bar):
        anchor, positive, negative = batch_args
        anchor, positive, negative = anchor.to(self.device), positive.to(self.device), negative.to(self.device)

        anchor_embedding = self.model(anchor)
        positive_embedding = self.model(positive)
        negative_embedding = self.model(negative)

        loss = self.loss_func(anchor_embedding, positive_embedding, negative_embedding)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        batch_loss = loss.item()

        progress_bar.set_postfix({"loss": batch_loss})

        #add batch loss to tensorboard
        self.writer.add_scalar('Loss/Batch_Loss', batch_loss, e_i * len(self.loader) + progress_bar.n)

        return batch_loss
    

    def process_epoch(self, e_i):
        self.model.train()
        progress_bar = tqdm(self.loader, desc = f"Epoch {e_i}/{self.epochs}", bar_format = "{l_bar}{bar:20}{r_bar}")

        #loop over all batchs in the epoch and perform the training step
        epoch_loss = 0
        for batch in progress_bar:
            batch_loss = self.process_batch(e_i, batch, progress_bar)
            epoch_loss += batch_loss

        avg_epoch_loss = epoch_loss / len(self.loader)
        self.writer.add_scalar('Loss/Epoch_Avg_Loss', avg_epoch_loss, e_i)
        print(f"Epoch {e_i} finished - Avg loss: {avg_epoch_loss}")

        #save model weights and add embedding to the tensor board
        if (e_i % self.save_freq) == 0 or e_i == self.epochs - 1:
            save_model(self.model, self.save_path, f"models/Epoch_{e_i}.pt")

        print()


    def train(self):
        for e_i in range(0 if self.resume is None else self.resume + 1, self.epochs):
            self.process_epoch(e_i)

        self.writer.close()
        print("Training finished!")
    

class ReIDTrainerBatchAllMining(ReIDTrainer):
    def process_batch(self, e_i, batch_args, progress_bar):
        images, labels = batch_args
        images = images.reshape(-1, *images.shape[2:]).to(self.device)
        labels = labels.reshape(-1).numpy()

        triplets = []
        limit = 20
        for i in np.unique(labels):
            anchor_indicies = np.where(labels == i)[0]
            negative_indicies = np.where(labels != i)[0]
            negative_indicies = np.random.choice(negative_indicies, limit, replace=True)

            pairs = list(itertools.permutations(anchor_indicies, 2))
            triplets += [pair + (n,) for pair in pairs for n in negative_indicies]

        anchor_indicies, positive_indicies, negative_indicies = zip(*triplets)

        embeddings = self.model(images)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        anchor_embeddings = embeddings[np.array(anchor_indicies)]
        positive_embeddings = embeddings[np.array(positive_indicies)]
        negative_embeddings = embeddings[np.array(negative_indicies)]

        loss = self.loss_func(anchor_embeddings, positive_embeddings, negative_embeddings)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        batch_loss = loss.item()

        progress_bar.set_postfix({"loss": batch_loss})

        #add batch loss to tensorboard
        self.writer.add_scalar('Loss/Batch_Loss', batch_loss, e_i * len(self.loader) + progress_bar.n)

        return batch_loss



class ReIDTrainerBatchHardMining(ReIDTrainer):
    def process_batch(self, e_i, batch_args, progress_bar):
        images, labels = batch_args
        images = images.reshape(-1, *images.shape[2:]).to(self.device)
        labels = labels.reshape(-1).to(self.device)

        embeddings = self.model(images)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        dist_mat = 1 - torch.matmul(embeddings, embeddings.T)

        #triplets = []
        #for i in range(dist_mat.shape[0]):
        #    row = dist_mat[i].detach().cpu().numpy()
        #    positive_candidate_incides = np.where(labels == labels[i])[0] 
        #    negative_candidate_incides = np.where(labels != labels[i])[0]

        #    hardest_positive = positive_candidate_incides[np.argmax(row[positive_candidate_incides])]
        #    hardest_negative = negative_candidate_incides[np.argmin(row[negative_candidate_incides])]

        #    triplets.append((i, hardest_positive, hardest_negative))

        #anchor_indicies, positive_indicies, negative_indicies = zip(*triplets)
        
        positive_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        pos_dist_mat = dist_mat * positive_mask.float() - ((~positive_mask).float() * 1e6)
        neg_dist_mat = dist_mat * (~positive_mask).float() + (positive_mask.float() * 1e6)
        
        anchor_indicies = torch.arange(dist_mat.shape[0])
        positive_indicies = torch.argmax(pos_dist_mat, dim=1)

        # pos + m > neg > pos
        semihard_neg_mask = (~positive_mask) & (dist_mat > dist_mat[:, positive_indicies]) & (dist_mat < dist_mat[:, positive_indicies] + self.margin)
        
        rows, cols = torch.nonzero(semihard_neg_mask, as_tuple=True)

        no_semihard = torch.any(semihard_neg_mask, dim=1)
        assert torch.all(no_semihard), "items with no semihards found"

        random_order = torch.randperm(rows.shape[0])
        rows = rows[random_order]
        cols = cols[random_order]
        unique_val, row_indicies = torch.unique(rows, return_inverse=True)
        col_indices = cols[row_indicies]
        #FIX
        
        negative_indicies = col_indices
        #negative_indicies = torch.argmin(neg_dist_mat, dim=1)

        anchor_embeddings = embeddings[anchor_indicies]
        positive_embeddings = embeddings[positive_indicies]
        negative_embeddings = embeddings[negative_indicies]

        avg_anc_pos_dist = (dist_mat[anchor_indicies, positive_indicies]).mean()
        avg_anc_neg_dist = (dist_mat[anchor_indicies, negative_indicies]).mean()
        self.writer.add_scalar('TripletSelection/Avg_Anc_Pos_Dist', avg_anc_pos_dist, e_i * len(self.loader) + progress_bar.n)
        self.writer.add_scalar('TripletSelection/Avg_Anc_Neg_Dist', avg_anc_neg_dist, e_i * len(self.loader) + progress_bar.n)

        loss = self.loss_func(anchor_embeddings, positive_embeddings, negative_embeddings)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        batch_loss = loss.item()

        progress_bar.set_postfix({"loss": batch_loss})

        #add batch loss to tensorboard
        self.writer.add_scalar('Loss/Batch_Loss', batch_loss, e_i * len(self.loader) + progress_bar.n)

        return batch_loss


class AblumentationsWrapper():
    def __init__(self, transformation):
        self.transformation = transformation

    def __call__(self, img):
        img = img.permute(1, 2, 0).numpy()
        transformed_img = self.transformation(image=img)["image"]
        return torch.from_numpy(transformed_img).permute(2, 0, 1)


if __name__ == "__main__":
    exp_name = "resnet_m05_batchSemiHard"

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


    dataset = ReIDMiningDataset(dir="C:/Users/kraw084/OneDrive - The University of Auckland/Desktop/reid_dataset_v2", 
                                items_per_id=10, 
                                min_track_length=50, 
                                transform=transform)
    
    #view_dataset(dataset)

    print(f"Dataset size: {len(dataset)}")
    print(f"Number of images: {dataset.num_of_images()}")

    #create experiment folder 
    if not os.path.isdir(f"runs/{exp_name}"):
        os.mkdir(f"runs/{exp_name}")

    #create model
    resnet = load_resnet()
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
                          save_freq = 10,
                          device = "cuda",
                          resume = None)
    

    trainer.train()