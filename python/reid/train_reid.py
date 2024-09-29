import os
import random
import functools

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import torch
import torchvision
import torchvision.transforms.v2 as v2
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from tqdm import tqdm


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
    

def load_resnet():
    return torch.hub.load("pytorch/vision:v0.10.0", "resnet50", num_classes=128, weights=None)


def save_model(model, path, name):
    """Save model weights to path/name"""
    torch.save(model.state_dict(), f"{path}/{name}")


def load_model(model, path, name):
    """Load model weights from path/name"""
    model.load_state_dict(torch.load(f"{path}/{name}", weights_only=True))


def train(model, dataset, epochs, batch_size, lr, weight_decay, save_path, save_freq, device, resume=None):
    """Performs the iterative deep clustering training loop"""

    #load model to resume training or if this is a new model set up folders
    if resume:
        load_model(model, save_path, f"models/Epoch_{resume}.pt")
    else:
        os.mkdir(f"{save_path}/models")
        os.mkdir(f"{save_path}/logs")

    #create dataloader and optimiser
    loader = torch.utils.data.DataLoader(dataset, batch_size)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_func = torch.nn.TripletMarginWithDistanceLoss(distance_function=cosine_distance)

    #create tensorboard writer
    writer = SummaryWriter(f"{save_path}/logs")

    #training iteration for each epoch
    for e_i in range(0 if resume is None else resume + 1, epochs):
        model.train()
        progress_bar = tqdm(loader, desc = f"Epoch {e_i}/{epochs}", bar_format = "{l_bar}{bar:20}{r_bar}")

        #loop over all batchs in the epoch and perform the training step
        epoch_loss = 0
        for anchor, positive, negative in progress_bar:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            anchor_embedding = model(anchor)
            positive_embedding = model(positive)
            negative_embedding = model(negative)

            loss = loss_func(anchor_embedding, positive_embedding, negative_embedding)

            opt.zero_grad()
            loss.backward()
            opt.step()

            batch_loss = loss.item()

            epoch_loss += batch_loss
            progress_bar.set_postfix({"loss": batch_loss})

            #add batch loss to tensorboard
            writer.add_scalar('Loss/Batch_Loss', batch_loss, e_i * len(loader) + progress_bar.n)

        #print epoch average loss and add it to the tensorboard
        avg_epoch_loss = epoch_loss / len(loader)
        writer.add_scalar('Loss/Epoch_Avg_Loss', avg_epoch_loss, e_i)
        print(f"Epoch {e_i} finished - Avg loss: {avg_epoch_loss}")

        #save model weights and add embedding to the tensor board
        if (e_i % save_freq) == 0 or e_i == epochs - 1:
            save_model(model, save_path, f"models/Epoch_{e_i}.pt")

        print()
    
    writer.close()
    print("Training finished!")


def cosine_distance(x1, x2):
    return 1 - torch.nn.functional.cosine_similarity(x1, x2, dim = -1)


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


if __name__ == "__main__":
    exp_name = "siamese_triplet_resnet_initial_test"

    resize_and_pad = functools.partial(resize_with_aspect_ratio, target_size=(256, 256))
    transform = v2.Compose([resize_and_pad,
                            v2.ColorJitter(0.2, 0.1, 0.1, 0.05), #brightness, contrast, saturation, hue
                            v2.RandomHorizontalFlip(),
                            v2.RandomCrop((224, 224)),
                            v2.GaussianBlur((3, 3), (0.01, 2.0)),
                            v2.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1))])


    dataset = ReID_dataset("C:/Users/kraw084/OneDrive - The University of Auckland/Desktop/reid_dataset", transform)
    resnet = load_resnet()
    resnet.cuda()

    #view_dataset(dataset)

    #create experiment folder 
    if not os.path.isdir(f"runs/{exp_name}"):
        os.mkdir(f"runs/{exp_name}")
    
    train(model = resnet,
          dataset = dataset,
          epochs = 100,
          batch_size = 64,
          lr = 1e-4,
          weight_decay = 0.0005,
          save_path = f"runs/{exp_name}",
          save_freq = 1,
          device = "cuda",
          resume = None)
   