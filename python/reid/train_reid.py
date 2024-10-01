import os
import random
import functools

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import torch
import torchvision.transforms.v2 as v2
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from reid import load_model, save_model, load_resnet, cosine_distance, plot_embeddings, ReIDModel
from reid_data_utils import resize_with_aspect_ratio, ReID_dataset


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
    loss_func = torch.nn.TripletMarginWithDistanceLoss(distance_function=cosine_distance, margin=2)

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

            encapsulated_model = ReIDModel(model, (224, 224), 0.3)
            plot_embeddings(encapsulated_model, dataset, writer, e_i)

        print()
    
    writer.close()
    print("Training finished!")

if __name__ == "__main__":
    exp_name = "siamese_triplet_resnet"

    resize_and_pad = functools.partial(resize_with_aspect_ratio, target_size=(256, 256))
    transform = v2.Compose([resize_and_pad,
                            v2.ColorJitter(0.2, 0.1, 0.1, 0.05), #brightness, contrast, saturation, hue
                            v2.RandomHorizontalFlip(),
                            v2.RandomCrop((224, 224)),
                            v2.GaussianBlur((3, 3), (0.01, 2.0)),
                            v2.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1))])


    dataset = ReID_dataset("C:/Users/kraw084/OneDrive - The University of Auckland/Desktop/reid_dataset_v2", transform)
    print(f"Dataset size: {len(dataset)}")

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
          weight_decay = 5e-4,
          save_path = f"runs/{exp_name}",
          save_freq = 5,
          device = "cuda",
          resume = None)
   