
import torch

from .train_reid import load_model, load_resnet, resize_with_aspect_ratio


class ReIDModel:
    def __init__(self, model, size=(224, 224), padding=0.3):
        self.model = model
        self.model.cuda()
        self.model.eval()

        self.size = size
        self.padding = padding

    def format_img(self, img):
        img = torch.tensor(img).float() / 255
        img = img.permute(2, 0, 1)
        return resize_with_aspect_ratio(img, self.size)

    def extract_feature(self, img, is_batch=True, numpy=False):
        if not is_batch: img = img.unsqueeze(0)
        img_vec = self.model(img.cuda())

        if numpy:
            return img_vec.cpu().detach().numpy()
        else:
            return img_vec
        
    def vector_similarity(self, vec1, vec2):
        return torch.nn.functional.cosine_similarity(vec1, vec2, dim=-1).item()
    
    def batch_vector_similarity(self, vecs1, vecs2, numpy=False):
        vecs1 = torch.nn.functional.normalize(vecs1, p=2, dim=1)
        vecs2 = torch.nn.functional.normalize(vecs2, p=2, dim=1)

        cosine_sim_matrix = torch.matmul(vecs1, vecs2.T)

        if numpy:
            return cosine_sim_matrix.cpu().detach().numpy()
        else:
            return cosine_sim_matrix
    

def create_reid_model():
    model = load_resnet()
    load_model(model, "runs/siamese_triplet_resnet_lowMargin", "models/Epoch_99.pt")
    return ReIDModel(model, (224, 224), 0.3)
    