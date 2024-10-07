import torch

try:
    from reid.reid_data_utils import resize_with_aspect_ratio
except ModuleNotFoundError:
    from reid_data_utils import resize_with_aspect_ratio


def load_resnet():
    return torch.hub.load("pytorch/vision:v0.10.0", "resnet50", num_classes=128, weights=None)


def save_model(model, path, name):
    """Save model weights to path/name"""
    torch.save(model.state_dict(), f"{path}/{name}")


def load_model(model, path, name):
    """Load model weights from path/name"""
    model.load_state_dict(torch.load(f"{path}/{name}", weights_only=True))


def cosine_distance(x1, x2):
    return 1 - torch.nn.functional.cosine_similarity(x1, x2, dim = -1)


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
    #load_model(model, "runs/siamese_triplet_resnet_lowMargin", "models/Epoch_99.pt")
    load_model(model, "runs/all_triplets", "models/Epoch_99.pt")
    return ReIDModel(model, (224, 224), 0.3)