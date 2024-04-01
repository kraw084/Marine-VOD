import torch


class YoloV5ObjectDetector:
    """Wrapper class for still image yolov5 model"""
    def __init__(self, weight_path, classes, conf=0.45, iou=0.6, img_size=1280, cuda=None):
        self.weight_path = weight_path
        self.conf = conf
        self.iou = iou
        self.img_size = img_size
        self.cuda = cuda if not cuda is None else torch.cuda.is_available()

        self.num_to_class = classes

        self.model = torch.hub.load("yolov5", "custom", path=self.weight_path, source="local")
        self.model.cuda() if self.cuda else self.model.cpu()
        self.model.conf = self.conf
        self.model.iou = self.iou

    def update_parameters(self, conf=0.45, iou=0.6):
        self.conf = conf
        self.model.conf = conf
        self.iou = iou
        self.model.iou = iou

    def predict(self, im):
        return self.model(im, size = self.img_size)

    def predict_batch(self, ims):
        return [self.predict(im) for im in ims]
    
    def pred_generator(self, ims):
        for im in ims:
            pred = self.predict(im)
            yield pred

    def label(self, number):
        return self.num_to_class[number]

    def __call__(self, im):
        return self.predict(im)
    
    def xywhcl(self, im):
        """Generates a prediction for im and returns a list of 1x6 arrays corrosponding to 
        the x, y, w, h, conf and label codes of each prediction"""
        pred = self(im).xywh[0].cpu().numpy()
        for row in pred:
            print(row)
            row[0] = round(row[0])
            row[1] = round(row[1])
            row[2] = round(row[2])
            row[3] = round(row[3])

            row[4] = round(row[4], 2)

        return pred


def create_urchin_model(cuda = None):
    return YoloV5ObjectDetector("models/urchin_bot.pt",
                                ["Evechinus chloroticus","Centrostephanus rodgersii"],
                                cuda=cuda)


def frame_predictions(model, frames):
    """Generates predictions on each frame of a video"""
    predictions = []
    for frame in frames:
        pred = model.xywhcl(frame)
        predictions.append(pred)
    
    return predictions


#model = create_urchin_model()
#model.xywhcl("C:/Users/kelha/Documents/Uni/Summer Research/Urchin-Detector/data/images/im2360606.JPG")