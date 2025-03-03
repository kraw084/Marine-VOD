import torch
import numpy as np
import os

#from YOLOX.exps.custom.yolox_x_ablation import Exp
#from YOLOX.tools.demo import Predictor

from .Config import Config


class YoloV5ObjectDetector:
    """Wrapper class for still image yolov5 model"""
    def __init__(self, weight_path, classes, colours, conf=0.45, iou=0.6, img_size=1280, cuda=None):
        self.weight_path = weight_path
        self.conf = conf
        self.iou = iou
        self.img_size = img_size
        self.cuda = cuda if not cuda is None else torch.cuda.is_available()

        self.num_to_class = classes
        self.num_to_colour = colours

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

    def __call__(self, im):
        return self.predict(im)
    
    def xywhcl(self, im):
        """Generates a prediction for im and returns a list of 1x6 arrays corrosponding to 
        the x, y, w, h, conf and label codes of each prediction"""
        pred = self(im).xywh[0].cpu().numpy()

        return pred


class PublicDetectionsDetector:
    def __init__(self, vid_name, classes, colours, detector="FRCNN", conf=0.45, half=0):
        self.conf = conf
        self.iou = None

        self.num_to_class = classes
        self.num_to_colour = colours

        self.vid_name = vid_name
        self.detector = detector

        if detector not in ("DPM", "FRCNN", "SDP"): raise ValueError("detector must be DPM, FRCNN or SDP")
        set_folder = "train" if os.path.isdir(f"MOT17/train/{vid_name}-{detector}") else "test"
        det_file = open(f"MOT17/{set_folder}/{vid_name}-{detector}/det/det.txt")
        self.dets = [tuple([float(num) for num in line.strip("/n").split(",")]) for line in det_file.readlines()]
        det_file.close()

        #whole video
        self.frame_counter_init = 0
        self.max_frame = max([d[0] for d in self.dets])

        #first half of video
        if half == 1:
            self.max_frame = (self.max_frame//2) - 1

        #second half of video
        if half == 2:
            self.frame_counter_init = (self.max_frame + 1)//2

        
        self.frame_counter = self.frame_counter_init

    def update_parameters(self, conf=0.45, iou=0.6):
        self.conf = conf
    
    def xywhcl(self, im):
        """Generates a prediction for im and returns a list of 1x6 arrays corrosponding to 
        the x, y, w, h, conf and label codes of each prediction"""

        frame_dets = []
        for frame, id, top_left_x, top_left_y, width, height, conf, *coords in self.dets:
            if frame - 1 != self.frame_counter: continue
            if conf < self.conf: continue

            center_x = top_left_x + width/2
            center_y = top_left_y + height/2
            box = np.array([center_x, center_y, width, height, conf, 0])

            frame_dets.append(box)

        self.frame_counter += 1

        if self.frame_counter > self.max_frame:
            self.frame_counter = self.frame_counter_init

        return frame_dets


class YoloXObjectDetector:
    def __init__(self, weight_path, classes, colours, conf=0.6, iou=0.6, cuda=None):
        self.weight_path = weight_path
        self.conf = conf
        self.iou = iou
        self.cuda = cuda if not cuda is None else torch.cuda.is_available()

        self.num_to_class = classes
        self.num_to_colour = colours

        device = "gpu" if self.cuda else "cpu"
        self.exp = Exp()
        self.exp.test_conf = self.conf
        self.exp.nmsthre = self.iou

        model = self.exp.get_model()
        ckpt_file = self.weight_path
        ckpt = torch.load(ckpt_file, map_location="cpu")
        model.load_state_dict(ckpt["model"])

        if device == "gpu":
            model.cuda()
        model.eval()

        self.model = Predictor(model, self.exp, self.num_to_class, device=device, legacy=True)

    def update_parameters(self, conf=0.6, iou=0.6):
        self.conf = conf
        self.exp.test_conf = conf
        self.model.confthre = conf

        self.iou = iou
        self.exp.nmsthre = iou
        self.model.nmsthre = iou

    def predict(self, im):
        results, _ = self.model.inference(im)
        return results

    def __call__(self, im):
        return self.predict(im)
    
    def xywhcl(self, im):
        """Generates a prediction for im and returns a list of 1x6 arrays corrosponding to 
        the x, y, w, h, conf and label codes of each prediction"""
        pred = self(im)[0].cpu().numpy()

        im_size_ratio = min(self.exp.test_size[0] / im.shape[0], self.exp.test_size[1] / im.shape[1])
        pred[:, :4] = pred[:, :4] / im_size_ratio
        
        formatted_pred = []
        for bbox in pred:            
            x_center = (bbox[0] + bbox[2]) / 2
            y_center = (bbox[1] + bbox[3]) / 2
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            conf = bbox[4] * bbox[5]
            label = bbox[6]
            formatted_pred.append(np.array([x_center, y_center, w, h, conf, label]))

        return formatted_pred


def create_urchin_model(cuda = None):
    return YoloV5ObjectDetector("models/urchin_bot.pt",
                                ["Evechinus chloroticus","Centrostephanus rodgersii"],
                                [(235, 250,  15), (227, 11, 11)],
                                cuda=cuda)


def create_brackish_model(cuda = None):
    bot = YoloV5ObjectDetector("models/brackishMOT_botV1.pt",
                                ["Jellyfish", "Fish", "Crab", "Shrimp", "Starfish", "Smallfish"],
                                [(212, 70, 200), (29, 32, 224), (224, 112, 20), (231, 235, 19), (204, 16, 16), (48, 219, 29)],
                                conf = 0.3,
                                iou = 0.4,
                                cuda = cuda
                                )
    
    bot.model.agnostic = True
    return bot


def create_MOT_model(vid_name, half=0):
    return PublicDetectionsDetector(vid_name, ["Person"], [(255, 0, 0)], conf=0.6, half=half, detector=Config.MOTDetector)


def create_MOT_YOLOX_model(cuda = None):
    return YoloXObjectDetector("YOLOX/weights/bytetrack_ablation.pth.tar",
                               ["Person"],
                               [(255, 0, 0)],
                               conf = 0.6,
                               iou = 0.6,
                               cuda = cuda
                               )


if __name__ == "__main__":
    #code for testing detectors and displaying their predictions
    model = create_urchin_model()
    model.update_parameters(0.001, 1)
    results = model.xywhcl("C:/Users/kelha/Documents/Uni/Summer Research/Urchin-Detector/data/images/im2360606.JPG")
    print(f"Number of preds: {len(results)}")

    import cv2
    from Video_utils import resize_image
    from VOD_utils import annotate_image

    im = cv2.imread("C:/Users/kelha/Documents/Uni/Summer Research/Urchin-Detector/data/images/im2360606.JPG")
    annotate_image(im, results, model.num_to_class, model.num_to_colour)
    im = resize_image(im, new_width=640)
    cv2.imshow("im", im)
    cv2.waitKey(0)