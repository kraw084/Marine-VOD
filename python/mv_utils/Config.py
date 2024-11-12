import torch
import os

class Config:
    urchin_vid_path = r"C:\Users\kraw084\OneDrive - The University of Auckland\Desktop\urchin video\All"
    urchin_vid_trimmed_path = r"C:\Users\kraw084\OneDrive - The University of Auckland\Desktop\urchin video\Trimmed"
    cuda = torch.cuda.is_available() #use cuda or not
    os.environ["TQDM_DISABLE"] = "1"

    draw_labels = True #draw labels when drawing bounding boxes
    minimal_labels = True #only draw id number and conf score in label
    labels_in_box = True #draw labels inside their box rather than on top
    data_text_colour = (0, 0, 0) #color to draw object counts in
    box_thickness = 1.5 #percentage multiplier of bounding box thickness
    label_font_size = 1.5 #percentage multiplier of font size
    label_font_thickness = 1.5 #percentage multiple of font thickness

    low_conf_colour = None #(255, 255, 255) #colour low conf labels differently (useful for testing byte track), set to none to colour them normally
    low_conf_th = 0.45 #conf score to be considered low conf for colouring purposes

    MOTDetector = "FRCNN" #What public detection set to use for MOT17 (FRCNN, SDP or DPM)