import torch

class Config:
    drive = "D" #what drive to find files on (D or E)
    cuda = torch.cuda.is_available() #use cuda or not

    draw_labels = True #draw labels when drawing bounding boxes
    minimal_labels = True #only draw id number and conf score in label
    labels_in_box = True #draw labels inside their box rather than on top
    data_text_colour = (0, 0, 0)
    label_font_size = 0.75
    label_font_thickness = 1


   