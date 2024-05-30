import torch

class Config():
    def __init__(self):
        self.config_dict = dict()


    def set(self, property, value):
        self.config_dict[property] = value
    
    def get(self, property):
        if property not in self.config_dict:
            return None
        else:
            return self.config_dict[property]
        

#Setting up config
c = Config()

c.set("drive", "D") #what drive to find files on (D or E)
c.set("cuda", torch.cuda.is_available()) #use cuda or not

c.set("draw labels", True) #draw labels when drawing bounding boxes
c.set("min label", False) #only draw id number and conf score in label
c.set("label in box", False) #draw labels inside their box rather than on top