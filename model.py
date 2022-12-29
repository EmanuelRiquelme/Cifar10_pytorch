import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights

class Model(nn.Module):
    def __init__(self,num_classes = 10):
        super(Model, self).__init__()
        self.num_classes = num_classes
        self.model = self.__load_model__()

    def __load_model__(self):
        model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(512,self.num_classes)
        return model

    def forward(self,input_data):
        return self.model(input_data)
