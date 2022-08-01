import timm
import torch
import torch.nn as nn
from torchvision import transforms


class MyModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.feature_extractor = timm.create_model("efficientnet_b4", num_classes=0, pretrained=True)
        self.avg_pooler = nn.AdaptiveAvgPool1d(64)

    def forward(self, x):
        x = transforms.functional.resize(x,size=[320, 320])
        x = x / 255.0
        x = transforms.functional.normalize(x, 
                                                mean=[0.485, 0.456, 0.406], 
                                                std=[0.229, 0.224, 0.225])
        x = self.feature_extractor(x)
        x = self.avg_pooler(x)
        return x


model = MyModel()
model.eval()
pass