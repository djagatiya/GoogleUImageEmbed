import timm

import torch
import torch.nn as nn
from torchvision import transforms, models


class MyModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.base_model = models.efficientnet_b7(pretrained=True)
        self.feature_extractor = torch.nn.Sequential(
            *list(self.base_model.children())[:-2],
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        self.avg_pooler = nn.AdaptiveAvgPool1d(64)

    def forward(self, x):
        x = transforms.functional.resize(x,size=[256, 256])
        x = transforms.functional.center_crop(x, 224)

        x = x/255.0
        x = transforms.functional.normalize(x, 
                                                mean=[0.485, 0.456, 0.406], 
                                                std=[0.229, 0.224, 0.225])
        x = self.feature_extractor(x)
        x = self.avg_pooler(x)
        
        return x


model = MyModel()
model.eval()
pass