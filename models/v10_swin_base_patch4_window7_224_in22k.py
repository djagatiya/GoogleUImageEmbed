# Run
# 32.3s - GPU

# Public Score
# 0.350

import timm

import torch
import torch.nn as nn
from torchvision import transforms


class MyModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.base_model = timm.create_model("swin_base_patch4_window7_224_in22k", pretrained=True)
        self.feature_extractor = torch.nn.Sequential(*list(self.base_model.children())[:-1])
        self.avg_pooler = nn.AdaptiveAvgPool1d(64)

    def forward(self, x):
        x = transforms.functional.resize(x,size=[256, 256])
        x = transforms.functional.center_crop(x, 224)

        x = x/255.0
        x = transforms.functional.normalize(x, 
                                                mean=[0.485, 0.456, 0.406], 
                                                std=[0.229, 0.224, 0.225])
        x = self.feature_extractor(x)
        x = x.mean(dim=1)
        x = self.avg_pooler(x)
        
        return x


model = MyModel()
model.eval()
pass