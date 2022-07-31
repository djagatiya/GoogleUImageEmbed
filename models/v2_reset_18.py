# Public Score: 0.222
# 21.4s - GPU

import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms

class MyModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.base_model = models.resnet10(pretrained=True)
    self.feature_extractor = torch.nn.Sequential(
        *list(self.base_model.children())[:-1],
        nn.Flatten(),
        nn.AdaptiveAvgPool1d(64)
    )

  def forward(self, x):
    x = transforms.functional.resize(x,size=[224, 224])
    x = x/255.0
    x = transforms.functional.normalize(x, 
                                            mean=[0.485, 0.456, 0.406], 
                                            std=[0.229, 0.224, 0.225])
    return self.feature_extractor(x)

model = MyModel()
model.eval()
passresnet50