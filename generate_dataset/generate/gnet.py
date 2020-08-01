#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: swati
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 23:16:57 2019

@author: swati
"""

import torchvision as TV
import torch.nn as nn
import sys

class GoogleNet(nn.Module):
    def __init__(self):
        super(GoogleNet, self).__init__()

        googlenet = TV.models.googlenet(pretrained=True)
        # exist DropOut
        self.googlenet = nn.Sequential(*(list(googlenet.children())[:-1]))
        self.trans = TV.transforms.Compose([TV.transforms.ToTensor()])
#                                            TV.transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                                                    std=[0.229, 0.224, 0.225])])

    def forward(self, x):
       
        x = self.trans(x)
        x = x.unsqueeze(0)
        x = self.googlenet(x)
        x = x.reshape((x.shape[0], -1))

        return x
m = GoogleNet()
print(m)