"""
StyleLoss.py

Faisal Habib
April 3, 2020

Description:
Perception Loss implementation
"""

import torch
import torch.nn as nn
import torchvision.models as models


class StyleLoss(nn.Module):
    def __init__(self):
        super(StyleLoss, self).__init__()
        blocks = [models.vgg16(pretrained=True).features[:4].eval(),
                  models.vgg16(pretrained=True).features[4:9].eval()]

        for block in blocks:
            for param in block:
                param.requires_grad = False

        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate

    @staticmethod
    def __gram_matrix(image):
        a, b, c, d = image.size()

        features = image.view(a * b, c * d)
        G = torch.mm(features, features.t())

        return G.div(a * b * c * d)

    def forward(self, x, y):
        x = x[:, 0:3, :, :]
        y = y[:, 0:3, :, :]

        x = self.transform(x, mode='bilinear', size=(224, 224), align_corners=False)
        y = self.transform(y, mode='bilinear', size=(224, 224), align_corners=False)

        for block in self.blocks:
            x = block(x)
            y = block(y)

        G_x = self.__gram_matrix(image=x)
        G_y = self.__gram_matrix(image=y)

        return nn.functional.l1_loss(G_x, G_y)
