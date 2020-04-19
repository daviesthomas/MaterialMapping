"""
GradientPenalty.py

Faisal Habib
April 10, 2020

Description:
Gradient Penalty implementation adapted from CSC 2516 PA #5 (DCGAN)
"""

import torch
import torch.nn as nn
from torch.autograd import Variable


class GradientPenalty(nn.Module):
    def __init__(self, Discriminator):
        super(GradientPenalty, self).__init__()

        self.discriminator = Discriminator

    def forward(self, fake, real, input_img=None, material_img=None):
        alpha = torch.rand(real.shape[0], 1, 1, 1)
        alpha = alpha.expand_as(real).cuda()
        interpolated_image = Variable(alpha * real + (1.0 - alpha) * fake, requires_grad=True).cuda()

        if (input_img is not None) and (material_img is not None):
            input_D = torch.cat((input_img, material_img, interpolated_image), 1)
        elif (input_img is not None) and (material_img is None):
            input_D = torch.cat((input_img, interpolated_image), 1)
        else:
            input_D = interpolated_image

        prediction = self.discriminator(input_D)

        gradients = torch.autograd.grad(outputs=prediction, inputs=input_D,
                                        grad_outputs=torch.ones(prediction.size()).cuda(),
                                        create_graph=True, retain_graph=True)[0]

        gradients = gradients.view(real.shape[0], -1)
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12).mean()

        return gradients_norm
