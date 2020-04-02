import torch
import torch.nn as nn

import HelperFunctions as hf


class Generator(nn.Module):
    def __init__(self, num_in_channels):
        super(Generator, self).__init__()

        self.generator = {
            'kernel_size': 4,
            'stride': 2,
            'padding': 1,
            'num_in_channels': num_in_channels, #6, #3,   #Original is 3, new experiment with 6 3 input + 3 material
            'num_out_channels': 3,
            'num_blocks': 16,
            'max_encoder_in_channels': 512,
            'max_encoder_out_channels': 512,
            'use_dropout': True
        }

        blocks = self.__build_generator()

        self.encoder1 = nn.Sequential(*blocks[0])
        self.encoder2 = nn.Sequential(*blocks[1])
        self.encoder3 = nn.Sequential(*blocks[2])
        self.encoder4 = nn.Sequential(*blocks[3])
        self.encoder5 = nn.Sequential(*blocks[4])
        self.encoder6 = nn.Sequential(*blocks[5])
        self.encoder7 = nn.Sequential(*blocks[6])
        self.encoder8 = nn.Sequential(*blocks[7])

        self.decoder1 = nn.Sequential(*blocks[8])
        self.decoder2 = nn.Sequential(*blocks[9])
        self.decoder3 = nn.Sequential(*blocks[10])
        self.decoder4 = nn.Sequential(*blocks[11])
        self.decoder5 = nn.Sequential(*blocks[12])
        self.decoder6 = nn.Sequential(*blocks[13])
        self.decoder7 = nn.Sequential(*blocks[14])
        self.decoder8 = nn.Sequential(*blocks[15])

        self.__initialize_generator()

    def set_requires_grad(self, requires_grad=False):
        for param in self.network.parameters():
            param.requires_grad = requires_grad

    def forward(self, x):
        """
        This method performs the forward operation of the generator
        :param x: input
        :return: output of the last layer
        """

        out_encoder1 = self.encoder1.forward(x)
        out_encoder2 = self.encoder2.forward(out_encoder1)
        out_encoder3 = self.encoder3.forward(out_encoder2)
        out_encoder4 = self.encoder4.forward(out_encoder3)
        out_encoder5 = self.encoder5.forward(out_encoder4)
        out_encoder6 = self.encoder6.forward(out_encoder5)
        out_encoder7 = self.encoder7.forward(out_encoder6)
        out_encoder8 = self.encoder8.forward(out_encoder7)

        out_decoder1 = self.decoder1.forward(out_encoder8)
        out_decoder2 = self.decoder2.forward(torch.cat((out_decoder1, out_encoder7), dim=1))
        out_decoder3 = self.decoder3.forward(torch.cat((out_decoder2, out_encoder6), dim=1))
        out_decoder4 = self.decoder4.forward(torch.cat((out_decoder3, out_encoder5), dim=1))
        out_decoder5 = self.decoder5.forward(torch.cat((out_decoder4, out_encoder4), dim=1))
        out_decoder6 = self.decoder6.forward(torch.cat((out_decoder5, out_encoder3), dim=1))
        out_decoder7 = self.decoder7.forward(torch.cat((out_decoder6, out_encoder2), dim=1))
        out_decoder8 = self.decoder8.forward(torch.cat((out_decoder7, out_encoder1), dim=1))

        return out_decoder8

    def __build_generator(self):
        """
        A private method that builds the generator network for pix2pix Image Translation.
        The blocks are stored in the discriminator dictionary as a list.
        :return: None
        """
        n_blocks = self.generator['num_blocks']
        blocks = [hf.create_block(hf.BlockType.ENCODER,
                                  self.generator['num_in_channels'],
                                  64,
                                  self.generator['kernel_size'],
                                  self.generator['stride'],
                                  self.generator['padding'],
                                  hf.NormType.NONE,
                                  hf.ActivationType.LEAKY_RELU,
                                  self.generator['use_dropout'])]

        for i in range(1, n_blocks // 2):
            in_channels = min(32 * (2 ** i), self.generator['max_encoder_in_channels'])
            out_channels = min(64 * (2 ** i), self.generator['max_encoder_out_channels'])

            blocks.append(hf.create_block(hf.BlockType.ENCODER,
                                          in_channels,
                                          out_channels,
                                          self.generator['kernel_size'],
                                          self.generator['stride'],
                                          self.generator['padding'],
                                          hf.NormType.INSTANCE_NORM,
                                          hf.ActivationType.LEAKY_RELU,
                                          self.generator['use_dropout']))

        blocks.append(hf.create_block(hf.BlockType.DECODER,
                                      self.generator['max_encoder_in_channels'],
                                      self.generator['max_encoder_in_channels'],
                                      self.generator['kernel_size'],
                                      self.generator['stride'],
                                      self.generator['padding'],
                                      hf.NormType.NONE,
                                      hf.ActivationType.LEAKY_RELU,
                                      self.generator['use_dropout']))

        for i in range(n_blocks // 2 - 1, 1, -1):
            in_channels = min(64 * (2 ** i), 2 * self.generator['max_encoder_out_channels'])
            out_channels = min(16 * (2 ** i), self.generator['max_encoder_in_channels'])

            blocks.append(hf.create_block(hf.BlockType.DECODER,
                                          in_channels,
                                          out_channels,
                                          self.generator['kernel_size'],
                                          self.generator['stride'],
                                          self.generator['padding'],
                                          hf.NormType.INSTANCE_NORM,
                                          hf.ActivationType.LEAKY_RELU,
                                          self.generator['use_dropout']))

        blocks.append(hf.create_block(hf.BlockType.DECODER,
                                      128,
                                      self.generator['num_out_channels'],
                                      self.generator['kernel_size'],
                                      self.generator['stride'],
                                      self.generator['padding'],
                                      hf.NormType.INSTANCE_NORM,
                                      hf.ActivationType.TANH,
                                      False))

        return blocks

    def __initialize_generator(self):
        self.encoder1.apply(hf.initialize_weights)
        self.encoder2.apply(hf.initialize_weights)
        self.encoder3.apply(hf.initialize_weights)
        self.encoder4.apply(hf.initialize_weights)
        self.encoder5.apply(hf.initialize_weights)
        self.encoder6.apply(hf.initialize_weights)
        self.encoder7.apply(hf.initialize_weights)
        self.encoder8.apply(hf.initialize_weights)

        self.decoder1.apply(hf.initialize_weights)
        self.decoder2.apply(hf.initialize_weights)
        self.decoder3.apply(hf.initialize_weights)
        self.decoder4.apply(hf.initialize_weights)
        self.decoder5.apply(hf.initialize_weights)
        self.decoder6.apply(hf.initialize_weights)
        self.decoder7.apply(hf.initialize_weights)
        self.decoder8.apply(hf.initialize_weights)