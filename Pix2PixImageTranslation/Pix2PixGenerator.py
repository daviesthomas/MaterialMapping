import torch
import torch.nn as nn

import HelperFunctions as hf


class Generator(nn.Module):
    def __init__(self, num_in_channels, num_out_channels, spectral_norm, material_mapping):
        super(Generator, self).__init__()

        self.generator = {
            'kernel_size': 4,
            'stride': 2,
            'padding': 1,
            'num_in_channels': num_in_channels,
            'num_out_channels': num_out_channels,
            'num_blocks': 16,
            'max_encoder_in_channels': 512,
            'max_encoder_out_channels': 512,
            'use_dropout': True,
            'material_mapping': material_mapping,
            'spectral_norm': spectral_norm
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

        if self.generator['material_mapping']:
            self.material_encoder1 = nn.Sequential(*blocks[16])
            self.material_encoder2 = nn.Sequential(*blocks[17])
            self.material_encoder3 = nn.Sequential(*blocks[18])
            self.material_encoder4 = nn.Sequential(*blocks[19])
            self.material_encoder5 = nn.Sequential(*blocks[20])
            self.material_encoder6 = nn.Sequential(*blocks[21])
            self.material_encoder7 = nn.Sequential(*blocks[22])
            self.material_encoder8 = nn.Sequential(*blocks[23])

        self.__initialize_generator()

    def set_requires_grad(self, requires_grad=False):
        for param in self.encoder1.parameters():
            param.requires_grad = requires_grad
        for param in self.encoder2.parameters():
            param.requires_grad = requires_grad
        for param in self.encoder3.parameters():
            param.requires_grad = requires_grad
        for param in self.encoder4.parameters():
            param.requires_grad = requires_grad
        for param in self.encoder5.parameters():
            param.requires_grad = requires_grad
        for param in self.encoder6.parameters():
            param.requires_grad = requires_grad
        for param in self.encoder7.parameters():
            param.requires_grad = requires_grad
        for param in self.encoder8.parameters():
            param.requires_grad = requires_grad
        for param in self.decoder1.parameters():
            param.requires_grad = requires_grad
        for param in self.decoder2.parameters():
            param.requires_grad = requires_grad
        for param in self.decoder3.parameters():
            param.requires_grad = requires_grad
        for param in self.decoder4.parameters():
            param.requires_grad = requires_grad
        for param in self.decoder5.parameters():
            param.requires_grad = requires_grad
        for param in self.decoder6.parameters():
            param.requires_grad = requires_grad
        for param in self.decoder7.parameters():
            param.requires_grad = requires_grad
        for param in self.decoder8.parameters():
            param.requires_grad = requires_grad

        if self.generator['material_mapping']:
            for param in self.material_encoder1.parameters():
                param.requires_grad = requires_grad
            for param in self.material_encoder2.parameters():
                param.requires_grad = requires_grad
            for param in self.material_encoder3.parameters():
                param.requires_grad = requires_grad
            for param in self.material_encoder4.parameters():
                param.requires_grad = requires_grad
            for param in self.material_encoder5.parameters():
                param.requires_grad = requires_grad
            for param in self.material_encoder6.parameters():
                param.requires_grad = requires_grad
            for param in self.material_encoder7.parameters():
                param.requires_grad = requires_grad
            for param in self.material_encoder8.parameters():
                param.requires_grad = requires_grad

    def forward(self, x, m=None):
        """
        This method performs the forward operation of the generator
        :param x: input image
        :param m: input material (image)
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

        if self.generator['material_mapping']:
            out_material1 = self.material_encoder1(m)
            out_material2 = self.material_encoder2(out_material1)
            out_material3 = self.material_encoder3(out_material2)
            out_material4 = self.material_encoder4(out_material3)
            out_material5 = self.material_encoder5(out_material4)
            out_material6 = self.material_encoder6(out_material5)
            out_material7 = self.material_encoder7(out_material6)
            out_material8 = self.material_encoder8(out_material7)

            out_decoder1 = self.decoder1.forward(torch.cat((out_encoder8, out_material8), dim=1))
            out_decoder2 = self.decoder2.forward(torch.cat((out_decoder1, out_encoder7, out_material7), dim=1))
            out_decoder3 = self.decoder3.forward(torch.cat((out_decoder2, out_encoder6, out_material6), dim=1))
            out_decoder4 = self.decoder4.forward(torch.cat((out_decoder3, out_encoder5, out_material5), dim=1))
            out_decoder5 = self.decoder5.forward(torch.cat((out_decoder4, out_encoder4, out_material4), dim=1))
            out_decoder6 = self.decoder6.forward(torch.cat((out_decoder5, out_encoder3, out_material3), dim=1))
            out_decoder7 = self.decoder7.forward(torch.cat((out_decoder6, out_encoder2, out_material2), dim=1))
            out_decoder8 = self.decoder8.forward(torch.cat((out_decoder7, out_encoder1, out_material1), dim=1))
        else:
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
                                  self.generator['use_dropout'],
                                  self.generator['spectral_norm'])]

        for i in range(1, n_blocks // 2):
            in_channels = min(32 * (2 ** i), self.generator['max_encoder_in_channels'])
            out_channels = min(64 * (2 ** i), self.generator['max_encoder_out_channels'])

            blocks.append(hf.create_block(hf.BlockType.ENCODER,
                                          in_channels,
                                          out_channels,
                                          self.generator['kernel_size'],
                                          self.generator['stride'],
                                          self.generator['padding'],
                                          hf.NormType.BATCH_NORM,
                                          hf.ActivationType.LEAKY_RELU,
                                          self.generator['use_dropout'],
                                          self.generator['spectral_norm']))

        blocks.append(hf.create_block(hf.BlockType.DECODER,
                                      2 * self.generator['max_encoder_out_channels'] if self.generator['material_mapping'] else self.generator['max_encoder_out_channels'],
                                      self.generator['max_encoder_in_channels'],
                                      self.generator['kernel_size'],
                                      self.generator['stride'],
                                      self.generator['padding'],
                                      hf.NormType.NONE,
                                      hf.ActivationType.LEAKY_RELU,
                                      self.generator['use_dropout'],
                                      self.generator['spectral_norm']))

        for i in range(n_blocks // 2 - 1, 1, -1):
            in_channels = min(96 * (2 ** i), 3 * self.generator['max_encoder_out_channels']) if self.generator['material_mapping'] else min(64 * (2 ** i), 2 * self.generator['max_encoder_out_channels'])
            out_channels = min(16 * (2 ** i), self.generator['max_encoder_in_channels'])

            blocks.append(hf.create_block(hf.BlockType.DECODER,
                                          in_channels,
                                          out_channels,
                                          self.generator['kernel_size'],
                                          self.generator['stride'],
                                          self.generator['padding'],
                                          hf.NormType.BATCH_NORM,
                                          hf.ActivationType.LEAKY_RELU,
                                          self.generator['use_dropout'],
                                          self.generator['spectral_norm']))

        blocks.append(hf.create_block(hf.BlockType.DECODER,
                                      192 if self.generator['material_mapping'] else 128,
                                      self.generator['num_out_channels'],
                                      self.generator['kernel_size'],
                                      self.generator['stride'],
                                      self.generator['padding'],
                                      hf.NormType.BATCH_NORM,
                                      hf.ActivationType.TANH,
                                      False,
                                      self.generator['spectral_norm']))

        if self.generator['material_mapping']:
            blocks.append(hf.create_block(hf.BlockType.ENCODER,
                                          self.generator['num_in_channels'],
                                          64,
                                          self.generator['kernel_size'],
                                          self.generator['stride'],
                                          self.generator['padding'],
                                          hf.NormType.NONE,
                                          hf.ActivationType.LEAKY_RELU,
                                          self.generator['use_dropout'],
                                          self.generator['spectral_norm']))

            for i in range(1, n_blocks // 2):
                in_channels = min(32 * (2 ** i), self.generator['max_encoder_in_channels'])
                out_channels = min(64 * (2 ** i), self.generator['max_encoder_out_channels'])

                blocks.append(hf.create_block(hf.BlockType.ENCODER,
                                              in_channels,
                                              out_channels,
                                              self.generator['kernel_size'],
                                              self.generator['stride'],
                                              self.generator['padding'],
                                              hf.NormType.BATCH_NORM,
                                              hf.ActivationType.LEAKY_RELU,
                                              self.generator['use_dropout'],
                                              self.generator['spectral_norm']))

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

        if self.generator['material_mapping']:
            self.material_encoder1.apply(hf.initialize_weights)
            self.material_encoder2.apply(hf.initialize_weights)
            self.material_encoder3.apply(hf.initialize_weights)
            self.material_encoder3.apply(hf.initialize_weights)
            self.material_encoder4.apply(hf.initialize_weights)
            self.material_encoder5.apply(hf.initialize_weights)
            self.material_encoder6.apply(hf.initialize_weights)
            self.material_encoder7.apply(hf.initialize_weights)
