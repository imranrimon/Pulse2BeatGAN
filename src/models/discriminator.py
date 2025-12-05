# --------------
#  Discriminator
# --------------

import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, in_channels=1, kernel_size = [6, 6, 6, 6], stride = [4, 4, 4, 4], bias=False):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, ksize=8, stride=3, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv1d(in_filters, out_filters, ksize,
                                stride=stride, padding_mode='replicate')]
            if normalization:
                layers.append(nn.InstanceNorm1d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 128, ksize= kernel_size[0], stride= stride[0], normalization=False),
            *discriminator_block(128, 256, ksize= kernel_size[1], stride= stride[1]),
            *discriminator_block(256, 512, ksize= kernel_size[2], stride= stride[2]),
            nn.Conv1d(512, 1, kernel_size=kernel_size[3], stride=stride[3], bias=False, padding_mode='replicate')
        )

    def forward(self, signal_A, signal_B):
        # Concatenate signals and condition signals by channels to produce input
        signal_input = torch.cat((signal_A, signal_B), 1)
        return self.model(signal_input)