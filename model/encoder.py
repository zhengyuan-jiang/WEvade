import torch
import torch.nn as nn
from model.conv_bn_relu import ConvBNRelu


class Encoder(nn.Module):
    ### Embed a watermark into the original image and output the watermarked image.
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.H = config.H
        self.W = config.W
        self.conv_channels = config.encoder_channels
        self.num_blocks = config.encoder_blocks

        layers = [ConvBNRelu(3, self.conv_channels)]

        for _ in range(config.encoder_blocks-1):
            layer = ConvBNRelu(self.conv_channels, self.conv_channels)
            layers.append(layer)

        self.conv_layers = nn.Sequential(*layers)
        self.after_concat_layer = ConvBNRelu(self.conv_channels + 3 + config.watermark_length, self.conv_channels)
        self.final_layer = nn.Conv2d(self.conv_channels, 3, kernel_size=1)

    def forward(self, original_image, watermark):
        # First, add two dummy dimensions in the end of the watermark.
        # This is required for the .expand to work correctly.
        expanded_watermark = watermark.unsqueeze(-1)
        expanded_watermark.unsqueeze_(-1)
        expanded_watermark = expanded_watermark.expand(-1, -1, self.H, self.W)
        encoded_image = self.conv_layers(original_image)

        # Concatenate expanded watermark and the original image.
        concat = torch.cat([expanded_watermark, encoded_image, original_image], dim=1)
        watermarked_image = self.after_concat_layer(concat)
        watermarked_image = self.final_layer(watermarked_image)

        return watermarked_image
