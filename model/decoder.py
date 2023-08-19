import torch.nn as nn
from model.conv_bn_relu import ConvBNRelu


class Decoder(nn.Module):
    # Receives an image and decodes the watermark. The input image may be watermarked or non-watermarked. Moreover,
    # the input image may have various kinds of noise applied to it, such as JpegCompression, Gaussian blur, and so on.
    # See Noise layers for more.
    def __init__(self, config):

        super(Decoder, self).__init__()
        self.channels = config.decoder_channels

        layers = [ConvBNRelu(3, self.channels)]
        for _ in range(config.decoder_blocks - 1):
            layers.append(ConvBNRelu(self.channels, self.channels))

        # layers.append(block_builder(self.channels, config.watermark_length))
        layers.append(ConvBNRelu(self.channels, config.watermark_length))

        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.layers = nn.Sequential(*layers)

        self.linear = nn.Linear(config.watermark_length, config.watermark_length)

    def forward(self, watermarked_image):
        decoded_watermark = self.layers(watermarked_image)
        # The output is of shape b x c x 1 x 1, and we want to squeeze out the last two dummy dimensions and make the
        # tensor of shape b x c. If we just call squeeze_() it will also squeeze the batch dimension when b=1.
        decoded_watermark.squeeze_(3).squeeze_(2)
        decoded_watermark = self.linear(decoded_watermark)

        return decoded_watermark
