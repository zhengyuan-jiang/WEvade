# Modified from https://github.com/ando-khachatryan/HiDDeN
from model.encoder import Encoder
from model.decoder import Decoder


class Configuration:
    ### The network configuration.
    def __init__(self, H: int, W: int, watermark_length: int,
                 encoder_blocks: int, encoder_channels: int,
                 decoder_blocks: int, decoder_channels: int,
                 use_discriminator: bool,
                 use_vgg: bool,
                 discriminator_blocks: int, discriminator_channels: int,
                 decoder_loss: float,
                 encoder_loss: float,
                 adversarial_loss: float,
                 enable_fp16: bool = False):
        self.H = H
        self.W = W
        self.watermark_length = watermark_length
        self.encoder_blocks = encoder_blocks
        self.encoder_channels = encoder_channels
        self.use_discriminator = use_discriminator
        self.use_vgg = use_vgg
        self.decoder_blocks = decoder_blocks
        self.decoder_channels = decoder_channels
        self.discriminator_blocks = discriminator_blocks
        self.discriminator_channels = discriminator_channels
        self.decoder_loss = decoder_loss
        self.encoder_loss = encoder_loss
        self.adversarial_loss = adversarial_loss
        self.enable_fp16 = enable_fp16


class Model:
    ### The model architecture.
    def __init__(self, image_size, watermark_length, device):
        super(Model, self).__init__()
        configuration = Configuration(H=image_size, W=image_size,
                                            watermark_length=watermark_length,
                                            encoder_blocks=4, encoder_channels=64,
                                            decoder_blocks=7, decoder_channels=64,
                                            use_discriminator=True,
                                            use_vgg=False,
                                            discriminator_blocks=3, discriminator_channels=64,
                                            decoder_loss=1,
                                            encoder_loss=0.7,
                                            adversarial_loss=1e-3,
                                            enable_fp16=False
                                            )
        self.encoder = Encoder(configuration).to(device)
        self.decoder = Decoder(configuration).to(device)
