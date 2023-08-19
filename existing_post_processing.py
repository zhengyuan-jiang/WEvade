import torch
import numpy as np
import argparse
import time
from tqdm import tqdm

from utils import get_data_loaders, transform_image, AverageMeter
from model.model import Model
from noise_layers.diff_jpeg import DiffJPEG
from noise_layers.gaussian import Gaussian
from noise_layers.gaussian_blur import GaussianBlur
from noise_layers.brightness import Brightness


def post_process(original_image, Encoder, Decoder, method, args):

    # Embed the ground-truth watermark into the original image.
    original_image = original_image.cuda()
    groundtruth_watermark = torch.from_numpy(np.load('./watermark/watermark_coco.npy')).cuda()
    watermarked_image = Encoder(original_image, groundtruth_watermark)
    watermarked_image = transform_image(watermarked_image)
    watermarked_image_cloned = watermarked_image.clone()

    if method == 'jpeg':
        noise_layer = DiffJPEG(args.Q)
        watermarked_image = noise_layer(watermarked_image)
    elif method == 'gaussian':
        noise_layer = Gaussian(args.sigma1)
        watermarked_image = noise_layer(watermarked_image)
    elif method == 'gaussianblur':
        noise_layer = GaussianBlur(args.sigma2)
        watermarked_image = noise_layer(watermarked_image)
    elif method == 'brightness':
        noise_layer = Brightness(args.a)
        watermarked_image = noise_layer(watermarked_image)

    post_processed_watermarked_image = watermarked_image
    decoded_watermark = Decoder(post_processed_watermarked_image)
    rounded_decoded_watermark = decoded_watermark.detach().cpu().numpy().round().clip(0, 1)
    bound = torch.norm(post_processed_watermarked_image - watermarked_image_cloned, float('inf'))
    bit_acc_groundtruth = 1 - np.sum(np.abs(rounded_decoded_watermark - groundtruth_watermark.cpu().numpy())) / (original_image.shape[0] * args.watermark_length)

    return bit_acc_groundtruth, bound.item()


def main():

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    parser = argparse.ArgumentParser(description='Existing post-processing methods Arguments.')
    parser.add_argument('--checkpoint', default='./ckpt/coco.pth', type=str, help='Model checkpoint file.')
    parser.add_argument('--dataset-folder', default='./dataset/coco/val', type=str, help='Dataset folder path.')
    parser.add_argument('--image-size', default=128, type=int, help='Size of the images (height and width).')
    parser.add_argument('--watermark-length', default=30, type=int, help='Number of bits in a watermark.')
    parser.add_argument('--tau', default=0.8, type=float, help='Detection threshold of the detector.')
    parser.add_argument('--Q', default=50, type=int, help='Parameter Q for JPEGCompression.')
    parser.add_argument('--sigma1', default=0.1, type=float, help='Parameter \sigma for Gaussian noise.')
    parser.add_argument('--sigma2', default=0.5, type=float, help='Parameter \sigma for Gaussian blur.')
    parser.add_argument('--a', default=1.5, type=float, help='Parameter a for Brightness/Contrast.')
    parser.add_argument('--detector-type', default='double-tailed', type=str, help='Using double-tailed/single-tailed detctor.')

    args = parser.parse_args()

    # Load model.
    model = Model(args.image_size, args.watermark_length, device)
    checkpoint = torch.load(args.checkpoint)
    model.encoder.load_state_dict(checkpoint['enc-model'])
    model.decoder.load_state_dict(checkpoint['dec-model'])

    # Load dataset.
    data = get_data_loaders(args.image_size, args.dataset_folder)

    # Existing post-processing methods.
    start_time = time.time()
    model.encoder.eval()
    model.decoder.eval()

    Bit_acc = AverageMeter()
    Perturbation = AverageMeter()
    Evasion_rate = AverageMeter()
    Batch_time = AverageMeter()

    method_list = ['jpeg', 'gaussian', 'gaussianblur', 'brightness']
    for method in method_list:
        print('method:', method)

        for image, _ in tqdm(data):
            image = transform_image(image)

            bit_acc, bound= post_process(image, model.encoder, model.decoder, method, args)

            # Detection for double-tailed/single-tailed detector.
            if args.detector_type == 'double-tailed':
                evasion = (1-args.tau <= bit_acc and bit_acc <= args.tau)
            elif args.detector_type == 'double-tailed':
                evasion = (bit_acc <= args.tau)

            bound = bound / 2  # [-1,1]->[0,1]
            Bit_acc.update(bit_acc, image.shape[0])
            Perturbation.update(bound, image.shape[0])
            Evasion_rate.update(evasion, image.shape[0])
            Batch_time.update(time.time() - start_time)
            start_time = time.time()

        print("Average Bit_acc=%.4f\t Average Perturbation=%.4f\t Evasion rate=%.2f\t Time=%.2f" % (Bit_acc.avg, Perturbation.avg, Evasion_rate.avg, Batch_time.sum))


if __name__ == '__main__':
    main()