import torch
import torch.nn as nn
import argparse
import time
from tqdm import tqdm

from utils import get_data_loaders, transform_image, AverageMeter
from model.model import Model
from WEvade import WEvade_W, WEvade_W_binary_search_r


def main():

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    parser = argparse.ArgumentParser(description='WEvade-W Arguments.')
    parser.add_argument('--checkpoint', default='./ckpt/coco.pth', type=str, help='Model checkpoint file.')
    parser.add_argument('--dataset-folder', default='./dataset/coco/val', type=str, help='Dataset folder path.')
    parser.add_argument('--image-size', default=128, type=int, help='Size of the images (height and width).')
    parser.add_argument('--watermark-length', default=30, type=int, help='Number of bits in a watermark.')
    parser.add_argument('--tau', default=0.8, type=float, help='Detection threshold of the detector.')
    parser.add_argument('--iteration', default=5000, type=int, help='Max iteration in WEvdae-W.')
    parser.add_argument('--epsilon', default=0.01, type=float, help='Epsilon used in WEvdae-W.')
    parser.add_argument('--alpha', default=0.1, type=float, help='Learning rate used in WEvade-W.')
    parser.add_argument('--rb', default=2, type=float, help='Upper bound of perturbation.')
    parser.add_argument('--WEvade-type', default='WEvade-W-II', type=str, help='Using WEvade-W-I/II.')
    parser.add_argument('--detector-type', default='double-tailed', type=str, help='Using double-tailed/single-tailed detctor.')
    # In our algorithm, we use binary-search to obtain perturbation upper bound. But in the experiment, we find
    # binary-search actually has no significant effect on the perturbation results. And we reduce time cost if not
    # using binary-search.
    parser.add_argument('--binary-search', default=False, type=bool, help='Whether use binary-search to find perturbation.')

    args = parser.parse_args()

    # Load model.
    model = Model(args.image_size, args.watermark_length, device)
    checkpoint = torch.load(args.checkpoint)
    model.encoder.load_state_dict(checkpoint['enc-model'])
    model.decoder.load_state_dict(checkpoint['dec-model'])

    # Load dataset.
    data = get_data_loaders(args.image_size, args.dataset_folder)

    # WEvade.
    start_time = time.time()
    model.encoder.eval()
    model.decoder.eval()
    criterion = nn.MSELoss().cuda()

    Bit_acc = AverageMeter()
    Perturbation = AverageMeter()
    Evasion_rate = AverageMeter()
    Batch_time = AverageMeter()

    for image, _ in tqdm(data):
        image = transform_image(image)

        # Repeat up to three times to prevent randomly picked target watermark from being bad.
        for k in range(3):
            if args.binary_search == False:
                bit_acc, bound, success = WEvade_W(image, model.encoder, model.decoder, criterion, args)
            else:
                bit_acc, bound, success = WEvade_W_binary_search_r(image, model.encoder, model.decoder, criterion, args)
            if success:
                break

        # Detection for double-tailed/single-tailed detector.
        if args.detector_type == 'double-tailed':
            evasion = (1-args.tau <= bit_acc and bit_acc <= args.tau)
        elif args.detector_type == 'single-tailed':
            evasion = (bit_acc <= args.tau)

        bound = bound / 2   # [-1,1]->[0,1]
        Bit_acc.update(bit_acc, image.shape[0])
        Perturbation.update(bound, image.shape[0])
        Evasion_rate.update(evasion, image.shape[0])
        Batch_time.update(time.time() - start_time)
        start_time = time.time()

    print("Average Bit_acc=%.4f\t Average Perturbation=%.4f\t Evasion rate=%.2f\t Time=%.2f" % (Bit_acc.avg, Perturbation.avg, Evasion_rate.avg, Batch_time.sum))


if __name__ == '__main__':
    main()