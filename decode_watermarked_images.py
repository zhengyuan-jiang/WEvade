import torch
import torch.nn
import torchvision
import argparse
import os
import numpy as np
import math
from PIL import Image
import random

import utils
from model.model import Model
from skimage.metrics import structural_similarity as ssim


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False



def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    parser = argparse.ArgumentParser(description='Decode watermarked images')
    parser.add_argument('--checkpoint', default='./ckpt/coco.pth', type=str, help='Model checkpoint file.')
    parser.add_argument('--dataset-folder', default='./dataset/coco/val', type=str, help='Dataset folder path.')
    parser.add_argument('--image-size', default=128, type=int, help='Size of the images (height and width).')
    parser.add_argument('--watermark-length', default=30, type=int, help='Number of bits in a watermark.')

    parser.add_argument('--exp', default="COCO", type=str, help='where to load watermarked images')
    parser.add_argument('--seed', default=2023, type=int)
    args = parser.parse_args()
    exp = args.exp
    setup_seed(args.seed)   



    ### Load model.
    model = Model(args.image_size, args.watermark_length, device)
    checkpoint = torch.load(args.checkpoint)
    model.encoder.load_state_dict(checkpoint['enc-model'])
    model.decoder.load_state_dict(checkpoint['dec-model'])

    ### Load watermarked dataset
    watermarked_dataset_dir = './watermarked_images/{}/'.format(exp)
    if not os.path.exists(watermarked_dataset_dir):
        assert "Watermarked dataset does not exists."

    load_array = False # load watermarked images in either RGB images or the number array
    if load_array:
        watermarked_images = np.load('{}/image_array.npy'.format(watermarked_dataset_dir))
        watermarked_images = torch.from_numpy(watermarked_images)
    else:
        watermarked_img_dir = os.path.join(watermarked_dataset_dir, 'watermarked') # load watermarked images: around 1.0 bit-accuracy
        # watermarked_img_dir = os.path.join(watermarked_dataset_dir, 'clean')     # load clean images for evaluation purpose: around 0.5 bit-accuracy
        num_images = len(os.listdir(watermarked_img_dir))
        watermarked_images = torch.zeros((num_images, 3, args.image_size, args.image_size))

        for i in range(num_images):
            img_path = os.path.join(watermarked_img_dir, '{}.png'.format(i))
            im = Image.open(img_path)
            im = torch.from_numpy(np.array(im)).permute(2, 0, 1)
            im = (im/255)*2-1
            watermarked_images[i] = im
    
    ### Load watermark
    msg_dir = './watermark/'
    msg_path = os.path.join(msg_dir, 'watermark_coco.npy')
    if not os.path.exists(msg_path):
        assert "Watermark does not exists."
    msg = np.load(msg_path)
    msg = torch.Tensor(msg)
    print("Finish loading model, watermarked dataset and watermark.")



    ### Decoding watermark
    bit_acc = 0
    model.encoder.eval()
    model.decoder.eval()
    for i in range(len(watermarked_images)):
        encoded_image_batch = watermarked_images[i:i+1].to(device)
        msg_batch = msg.repeat(1,1)

        decoded_msg_batch = model.decoder(encoded_image_batch)
        decoded_msg_batch = decoded_msg_batch.detach().cpu().numpy().round().clip(0, 1)
        this_bit_acc = 1 - np.sum(np.abs(decoded_msg_batch - msg_batch.numpy())) / (1 * msg_batch.shape[1])
        bit_acc += this_bit_acc
    print('Average bit_acc:', bit_acc/len(watermarked_images))

if __name__ == '__main__':
    main()
