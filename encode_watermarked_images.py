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

    parser = argparse.ArgumentParser(description='Encode watermarked images')
    parser.add_argument('--checkpoint', default='./ckpt/coco.pth', type=str, help='Model checkpoint file.')
    parser.add_argument('--dataset-folder', default='./dataset/coco/val', type=str, help='Dataset folder path.')
    parser.add_argument('--image-size', default=128, type=int, help='Size of the images (height and width).')
    parser.add_argument('--watermark-length', default=30, type=int, help='Number of bits in a watermark.')

    parser.add_argument('--exp', default="COCO", type=str, help='where to save watermarked images')
    parser.add_argument('--seed', default=2023, type=int)
    parser.add_argument('--num-images', default=100, type=int, help='number of images to embed watermark')
    args = parser.parse_args()
    exp = args.exp
    setup_seed(args.seed)   

    ### Load model
    model = Model(args.image_size, args.watermark_length, device)
    checkpoint = torch.load(args.checkpoint)
    model.encoder.load_state_dict(checkpoint['enc-model'])
    model.decoder.load_state_dict(checkpoint['dec-model'])

    ### Load dataset
    val_data = utils.get_data_loaders(args.image_size, args.dataset_folder)

    ### Load watermark
    msg_dir = './watermark/'
    msg_path = os.path.join(msg_dir, 'watermark_coco.npy')
    if os.path.exists(msg_path):
        msg = np.load(msg_path)
    else:
        print("Generate watermark.")
        msg = np.random.choice([0, 1], (1, args.watermark_length))
        np.save(msg_path, msg)
    msg = torch.Tensor(msg)
    print("Finish loading model, dataset and watermark.")



    ### Initialize result directory
    result_dir = './watermarked_images/{}/'.format(exp)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    watermarked_img_dir = os.path.join(result_dir, 'watermarked')
    if not os.path.exists(watermarked_img_dir):
        os.makedirs(watermarked_img_dir)
    original_img_dir = os.path.join(result_dir, 'clean')
    if not os.path.exists(original_img_dir):
        os.makedirs(original_img_dir)

    ### Save watermarked images in both RGB images (i.e., integer numbers in [0,255]) and the numpy array (i.e., arbitrary float numbers)   
    num_images = min(len(val_data.dataset), args.num_images)
    image_array = np.zeros((num_images, 3, args.image_size, args.image_size)).astype(np.float32)



    ### Encoding watermark
    image_idx = 0
    model.encoder.eval()
    model.decoder.eval()
    for image_batch, _ in val_data:
        image_batch = image_batch.to(device)
        msg_batch = msg.repeat(1,1).to(device)
        encoded_image_batch = model.encoder(image_batch, msg_batch)
        encoded_image_batch = encoded_image_batch.detach().cpu()
        
        # use the numpy array (i.e., arbitrary float numbers)
        image_array[image_idx:image_idx+1] = np.array(encoded_image_batch).copy()   

        # option 1:
        # save clean and watermarked images in RGB images (i.e., integer numbers in [0,255])
        # clean images
        filename = os.path.join(original_img_dir, '{}.png'.format(image_idx))
        utils.save_image_from_tensor(image_batch[0], filename)
        # watermarked images
        filename = os.path.join(watermarked_img_dir, '{}.png'.format(image_idx))
        utils.save_image_from_tensor(encoded_image_batch[0], filename)

        image_idx += 1
        if image_idx==num_images:
            break

    # option 2: 
    # save watermarked images in the numpy array (i.e., arbitrary float numbers)
    filename = os.path.join(result_dir, 'image_array.npy')
    np.save(filename, image_array)
    print("Save watermarked images in RGB images and the numpy array at {}".format(result_dir))

if __name__ == '__main__':
    main()