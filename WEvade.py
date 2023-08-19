import torch
import numpy as np

from utils import transform_image, project


def WEvade_W(original_image, Encoder, Decoder, criterion, args):

    # Embed the ground-truth watermark into the original image.
    original_image = original_image.cuda()
    groundtruth_watermark = torch.from_numpy(np.load('./watermark/watermark_coco.npy')).cuda()
    watermarked_image = Encoder(original_image, groundtruth_watermark)
    watermarked_image = transform_image(watermarked_image)
    watermarked_image_cloned = watermarked_image.clone()

    r = args.rb
    lr = args.alpha
    epsilon = args.epsilon
    success = False

    # WEvade_W_II target watermark selection.
    if args.WEvade_type == 'WEvade-W-II':
        random_watermark = np.random.choice([0, 1], (original_image.shape[0], args.watermark_length))
        target_watermark = torch.from_numpy(random_watermark).cuda().float()

    # WEvade_W_I target watermark selection.
    elif args.WEvade_type == 'WEvade-W-I':
        chosen_watermark = Decoder(watermarked_image).detach().cpu().numpy().round().clip(0, 1)
        chosen_watermark = 1 - chosen_watermark
        target_watermark = torch.from_numpy(chosen_watermark).cuda()

    for i in range(args.iteration):
        watermarked_image = watermarked_image.requires_grad_(True)
        min_value, max_value = torch.min(watermarked_image), torch.max(watermarked_image)
        decoded_watermark = Decoder(watermarked_image)

        # Post-process the watermarked image.
        loss = criterion(decoded_watermark, target_watermark)
        grads = torch.autograd.grad(loss, watermarked_image)
        with torch.no_grad():
            watermarked_image = watermarked_image - lr * grads[0]
            watermarked_image = torch.clamp(watermarked_image, min_value, max_value)

        # Projection.
        perturbation_norm = torch.norm(watermarked_image - watermarked_image_cloned, float('inf'))
        if perturbation_norm.cpu().detach().numpy() >= r:
            c = r / perturbation_norm
            watermarked_image = project(watermarked_image, watermarked_image_cloned, c)

        decoded_watermark = Decoder(watermarked_image)
        rounded_decoded_watermark = decoded_watermark.detach().cpu().numpy().round().clip(0, 1)
        bit_acc_target = 1 - np.sum(np.abs(rounded_decoded_watermark - target_watermark.cpu().numpy())) / (original_image.shape[0] * args.watermark_length)

        # Early Stopping.
        if perturbation_norm.cpu().detach().numpy() >= r:
            break

        if bit_acc_target >= 1 - epsilon:
            success = True
            break

    post_processed_watermarked_image = watermarked_image
    bound = torch.norm(post_processed_watermarked_image - watermarked_image_cloned, float('inf'))
    bit_acc_groundtruth = 1 - np.sum(np.abs(rounded_decoded_watermark - groundtruth_watermark.cpu().numpy())) / (original_image.shape[0] * args.watermark_length)

    return bit_acc_groundtruth, bound.item(), success


def WEvade_W_binary_search_r(original_image, Encoder, Decoder, criterion, args):

    original_image = original_image.cuda()
    groundtruth_watermark = torch.from_numpy(np.load('./watermark/watermark_coco.npy')).cuda()
    watermarked_image = Encoder(original_image, groundtruth_watermark)
    watermarked_image = transform_image(watermarked_image)
    watermarked_image_cloned = watermarked_image.clone()

    rb = args.rb
    ra = 0
    lr = args.alpha
    epsilon = args.epsilon

    if args.WEvade_type == 'WEvade-W-II':
        random_watermark = np.random.choice([0, 1], (original_image.shape[0], args.watermark_length))
        target_watermark = torch.from_numpy(random_watermark).cuda().float()

    elif args.WEvade_type == 'WEvade-W-I':
        chosen_watermark = Decoder(watermarked_image).detach().cpu().numpy().round().clip(0, 1)
        chosen_watermark = 1 - chosen_watermark
        target_watermark = torch.from_numpy(chosen_watermark).cuda()

    while (rb - ra >= 0.001):
        r = (rb + ra) / 2
        success = False

        for i in range(args.iteration):
            watermarked_image = watermarked_image.requires_grad_(True)
            min_value, max_value = torch.min(watermarked_image), torch.max(watermarked_image)
            decoded_watermark = Decoder(watermarked_image)

            loss = criterion(decoded_watermark, target_watermark)
            grads = torch.autograd.grad(loss, watermarked_image)
            with torch.no_grad():
                watermarked_image = watermarked_image - lr * grads[0]
                watermarked_image = torch.clamp(watermarked_image, min_value, max_value)

            perturbation_norm = torch.norm(watermarked_image - watermarked_image_cloned, float('inf'))
            if perturbation_norm.cpu().detach().numpy() >= r:
                c = r / perturbation_norm
                watermarked_image = project(watermarked_image, watermarked_image_cloned, c)

            decoded_watermark = Decoder(watermarked_image)
            rounded_decoded_watermark = decoded_watermark.detach().cpu().numpy().round().clip(0, 1)
            bit_acc_target = 1 - np.sum(np.abs(rounded_decoded_watermark - target_watermark.cpu().numpy())) / (original_image.shape[0] * args.watermark_length)

            if perturbation_norm.cpu().detach().numpy() >= r:
                break

            if bit_acc_target >= 1 - epsilon:
                success = True
                break

        # Binary search
        if success:
            rb = r
        else:
            ra = r

    post_processed_watermark_image = watermarked_image
    bound = torch.norm(post_processed_watermark_image - watermarked_image_cloned, float('inf'))
    bit_acc_groundtruth = 1 - np.sum(np.abs(rounded_decoded_watermark - groundtruth_watermark.cpu().numpy())) / (original_image.shape[0] * args.watermark_length)

    return bit_acc_groundtruth, bound.item(), success



