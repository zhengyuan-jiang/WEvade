import torch
from torchvision import datasets, transforms
from PIL import Image


def save_image_from_tensor(tensor, file_path):
    # Save a single image from torch tensor
    # refer to https://pytorch.org/vision/stable/_modules/torchvision/utils.html
    tensor = (tensor + 1) / 2   # for HiDDeN watermarking method only
    tensor = tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1,2,0).to("cpu", torch.uint8).numpy()
    im = Image.fromarray(tensor)
    im.save(file_path)


def get_data_loaders(image_size, dataset_folder):
    # Get torch data loaders. The data loaders take a crop of the image, transform it into tensor, and normalize it.
    data_transforms = transforms.Compose([
            transforms.CenterCrop((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    dataset_images = datasets.ImageFolder(dataset_folder, data_transforms)
    dataset_loader = torch.utils.data.DataLoader(dataset_images, batch_size=1, shuffle=False, num_workers=4)

    return dataset_loader


def transform_image(image):
    # For HiDDeN watermarking method, image pixel value range should be [-1, 1]. Transform an image into [-1, 1] range.
    cloned_encoded_images = (image + 1) / 2  # for HiDDeN watermarking method only
    cloned_encoded_images = cloned_encoded_images.mul(255).clamp_(0, 255)

    cloned_encoded_images = cloned_encoded_images / 255
    cloned_encoded_images = cloned_encoded_images * 2 - 1  # for HiDDeN watermarking method only
    image = cloned_encoded_images.cuda()

    return image


def project(param_data, backup, epsilon):
    # If the perturbation exceeds the upper bound, project it back.
    r = param_data - backup
    r = epsilon * r

    return backup + r


class AverageMeter(object):
    # Computes and stores the average and current value.
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count