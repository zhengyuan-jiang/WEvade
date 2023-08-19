import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import numpy as np
import math
from tqdm import tqdm
import random
import pickle
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

import utils
from model.model import Model
from skimage.metrics import structural_similarity as ssim
from noise_layers.diff_jpeg import DiffJPEG
from art.estimators.classification import PyTorchClassifier
from hop_skip_jump import HopSkipJump


# Classification layer of watermark detector   
class Class_Layer(nn.Module):
    def __init__(self, message, detector_type, th, device):
        super(Class_Layer, self).__init__()
        self.message = message.to(device)
        self.detector_type = detector_type
        self.th = th

    def forward(self, decoded_messages):
        decoded_messages = torch.round(decoded_messages)
        decoded_messages = torch.clamp(decoded_messages, min=0, max=1)
        groundtruth = self.message.repeat(decoded_messages.shape[0], 1)
        bit_acc = 1-torch.sum(torch.abs(decoded_messages-groundtruth), 1)/groundtruth.shape[1]
        if self.detector_type=='double-tailed':
            class_idx = torch.logical_or((bit_acc>self.th), (bit_acc<(1-self.th))).long()
        if self.detector_type=='single-tailed':
            class_idx = (bit_acc>self.th).long()
        return F.one_hot(class_idx, num_classes=2)



def get_watermark_detector(decoder, msg, detector_type, th, device):
    cls_layer = Class_Layer(msg, detector_type, th, device)
    detector = nn.Sequential(decoder, cls_layer).to(device)
    detector.eval()
    return detector



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False



# Note: images and labels are in numpy arrays
def evaluate(images, labels, detector):
    # evaluate the accuracy of watermark detector
    acc = 0
    evade_indices = []
    for i in range(len(images)):
        image = images[i:i+1]
        pred = detector.predict(image)
        pred = np.argmax(pred,-1)[0]
        acc += np.sum(pred==labels[i])
        if pred!=labels[i]:
            evade_indices.append(i)
    return acc, evade_indices



# Note: watermarked_images, labels and adv_images are in numpy arrays
def JPEG_initailization(watermarked_images, labels, detector, quality_ls, natural_adv=None, verbose=True):
    # JPEG initialization
    adv_images = watermarked_images.copy()
    num_images = len(watermarked_images)
    init_num_queries_ls = np.zeros((num_images)) # number of queries used for initialization
    flags = np.zeros((num_images))               # whether an adversarial example has been found
    if natural_adv is not None:
        flags[natural_adv] = 1

    for quality in tqdm(quality_ls): 
        jpeg_module = DiffJPEG(quality=quality).cuda()
        for k in range(len(adv_images)):
            if flags[k]==1: # pass
                continue
            init_num_queries_ls[k] += 1
            jpeg_image = torch.from_numpy(watermarked_images[k:k+1])
            jpeg_image_max = torch.max(jpeg_image)
            jpeg_image_min = torch.min(jpeg_image)
            jpeg_image = (jpeg_image-jpeg_image_min)/(jpeg_image_max-jpeg_image_min)
            jpeg_image = jpeg_module(jpeg_image.cuda()).detach().cpu()
            jpeg_image = jpeg_image*(jpeg_image_max-jpeg_image_min)+jpeg_image_min
            jpeg_image = jpeg_image.numpy()

            pred = detector.predict(jpeg_image)
            pred = np.argmax(pred,-1)[0]
            if pred!=labels[k:k+1]: # succeed
                adv_images[k:k+1] = jpeg_image
                flags[k]=1
        del jpeg_module
    print("Finish JPEG Initialization.")
    
    if verbose:
        print("Flags:", flags)

    return adv_images, init_num_queries_ls



# Note: watermarked_images, init_adv_images and best_adv are in numpy arrays
def WEvade_B_Q(args, watermarked_images, init_adv_images, detector, num_queries_ls, verbose=True):
    num_images = len(watermarked_images)
    norm = args.norm
    attack = HopSkipJump(classifier=detector, targeted=False, norm=norm, max_iter=0, max_eval=args.max_eval, init_eval=args.init_eval, batch_size=args.batch_size)
    
    total_num_queries = 0
    saved_num_queries_ls = num_queries_ls.copy()
    es_ls = np.zeros((num_images))    # a list of 'es' in Algorithm 3
    es_flags = np.zeros((num_images)) # whether the attack has been early stopped
    num_natural_adv = 0
    num_early_stop = 0
    num_regular_stop = 0

    adv_images = init_adv_images.copy()
    best_adv = init_adv_images
    best_norms = np.ones((num_images))*1e8

    ### Algorithm
    max_iterations = 1000 # a random large number
    for i in range(int(max_iterations/args.iter_step)):
        adv_images, num_queries_ls = attack.generate(x=watermarked_images, x_adv_init=adv_images, num_queries_ls=num_queries_ls, resume=True) # use resume to continue previous attack
        if verbose:
            print("Step: {}; Number of queries: {}".format((i * args.iter_step), num_queries_ls))

        # save the best results
        avg_error = 0
        for k in range(len(adv_images)):
            if norm == 'inf':
                error = np.max(np.abs(adv_images[k] - watermarked_images[k]))
            else:
                error = np.linalg.norm(adv_images[k] - watermarked_images[k])

            if es_flags[k]==0: # update if the attack has not been early stopped
                if error<best_norms[k]:
                    best_norms[k] = error
                    best_adv[k] = adv_images[k]
                    es_ls[k] = 0
                else:
                    es_ls[k]+=1
            avg_error += best_norms[k]
        avg_error = avg_error/2 # [-1,1]->[0,1]
        if verbose:
            print("Adversarial images at step {}.".format(i * args.iter_step))
            print("Average best error in l_{} norm: {}\n".format(norm, avg_error/len(adv_images)))

        # stopping criteria
        # natural_adv
        for k in range(len(adv_images)):
            if best_norms[k]==0 and es_flags[k]==0:
                es_flags[k] = 1
                total_num_queries += 0
                saved_num_queries_ls[k] = 0
                num_natural_adv += 1
        # regular_stop
        for k in range(len(adv_images)):
            if num_queries_ls[k]>=args.budget and es_flags[k]==0:
                es_flags[k] = 1
                total_num_queries += num_queries_ls[k]
                saved_num_queries_ls[k] = num_queries_ls[k]
                num_regular_stop+=1
        # early_stop
        for k in range(len(adv_images)):
            if es_ls[k]==args.ES and es_flags[k]==0:
                es_flags[k] = 1
                total_num_queries += num_queries_ls[k]
                saved_num_queries_ls[k] = num_queries_ls[k]
                num_early_stop += 1

        if np.sum(es_flags==0)==0:
            break
        attack.max_iter = args.iter_step

    assert np.sum(es_flags)==num_images
    assert num_natural_adv+num_regular_stop+num_early_stop==num_images
    assert np.sum(saved_num_queries_ls)==total_num_queries
    del attack

    if verbose:
        print("Number of queries used for each sample:")
        print(saved_num_queries_ls)

    return best_adv, saved_num_queries_ls



def draw_curve(results_dict, file_path, norm='inf'):
    # draw 'Detection Threshold' vs 'Average Perturbation' curve
    def params_init():
        matplotlib.rc('xtick', labelsize=28)
        matplotlib.rc('ytick', labelsize=28)
        matplotlib.rc('lines',markersize=10)
        font = {'family': 'serif',
                    'size': 28,
                    }
        matplotlib.rc('font', **font)
        matplotlib.rcParams['text.usetex'] = True
        matplotlib.rcParams['figure.figsize'] = 8,6
    params_init()
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.12,right=0.95,top=0.95, bottom=0.15)

    tau_ls = []
    pert_ls = []
    er_ls = []
    for tau, [pert, ER] in results_dict.items():
        tau_ls.append(tau)
        pert_ls.append(pert)
        er_ls.append(ER)
    print("Evading rate:", er_ls) ### always 1.0
    
    plt.plot(tau_ls, pert_ls, marker='v', label='WEvade-B-Q', markersize=7, linewidth=3)
    ax.set_ylim(1, 0.5)
    ax.set_xlabel(r"Detection Threshold $\tau$")
    ax.set_xticks(np.arange(0.5, 1.05, step=0.1))
    ax.invert_xaxis()
    ax.set_ylabel(r"Average Perturbation")
    if norm=='inf':
        y_top = 0.04
    if norm==2:
        y_top = 4
    ax.set_ylim(0, y_top)
    plt.legend(fontsize=30)
    plt.savefig(file_path)



def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    parser = argparse.ArgumentParser(description='Blax-box attack')

    # General settings
    parser.add_argument('--checkpoint', default='./ckpt/coco.pth', type=str, help='Model checkpoint file.')
    parser.add_argument('--dataset-folder', default='./dataset/coco/val', type=str, help='Dataset folder path.')
    parser.add_argument('--image-size', default=128, type=int, help='Size of the images (height and width).')
    parser.add_argument('--watermark-length', default=30, type=int, help='Number of bits in a watermark.')
    parser.add_argument('--detector-type', default='double-tailed', choices=['double-tailed','single-tailed'], type=str, help='Using double-tailed/single-tailed detctor.')
    parser.add_argument('--exp', default="COCO", type=str, help='Subfolder where to load the watermarked dataset and save attack results')
    parser.add_argument('--seed', default=10, type=int)

    # Attack settings
    parser.add_argument('--num-attack', default=100, type=int, help='number of images to attack')
    parser.add_argument('--budget', default=2000, type=int, help='query budget')
    parser.add_argument('--init-eval', default=5, type=int, help='hopskipjump parameters')
    parser.add_argument('--max-eval', default=1000, type=int, help='hopskipjump parameters')
    parser.add_argument('--iter-step', default=1, type=int, help='print interval')
    parser.add_argument('--ES', default=20, type=int, help='early stopping criterion')
    parser.add_argument('--norm', default='inf', choices=['2','inf'], help='norm metric') # We optimize different norm for Hopskipjump when using different norm as the metric following their original work
    parser.add_argument('--batch-size', default=256, type=int, help='batch size for hopskipjump')
    parser.add_argument('--verbose', default=True, type=bool, help='verbose mode')
    parser.add_argument('--save-image', default=True, type=bool, help='save adversarial images')
    parser.add_argument('--draw-curve', default=True, type=bool, help='draw result curves')
    args = parser.parse_args()   
    exp = args.exp
    if args.norm=='2':
        args.norm=2
    setup_seed(args.seed)



    ### Load model
    model = Model(args.image_size, args.watermark_length, device)
    checkpoint = torch.load(args.checkpoint)
    model.encoder.load_state_dict(checkpoint['enc-model'])
    model.decoder.load_state_dict(checkpoint['dec-model'])

    ### Load watermarked dataset (in numpy array)
    watermarked_dataset_dir = './watermarked_images/{}/'.format(exp)
    if not os.path.exists(watermarked_dataset_dir):
        assert "Watermarked dataset does not exists."

    # Load watermarked images in either RGB images or the number array
    # We experiment with RGB images by default because in practice, watermarked images are represented as RGB values rather than the numpy array   
    load_array = False
    if load_array:
        watermarked_images = np.load('{}/image_array.npy'.format(watermarked_dataset_dir)).astype(np.float32)
    else:
        watermarked_img_dir = os.path.join(watermarked_dataset_dir, 'watermarked')
        num_images = len(os.listdir(watermarked_img_dir))
        watermarked_images = np.zeros((num_images, 3, args.image_size, args.image_size)).astype(np.float32)

        for i in range(num_images):
            img_path = os.path.join(watermarked_img_dir, '{}.png'.format(i))
            im = Image.open(img_path)
            im = np.array(im).transpose((2, 0, 1))
            im = (im/255)*2-1
            watermarked_images[i] = im

    ### Load watermark
    msg_dir = './watermark/'
    msg_path = os.path.join(msg_dir, 'watermark_coco.npy')
    if not os.path.exists(msg_path):
        assert "Watermark does not exists."
    msg = np.load(msg_path)
    msg = torch.Tensor(msg)

    ### Load labels (in numpy array)
    labels = np.ones((len(watermarked_images)))
    print("Finish loading model, watermarked dataset, watermark and labels.\n")     



    ### Experimental Parameters
    num2attack = min(len(watermarked_images), args.num_attack)
    watermarked_images = watermarked_images[:num2attack]
    labels = labels[:num2attack]
    print("Number of images to attack: {}".format(num2attack))
    norm = args.norm
    print("Use l_{} norm as the metric.\n".format(norm))

    quality_ls = [99,90,70,50,30,10,5,3,2,1]
    # th_ls = [0.83]
    th_ls = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    verbose = args.verbose
    results_dict = {}
    results_dir = "./results/{}/".format(exp)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)



    ### Attack   
    for th in th_ls:
        print("############################## THRESHOLD {} ##############################".format(th))
        

        ### Initialize watermark detector
        detector = get_watermark_detector(model.decoder, msg, args.detector_type, th, device)
        if load_array: # arbitrary float numbers
            detector = PyTorchClassifier(
                model=detector,
                input_shape=(3, args.image_size, args.image_size),
                nb_classes=2,
                use_amp=False,
                channels_first=True,
                loss=None,
            )
        else:
            detector = PyTorchClassifier(
                model=detector,
                clip_values=(-1.0, 1.0),
                input_shape=(3, args.image_size, args.image_size),
                nb_classes=2,
                use_amp=False,
                channels_first=True,
                loss=None,
            )


        ### Evaluate clean images
        acc, natural_adv = evaluate(watermarked_images, labels, detector)
        print('Clean accuracy under threshold {}: {}'.format(th, acc/len(watermarked_images)))
        print('Natural adversarial examples:', natural_adv, len(natural_adv))
        
        ### JPEG initialization
        init_adv_images, num_queries_ls = JPEG_initailization(watermarked_images, labels, detector, quality_ls, natural_adv=None, verbose=verbose)

        ### Evaluate adversarial images (before black-box attack)
        acc, _ = evaluate(init_adv_images, labels, detector)
        print('Initial adv accuracy under threshold {}: {}\n'.format(th, acc/len(watermarked_images)))


        ### Run WEvade-B-Q
        best_adv_images, saved_num_queries_ls = WEvade_B_Q(args, watermarked_images, init_adv_images, detector, num_queries_ls, verbose=verbose)
        print("Average number of queries: {}\n".format(np.sum(saved_num_queries_ls)/len(best_adv_images)))
        

        ### Save result images
        if args.save_image:
            result_img_dir = os.path.join(results_dir, 'l_{}_adv_images/{}'.format(norm,th))
            print(result_img_dir)
            if not os.path.exists(result_img_dir):
                os.makedirs(result_img_dir)
            for k in range(len(best_adv_images)):
                filename = os.path.join(result_img_dir, '{}.png'.format(k))
                utils.save_image_from_tensor(torch.from_numpy(best_adv_images[k]), filename)

        ### Evaluate adversarial images (after black-box attack) and evading rate
        acc, _ = evaluate(best_adv_images, labels, detector)
        ER = 1-acc
        print('Adv accuracy under threshold {}: {}'.format(th, acc/len(best_adv_images)))
        print('Evading rate under threshold {}: {}'.format(th, ER))

        ### Evaluate bit accuracy before and after attack
        acc = 0
        bit_acc_before_attack = []
        bit_acc_after_attack = []
        for i in range(len(best_adv_images)):
            message = msg.repeat(1,1)

            decoded_message = model.decoder(torch.from_numpy(watermarked_images[i:i+1]).to(device))
            decoded_message = decoded_message.detach().cpu().numpy().round().clip(0, 1)
            bit_acc = 1 - np.sum(np.abs(decoded_message - message.numpy())) / (1 * message.shape[1])
            bit_acc_before_attack.append(bit_acc)

            decoded_message = model.decoder(torch.from_numpy(best_adv_images[i:i+1]).to(device))
            decoded_message = decoded_message.detach().cpu().numpy().round().clip(0, 1)
            bit_acc = 1 - np.sum(np.abs(decoded_message - message.numpy())) / (1 * message.shape[1])
            bit_acc_after_attack.append(bit_acc)
        print("Average bit-accuracy before attack under threshold {}: {}".format(th, np.mean(bit_acc_before_attack)))
        print("Average bit-accuracy after attack under threshold {}: {}".format(th, np.mean(bit_acc_after_attack)))

        ### Get final perturbations
        avg_error = 0
        for k in range(len(best_adv_images)):
            if norm == 'inf':
                error = np.max(np.abs(best_adv_images[k] - watermarked_images[k]))
            else:
                error = np.linalg.norm(best_adv_images[k] - watermarked_images[k])
            avg_error += error
        avg_error = avg_error/2 # [-1,1]->[0,1]
        print("Average best error in l_{} norm: {}\n".format(norm, avg_error/len(best_adv_images)))


        results_dict[th] = [avg_error/len(best_adv_images), ER]
        # break

    print("Finish Attack.")



    ### Save and draw result_dict   
    results_path = os.path.join(results_dir, "l_{}.pkl".format(norm))
    with open(results_path, 'wb') as f:
        pickle.dump(results_dict, f)
    with open(results_path, 'rb') as f:
        results_dict = pickle.load(f)
    if args.draw_curve==True:
        print("Results:", results_dict)
        draw_curve(results_dict, os.path.join(results_dir, "l_{}.png".format(norm)), norm=norm)

if __name__ == '__main__':
    main()