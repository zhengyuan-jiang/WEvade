# Official Implementation of WEvade

This code is the official implementation of our paper: Evading Watermark based Detection of AI-Generated Content [Paper](https://arxiv.org/abs/2305.03807).

## Preparation

1. Clone this repo from the GitHub.
	
		git clone https://github.com/zhengyuan-jiang/WEvade.git

2. Setup environment.

- Install the [pytorch](https://pytorch.org/). The latest codes are tested on Ubuntu 16.04, PyTorch 1.x.x and Python 3.x:

- Install the [adversarial-robustness-toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox):

      pip install adversarial-robustness-toolbox

## Run WEvade-W (white-box attack)

Run WEvade-W-II:

```
## Standard training
python3 main.py --checkpoint './ckpt/coco.pth' --dataset-folder './dataset/coco/val'

## Adversarial training
python3 main.py --checkpoint './ckpt/coco_adv_train.pth' --dataset-folder './dataset/coco/val'
```

Run other variants:

```
## WEvade-W-I
python3 main.py --checkpoint './ckpt/coco.pth' --dataset-folder './dataset/coco/val' --WEvade-type 'WEvade-W-I'

## Single-tailed detector
python3 main.py --checkpoint './ckpt/coco.pth' --dataset-folder './dataset/coco/val' --detector-type 'single-tailed'

## Binary search
python3 main.py --checkpoint './ckpt/coco.pth' --dataset-folder './dataset/coco/val' --binary-search True
```

Run existing post-processing methods:

```
## Standard training
python3 existing_post_processing.py --checkpoint './ckpt/coco.pth' --dataset-folder './dataset/coco/val'

## Adversarial training
python3 existing_post_processing.py --checkpoint './ckpt/coco_adv_train.pth' --dataset-folder './dataset/coco/val'
```

## Run WEvade-B-Q (query based black-box attack)

1. Encode watermark and save watermarked images (watermark is generated using the random seed):

		## Standard training
		python3 encode_watermarked_images.py --checkpoint './ckpt/coco.pth' --dataset-folder './dataset/coco/val' --exp 'COCO'
	
		## Adversarial training
		python3 encode_watermarked_images.py --checkpoint './ckpt/coco_adv_train.pth' --dataset-folder './dataset/coco/val' --exp 'COCO-ADV'

1. (Optional) Decode watermarked images for evaluation:

		## Standard training
		python3 decode_watermarked_images.py --checkpoint './ckpt/coco.pth' --dataset-folder './dataset/coco/val' --exp 'COCO'
	
		## Adversarial training
		python3 decode_watermarked_images.py --checkpoint './ckpt/coco_adv_train.pth' --dataset-folder './dataset/coco/val' --exp 'COCO-ADV'

3. Run WEvade-B-Q:

		## Standard training
		python3 main_WEvade_B_Q.py --checkpoint './ckpt/coco.pth' --dataset-folder './dataset/coco/val' --exp 'COCO' --num-attack 10 --norm 'inf'
		
		## Adversarial training
		python3 main_WEvade_B_Q.py --checkpoint './ckpt/coco_adv_train.pth' --dataset-folder './dataset/coco/val' --exp 'COCO-ADV' --num-attack 10 --norm 'inf'

## Citation

If you find our work useful for your research, please consider citing the paper (To appear in ACM CCS 2023)
```
@inproceedings{jiang2023evading,
  title={Evading Watermark based Detection of AI-Generated Content},
  author={Jiang, Zhengyuan and Zhang, Jinghuai and Gong, Neil Zhenqiang},
  booktitle={ACM Conference on Computer and Communications Security (CCS)},
  year={2023}
}
```
