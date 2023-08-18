# WEvade

This code is the official implementation of [WEvade](https://arxiv.org/abs/2305.03807).

## Preparation

1. Clone this repo from the GitHub.
	
		git clone https://github.com/zhengyuan-jiang/WEvade.git

2. Setup environment.

- Install the [pytorch](https://pytorch.org/). The latest codes are tested on Ubuntu 16.04, PyTorch 1.x.x and Python 3.x:

- Install the [adversarial-robustness-toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox):

      pip install adversarial-robustness-toolbox

## Run Wevade-B-Q (query based black-box attack)

1. Encode watermark and save watermarked images (watermark is generated using the random seed):

		python3 encode_watermarked_images.py --checkpoint './ckpt/coco.pth' --dataset-folder './dataset/coco/val' --exp 'COCO'
		<!-- python3 encode_watermarked_images.py --checkpoint './ckpt/coco_adv_train.pth' --dataset-folder './dataset/coco/val' --exp 'COCO-ADV' -->

2. (optional) Decode watermarked images for evaluation:

		python3 decode_watermarked_images.py --checkpoint './ckpt/coco.pth' --dataset-folder './dataset/coco/val' --exp 'COCO'
		<!-- python3 decode_watermarked_images.py --checkpoint './ckpt/coco_adv_train.pth' --dataset-folder './dataset/coco/val' --exp 'COCO-ADV' -->

3. Run our black-box attack:

        python3 main_WEvade_B_Q.py --checkpoint './ckpt/coco.pth' --dataset-folder './dataset/coco/val' --exp 'COCO' --num-attack 10 --norm 'inf'
        <!-- python3 main_WEvade_B_Q.py --checkpoint './ckpt/coco_adv_train.pth' --dataset-folder './dataset/coco/val' --exp 'COCO-ADV' --num-attack 10 --norm 'inf' -->

## Citation

If you find our work useful for your research, please consider citing the paper
```
@article{jiang2023evading,
  title={Evading Watermark based Detection of AI-Generated Content},
  author={Jiang, Zhengyuan and Zhang, Jinghuai and Gong, Neil Zhenqiang},
  journal={arXiv preprint arXiv:2305.03807},
  year={2023}
}
```