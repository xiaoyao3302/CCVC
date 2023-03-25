# CCVC

This is the official PyTorch implementation of our paper:

> **[Conflict-Based Cross-View Consistency for Semi-Supervised Semantic Segmentation](https://arxiv.org/abs/2303.01276)**
> *In Conference on Computer Vision and Pattern Recognition (CVPR), 2023*



> **Abstract.** 
> Semi-supervised semantic segmentation (SSS) has recently gained increasing research interest as it can reduce the requirement for large-scale fully-annotated training data. The current methods often suffer from the confirmation bias from the pseudo-labelling process, which can be alleviated by the co-training framework. The current co-training-based SSS methods rely on hand-crafted perturbations to prevent the different sub-nets from collapsing into each other, but these artificial perturbations cannot lead to the optimal solution. In this work, we propose a new conflict-based cross-view consistency (CCVC) method based on a two-branch co-training framework which aims at enforcing the two sub-nets to learn informative features from irrelevant views. In particular, we first propose a new cross-view consistency (CVC) strategy that encourages the two sub-nets to learn distinct features from the same input by introducing a feature discrepancy loss, while these distinct features are expected to generate consistent prediction scores of the input. The CVC strategy helps to prevent the two sub-nets from stepping into the collapse. In addition, we further propose a conflict-based pseudo-labelling (CPL) method to guarantee the model will learn more useful information from conflicting predictions, which will lead to a stable training process. We validate our new CCVC approach on the SSS benchmark datasets where our method achieves new state-of-the-art performance.



## Getting Started

### Installation

```bash
cd CCVC
conda create -n CCVC python=3.6
conda activate CCVC
pip install -r requirements.txt
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
```

Please refer to **[UniMatch](https://github.com/LiheYoung/UniMatch)** for more implement details

### Pretrained Backbone:

[ResNet-50](https://drive.google.com/file/d/1mqUrqFvTQ0k5QEotk4oiOFyP6B9dVZXS/view?usp=sharing) | [ResNet-101](https://drive.google.com/file/d/1Rx0legsMolCWENpfvE2jUScT3ogalMO8/view?usp=sharing)

```
├── ./pretrained
    ├── resnet50.pth
    └── resnet101.pth
```

### Dataset:

- Pascal: [JPEGImages](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) | [SegmentationClass](https://drive.google.com/file/d/1ikrDlsai5QSf2GiSUR3f8PZUzyTubcuF/view?usp=sharing)
- Cityscapes: [leftImg8bit](https://www.cityscapes-dataset.com/file-handling/?packageID=3) | [gtFine](https://drive.google.com/file/d/1E_27g9tuHm6baBqcA7jct_jqcGA89QPm/view?usp=sharing)

Please modify the dataset path in configuration files. 

*The groundtruth mask ids have already been pre-processed. You may use them directly.*

```
├── [Your Pascal Path]
    ├── JPEGImages
    └── SegmentationClass
    
├── [Your Cityscapes Path]
    ├── leftImg8bit
    └── gtFine
```



## Usage

### CCVC (without data augmentation)

```bash
python CCVC_no_aug.py \
    --config 'configs/pascal.yaml' \
    --backbone 'resnet101' \
    --labeled_id_path 'partitions/pascal/366/labeled.txt' \
    --unlabeled_id_path 'partitions/pascal/366/unlabeled.txt' \
    --save_path 'exp/pascal/366/test/' \
    --load_path 'test_DDP' \
    --nodes 1 \
    --port 4434 \
    --gpus 4 \
    --epochs 40 \
    --batch_size 2 \
    --crop_size 512 \
    --mode_mapping 'else' \
    --mode_confident 'vote_threshold' \
    --conf_threshold 0.9 \
    --use_SPL False \
    --use_con True \
    --use_dis True \
    --use_MLP True \
    --use_norm True \
    --use_dropout True \
    --w_CE 5.0 \
    --w_con 2.0 \
    --w_dis 1.0 \
    --lr_network 10.0 \
    --lr_backbone 1.0
```

or

### CCVC (with data augmentation)

```bash
python CCVC_aug.py \
    --config 'configs/pascal.yaml' \
    --backbone 'resnet101' \
    --labeled_id_path 'partitions/pascal/366/labeled.txt' \
    --unlabeled_id_path 'partitions/pascal/366/unlabeled.txt' \
    --save_path 'exp/pascal/366/test/' \
    --load_path 'test_DDP' \
    --nodes 1 \
    --port 4434 \
    --gpus 4 \
    --epochs 40 \
    --batch_size 4 \
    --crop_size 512 \
    --mode_mapping 'else' \
    --mode_confident 'vote_threshold' \
    --conf_threshold 0.9 \
    --use_SPL False \
    --use_MLP True \
    --use_norm True \
    --use_dropout True \
    --w_CE 5.0 \
    --w_con 2.0 \
    --w_dis 1.0 \
    --lr_network 10.0 \
    --lr_backbone 1.0
```

or

you can directly run the sh files like:

bash ./tools/train.sh



To run with different settings, please modify the above mentioned settings.

If you run the code for more epochs, you may get a better result.

Note that all of our experiments are tested on 4 A6000 GPUs.



## Citation

If you find these projects useful, please consider citing:

```bibtex
@article{wang2023conflict,
  title={Conflict-Based Cross-View Consistency for Semi-Supervised Semantic Segmentation},
  author={Wang, Zicheng and Zhao, Zhen and Zhou, Luping and Xu, Dong and Xing, Xiaoxia and Kong, Xiangyu},
  journal={arXiv preprint arXiv:2303.01276},
  year={2023}
}
```

## Acknowledgement

We thank [AEL](https://github.com/hzhupku/SemiSeg-AEL), [CPS](https://github.com/charlesCXK/TorchSemiSeg), [CutMix-Seg](https://github.com/Britefury/cutmix-semisup-seg), [DeepLabv3Plus](https://github.com/YudeWang/deeplabv3plus-pytorch), [PseudoSeg](https://github.com/googleinterns/wss), [PS-MT](https://github.com/yyliu01/PS-MT), [SimpleBaseline](https://github.com/jianlong-yuan/SimpleBaseline), [U<sup>2</sup>PL](https://github.com/Haochen-Wang409/U2PL), [UniMatch](https://github.com/LiheYoung/UniMatch) and other relevant works for their amazing open-sourced projects!
