import numpy as np
import logging
import os
import torch


def count_params(model):
    param_num = sum(p.numel() for p in model.parameters())
    return param_num / 1e6


def color_map(dataset='pascal'):
    cmap = np.zeros((256, 3), dtype='uint8')

    if dataset == 'pascal' or dataset == 'coco':
        def bitget(byteval, idx):
            return (byteval & (1 << idx)) != 0

        for i in range(256):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << 7-j)
                g = g | (bitget(c, 1) << 7-j)
                b = b | (bitget(c, 2) << 7-j)
                c = c >> 3

            cmap[i] = np.array([r, g, b])

    elif dataset == 'cityscapes':
        cmap[0] = np.array([128, 64, 128])
        cmap[1] = np.array([244, 35, 232])
        cmap[2] = np.array([70, 70, 70])
        cmap[3] = np.array([102, 102, 156])
        cmap[4] = np.array([190, 153, 153])
        cmap[5] = np.array([153, 153, 153])
        cmap[6] = np.array([250, 170, 30])
        cmap[7] = np.array([220, 220, 0])
        cmap[8] = np.array([107, 142, 35])
        cmap[9] = np.array([152, 251, 152])
        cmap[10] = np.array([70, 130, 180])
        cmap[11] = np.array([220, 20, 60])
        cmap[12] = np.array([255,  0,  0])
        cmap[13] = np.array([0,  0, 142])
        cmap[14] = np.array([0,  0, 70])
        cmap[15] = np.array([0, 60, 100])
        cmap[16] = np.array([0, 80, 100])
        cmap[17] = np.array([0,  0, 230])
        cmap[18] = np.array([119, 11, 32])

    return cmap


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, num=1):
        if self.length > 0:
            # currently assert num==1 to avoid bad usage, refine when there are some explict requirements
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val * num
            self.count += num
            self.avg = self.sum / self.count


def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.ndim in [1, 2, 3]
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))
    area_output, _ = np.histogram(output, bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


logs = set()


def init_log(name, level=logging.INFO):
    if (name, level) in logs:
        return
    logs.add((name, level))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        logger.addFilter(lambda record: rank == 0)
    else:
        rank = 0
    format_str = "[%(asctime)s][%(levelname)8s] %(message)s"
    formatter = logging.Formatter(format_str)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def init_log_save(save_dir, name, level=logging.INFO):
    save_name = save_dir + 'logger.txt'
    if (name, level) in logs:
        return
    logs.add((name, level))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.FileHandler(save_name)
    ch.setLevel(level)
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        logger.addFilter(lambda record: rank == 0)
    else:
        rank = 0
    format_str = "[%(asctime)s][%(levelname)8s] %(message)s"
    formatter = logging.Formatter(format_str)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def rand_bbox_1(size, lam=None):
    # past implementation
    W = size[2]
    H = size[3]
    B = size[0]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    cx = np.random.randint(size=[B, ], low=int(W/8), high=W)
    cy = np.random.randint(size=[B, ], low=int(H/8), high=H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)

    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cut_mix_unlabeled(to_process, local_rank):
    # unlabeled_img1 and unlabeled_img2
    # unlabeled1_pred1 and unlabeled2_pred1
    # unlabeled1_pred2 and unlabeled2_pred2
    # ignore_img_mask1 and ignore_img_mask2

    # image: B, C, W, H
    # mask: B, W, H
    unlabeled_img1, unlabeled_img2, unlabeled1_pred1, unlabeled2_pred1, unlabeled1_pred2, unlabeled2_pred2, unlabeled1_pred3, unlabeled2_pred3, ignore_img_mask1, ignore_img_mask2 = to_process
    
    mix_unlabeled_image = unlabeled_img1.clone()
    mix_ignore_image_mask = ignore_img_mask1.clone()
    mix_unlabeled_pred1 = unlabeled1_pred1.clone()
    mix_unlabeled_pred2 = unlabeled1_pred2.clone()
    mix_unlabeled_pred3 = unlabeled1_pred3.clone()

    # double check, different loader, different idx in a batch
    u_rand_index = torch.randperm(unlabeled_img1.size()[0])[:unlabeled_img1.size()[0]].cuda(local_rank)
    u_bbx1, u_bby1, u_bbx2, u_bby2 = rand_bbox_1(unlabeled_img1.size(), lam=np.random.beta(4, 4))

    for i in range(0, mix_unlabeled_image.shape[0]):
        mix_unlabeled_image[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            unlabeled_img2[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]

        mix_unlabeled_pred1[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            unlabeled2_pred1[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]

        mix_unlabeled_pred2[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            unlabeled2_pred2[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
        
        mix_unlabeled_pred3[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            unlabeled2_pred3[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]

        mix_ignore_image_mask[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            ignore_img_mask2[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]

    del unlabeled_img1, unlabeled_img2, unlabeled1_pred1, unlabeled2_pred1, unlabeled1_pred2, unlabeled2_pred2, unlabeled1_pred3, unlabeled2_pred3, ignore_img_mask1, ignore_img_mask2

    return mix_unlabeled_image, mix_ignore_image_mask, mix_unlabeled_pred1, mix_unlabeled_pred2, mix_unlabeled_pred3


def cut_mix_labeled(to_process, local_rank):
    # labeled_img1 and labeled_img2
    # labeled_img_mask1 and labeled_img_mask2

    # image: B, C, W, H
    # mask: B, W, H
    labeled_img1, labeled_img2, labeled_img_mask1, labeled_img_mask2 = to_process
    
    mix_labeled_image = labeled_img1.clone()
    mix_labeled_mask = labeled_img_mask1.clone()

    # double check, different loader, different idx in a batch
    u_rand_index = torch.randperm(labeled_img1.size()[0])[:labeled_img1.size()[0]].cuda(local_rank)
    u_bbx1, u_bby1, u_bbx2, u_bby2 = rand_bbox_1(labeled_img1.size(), lam=np.random.beta(4, 4))

    for i in range(0, mix_labeled_image.shape[0]):
        mix_labeled_image[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            labeled_img2[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]

        mix_labeled_mask[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            labeled_img_mask2[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]

    del labeled_img1, labeled_img2, labeled_img_mask1, labeled_img_mask2

    return mix_labeled_image, mix_labeled_mask


def cut_mix_aug(to_process, local_rank):
    # unlabeled_img1, unlabeled_img2, unlabeled_pred1, unlabeled_pred1, ignore_img_mask1, ignore_img_mask2

    # image: B, C, W, H
    # mask: B, W, H
    unlabeled_img1, unlabeled_img2, unlabeled_aug_img1, unlabeled_aug_img2, unlabeled_pred1, unlabeled_pred2, ignore_img_mask1, ignore_img_mask2 = to_process
    
    mix_unlabeled_image = unlabeled_img1.clone()
    mix_aug_unlabeled_image = unlabeled_aug_img1.clone()
    mix_ignore_image_mask = ignore_img_mask1.clone()
    mix_unlabeled_pred = unlabeled_pred1.clone()

    # double check, different loader, different idx in a batch
    u_rand_index = torch.randperm(unlabeled_img1.size()[0])[:unlabeled_img1.size()[0]].cuda(local_rank)
    u_bbx1, u_bby1, u_bbx2, u_bby2 = rand_bbox_1(unlabeled_img1.size(), lam=np.random.beta(4, 4))

    for i in range(0, mix_unlabeled_image.shape[0]):
        mix_unlabeled_image[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            unlabeled_img2[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]

        mix_aug_unlabeled_image[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            unlabeled_aug_img2[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]

        mix_unlabeled_pred[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            unlabeled_pred2[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]

        mix_ignore_image_mask[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            ignore_img_mask2[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]

    del unlabeled_img1, unlabeled_img2, unlabeled_aug_img1, unlabeled_aug_img2, unlabeled_pred1, unlabeled_pred2, ignore_img_mask1, ignore_img_mask2

    return mix_unlabeled_image, mix_aug_unlabeled_image, mix_ignore_image_mask, mix_unlabeled_pred


def cut_mix_vote(to_process, local_rank):
    # image: B, C, W, H
    # mask: B, W, H
    unlabeled_img1, unlabeled_img2, mix_pseudo_label_img1, mix_pseudo_label_img2, mix_pseudo_mask_img1, mix_pseudo_mask_img2, ignore_img_mask1, ignore_img_mask2 = to_process
    
    mix_unlabeled_image = unlabeled_img1.clone()
    mix_ignore_image_mask = ignore_img_mask1.clone()
    mix_pseudo_label_img = mix_pseudo_label_img1.clone()
    mix_pseudo_mask_img = mix_pseudo_mask_img1.clone()

    # double check, different loader, different idx in a batch
    u_rand_index = torch.randperm(unlabeled_img1.size()[0])[:unlabeled_img1.size()[0]].cuda(local_rank)
    u_bbx1, u_bby1, u_bbx2, u_bby2 = rand_bbox_1(unlabeled_img1.size(), lam=np.random.beta(4, 4))

    for i in range(0, mix_unlabeled_image.shape[0]):
        mix_unlabeled_image[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            unlabeled_img2[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]

        mix_pseudo_label_img[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            mix_pseudo_label_img2[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]

        mix_pseudo_mask_img[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            mix_pseudo_mask_img2[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]

        mix_ignore_image_mask[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            ignore_img_mask2[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]

    del unlabeled_img1, unlabeled_img2, mix_pseudo_label_img1, mix_pseudo_label_img2, mix_pseudo_mask_img1, mix_pseudo_mask_img2, ignore_img_mask1, ignore_img_mask2

    return mix_unlabeled_image, mix_ignore_image_mask, mix_pseudo_label_img, mix_pseudo_mask_img


def vote_soft_label_selection(pred1, pred2, threshold):
    """ 
        input:
            pred1 & pred2: logits with per-class prediction probability: B, C, H, W
            threshold: confident predictions

        output:
            label: one label for all three branches
            mask: confident or not
    """
    # vote and obey
    # same prediction: weight = 1
    # different prediction and confident: weight = 0.75
    # different prediction and unconfident: weight = 0.5
    
    pred1_confidence, pred1_label = pred1.max(dim=1)
    pred2_confidence, pred2_label = pred2.max(dim=1)

    same_pred = (pred1_label == pred2_label)

    different_confident_pred1 = (pred1_label != pred2_label) & (pred1_confidence > threshold)
    different_confident_pred2 = (pred1_label != pred2_label) & (pred2_confidence > threshold)

    different_unconfident_pred1 = (pred1_label != pred2_label) & (pred1_confidence <= threshold)
    different_unconfident_pred2 = (pred1_label != pred2_label) & (pred2_confidence <= threshold)

    return same_pred, different_confident_pred1, different_confident_pred2, different_unconfident_pred1, different_unconfident_pred2


def soft_label_selection(pred1, pred2, threshold):
    """ 
        input:
            pred1 & pred2: logits with per-class prediction probability: B, C, H, W
            threshold: confident predictions

        output:
            label: one label for all three branches
            mask: confident or not
    """
    # only soft
    # confident: weight = 1.0
    # unconfident: weight = 0.5
    
    pred1_confidence, pred1_label = pred1.max(dim=1)
    pred2_confidence, pred2_label = pred2.max(dim=1)

    confident_pred1 = (pred1_confidence > threshold)
    confident_pred2 = (pred2_confidence > threshold)

    unconfident_pred1 = (pred1_confidence <= threshold)
    unconfident_pred2 = (pred2_confidence <= threshold)

    return confident_pred1, confident_pred2, unconfident_pred1, unconfident_pred2


def vote_label_selection(pred1, pred2):
    """ 
        input:
            pred1 & pred2: logits with per-class prediction probability: B, C, H, W
            threshold: confident predictions

        output:
            label: one label for all three branches
            mask: confident or not
    """
    # only vote
    # same prediction: weight = 1.0
    # different prediction: weight = 0.5
    
    pred1_confidence, pred1_label = pred1.max(dim=1)
    pred2_confidence, pred2_label = pred2.max(dim=1)

    same_pred = (pred1_label == pred2_label)
    different_pred = (pred1_label != pred2_label)

    return same_pred, different_pred


def vote_threshold_label_selection(pred1, pred2, threshold):
    """ 
        input:
            pred1 & pred2: logits with per-class prediction probability: B, C, H, W
            threshold: confident predictions

        output:
            label: one label for all three branches
            mask: confident or not
    """
    # only vote
    # same prediction: weight = 1.0
    # different prediction: weight = 0.5
    
    pred1_confidence, pred1_label = pred1.max(dim=1)
    pred2_confidence, pred2_label = pred2.max(dim=1)

    same_pred = (pred1_label == pred2_label)
    different_pred = (pred1_label != pred2_label)

    different1_confident = different_pred * (pred1_confidence > threshold)
    different1_else = ~different1_confident
    different2_confident = different_pred * (pred2_confidence > threshold)
    different2_else = ~different2_confident

    return different1_confident, different1_else, different2_confident, different2_else


def cut_mix(to_process, local_rank):
    # unlabeled_img1 and unlabeled_img2
    # unlabeled1_pred1 and unlabeled2_pred1
    # unlabeled1_pred2 and unlabeled2_pred2
    # ignore_img_mask1 and ignore_img_mask2

    # image: B, C, W, H
    # mask: B, W, H
    unlabeled_img1, unlabeled_img2, unlabeled1_pred1, unlabeled2_pred1, unlabeled1_pred2, unlabeled2_pred2, ignore_img_mask1, ignore_img_mask2 = to_process
    
    mix_unlabeled_image = unlabeled_img1.clone()
    mix_ignore_image_mask = ignore_img_mask1.clone()
    mix_unlabeled_pred1 = unlabeled1_pred1.clone()
    mix_unlabeled_pred2 = unlabeled1_pred2.clone()

    # double check, different loader, different idx in a batch
    u_rand_index = torch.randperm(unlabeled_img1.size()[0])[:unlabeled_img1.size()[0]].cuda(local_rank)
    u_bbx1, u_bby1, u_bbx2, u_bby2 = rand_bbox_1(unlabeled_img1.size(), lam=np.random.beta(4, 4))

    for i in range(0, mix_unlabeled_image.shape[0]):
        mix_unlabeled_image[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            unlabeled_img2[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]

        mix_unlabeled_pred1[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            unlabeled2_pred1[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]

        mix_unlabeled_pred2[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            unlabeled2_pred2[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]

        mix_ignore_image_mask[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            ignore_img_mask2[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]

    del unlabeled_img1, unlabeled_img2, unlabeled1_pred1, unlabeled2_pred1, unlabeled1_pred2, unlabeled2_pred2, ignore_img_mask1, ignore_img_mask2

    return mix_unlabeled_image, mix_ignore_image_mask, mix_unlabeled_pred1, mix_unlabeled_pred2

