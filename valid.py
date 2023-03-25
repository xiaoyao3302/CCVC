import argparse
from itertools import cycle
import logging
import os
import pprint

import torch
import numpy as np
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
import yaml

from dataset.semi import SemiDataset
from model.semseg.deeplabv3plus import DeepLabV3Plus
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, AverageMeter, intersectionAndUnion, init_log
from util.dist_helper import setup_distributed


parser = argparse.ArgumentParser(description='Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, default=None)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)


def evaluate(local_rank, model, loader, mode, args, cfg):
    model.eval()
    assert mode in ['original', 'center_crop', 'sliding_window']
    intersection_meter1 = AverageMeter()
    union_meter1 = AverageMeter()
    intersection_meter2 = AverageMeter()
    union_meter2 = AverageMeter()
    intersection_meter_ave = AverageMeter()
    union_meter_ave = AverageMeter()

    different_pred = 0
    total_pred = 0

    with torch.no_grad():
        for img, mask, id in loader:
            img = img.cuda(local_rank)

            total_pred += img.shape[0]

            if mode == 'sliding_window':
                grid = args.crop_size
                b, _, h, w = img.shape
                final_1 = torch.zeros(b, 19, h, w).cuda(local_rank)
                final_2 = torch.zeros(b, 19, h, w).cuda(local_rank)
                final_ave = torch.zeros(b, 19, h, w).cuda(local_rank)
                row = 0
                while row < h:
                    col = 0
                    while col < w:
                        logits = model(img[:, :, row: min(h, row + grid), col: min(w, col + grid)])
                        pred1 = logits['pred1']
                        pred2 = logits['pred2']
                        pred_ave = (pred1 + pred2) / 2

                        final_1[:, :, row: min(h, row + grid), col: min(w, col + grid)] += pred1.softmax(dim=1)
                        final_2[:, :, row: min(h, row + grid), col: min(w, col + grid)] += pred2.softmax(dim=1)
                        final_ave[:, :, row: min(h, row + grid), col: min(w, col + grid)] += pred_ave.softmax(dim=1)

                        col += int(grid * 2 / 3)

                    row += int(grid * 2 / 3)

                pred1 = final_1.argmax(dim=1)
                pred2 = final_2.argmax(dim=1)
                pred_ave = final_ave.argmax(dim=1)

            else:
                if mode == 'center_crop':
                    h, w = img.shape[-2:]
                    start_h, start_w = (h - args.crop_size) // 2, (w - args.crop_size) // 2
                    img = img[:, :, start_h:start_h + args.crop_size, start_w:start_w + args.crop_size]
                    mask = mask[:, start_h:start_h + args.crop_size, start_w:start_w + args.crop_size]

                logits = model(img)

                pred1 = logits['pred1']
                pred2 = logits['pred2']
                pred_ave = (pred1 + pred2) / 2

                pred1 = pred1.argmax(dim=1)
                pred2 = pred2.argmax(dim=1)
                pred_ave = pred_ave.argmax(dim=1)

            intersection1, union1, target1 = \
                intersectionAndUnion(pred1.cpu().numpy(), mask.numpy(), cfg['nclass'], 255)
            intersection2, union2, target2 = \
                intersectionAndUnion(pred2.cpu().numpy(), mask.numpy(), cfg['nclass'], 255)
            intersection_ave, union_ave, target_ave = \
                intersectionAndUnion(pred_ave.cpu().numpy(), mask.numpy(), cfg['nclass'], 255)

            reduced_intersection1 = torch.from_numpy(intersection1).cuda(local_rank)
            reduced_union1 = torch.from_numpy(union1).cuda(local_rank)
            reduced_target1 = torch.from_numpy(target1).cuda(local_rank)

            reduced_intersection2 = torch.from_numpy(intersection2).cuda(local_rank)
            reduced_union2 = torch.from_numpy(union2).cuda(local_rank)
            reduced_target2 = torch.from_numpy(target2).cuda(local_rank)

            reduced_intersection_ave = torch.from_numpy(intersection_ave).cuda(local_rank)
            reduced_union_ave = torch.from_numpy(union_ave).cuda(local_rank)
            reduced_target_ave = torch.from_numpy(target_ave).cuda(local_rank)

            intersection_meter1.update(reduced_intersection1.cpu().numpy())
            union_meter1.update(reduced_union1.cpu().numpy())
            intersection_meter2.update(reduced_intersection2.cpu().numpy())
            union_meter2.update(reduced_union2.cpu().numpy())
            intersection_meter_ave.update(reduced_intersection_ave.cpu().numpy())
            union_meter_ave.update(reduced_union_ave.cpu().numpy())

    iou_class1 = intersection_meter1.sum / (union_meter1.sum + 1e-10)
    mIOU1 = np.mean(iou_class1) * 100.0
    iou_class2 = intersection_meter2.sum / (union_meter2.sum + 1e-10)
    mIOU2 = np.mean(iou_class2) * 100.0
    iou_class_ave = intersection_meter_ave.sum / (union_meter_ave.sum + 1e-10)
    mIOU_ave = np.mean(iou_class_ave) * 100.0

    result = {}

    result['IOU1'] = mIOU1
    result['IOU2'] = mIOU2
    result['IOU_ave'] = mIOU_ave
    result['iou_class1'] = iou_class1
    result['iou_class2'] = iou_class2
    result['iou_class_ave'] = iou_class_ave

    return result


# if __name__ == '__main__':
#     evaluate()
