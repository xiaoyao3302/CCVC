import argparse
from copy import deepcopy
import logging
import os
import pprint
import math
import pdb
import random
import numpy as np
import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import yaml
from dataset.semi import SemiDataset
from model.semseg.deeplabv3plus import Discrepancy_DeepLabV3Plus
from valid import evaluate
from valid_save import evaluate_save
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, init_log_save, vote_label_selection, soft_label_selection, vote_soft_label_selection, vote_threshold_label_selection
from util.dist_helper import setup_distributed


def str2bool(v):
    """
    Input:
        v - string
    output:
        True/False
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='Semi-Supervised Semantic Segmentation')
# default settings
parser.add_argument('--config', default='configs/pascal.yaml', type=str)
parser.add_argument('--backbone', default='resnet101', type=str)
parser.add_argument('--labeled_id_path', default='partitions/pascal/366/labeled.txt', type=str)
parser.add_argument('--unlabeled_id_path', default='partitions/pascal/366/unlabeled.txt', type=str)
parser.add_argument('--save_path', default='test', type=str)
parser.add_argument('--load_path', default='test', type=str) 
parser.add_argument('--port', default=2020, type=int)
parser.add_argument('--epochs', default=80, type=int)
parser.add_argument('--batch_size', default=6, type=int)                        # batch size per GPU
parser.add_argument('--crop_size', default=321, type=int)                       # 512 or 321
parser.add_argument('--seed', default=22, type=int)                             # 512 or 321   
parser.add_argument('--num_workers', default=2, type=int)                             # 512 or 321   
# ddp settings
parser.add_argument('--gpus', default=4, type=int, help='number of gpus per node')
parser.add_argument('--nodes', default=1, type=int, help='number of nodes')
parser.add_argument("--ddp", default=True, type=str2bool, help='distributed data parallel training or not')
# network settings
parser.add_argument('--mode_mapping', default='else', type=str)                         # both or else, the only difference is whether to use mapping on branch1
parser.add_argument('--use_con', default=False, type=str2bool)
parser.add_argument('--use_dis', default=False, type=str2bool)
parser.add_argument('--use_MLP', default=False, type=str2bool)
parser.add_argument('--use_norm', default=False, type=str2bool)
parser.add_argument('--use_dropout', default=False, type=str2bool)
parser.add_argument('--dropout', default=0.5, type=float)
parser.add_argument('--conf_threshold', default=0.5, type=float)
parser.add_argument('--mode_confident', default='no', type=str, choices=['normal', 'soft', 'vote', 'vote_threshold', 'vote_soft', 'no']) 
parser.add_argument('--use_SPL', default=False, type=str2bool)
# weight settings
parser.add_argument('--w_CE', default=1.0, type=float)
parser.add_argument('--w_con', default=1.0, type=float)
parser.add_argument('--w_dis', default=1.0, type=float)
parser.add_argument('--w_confident', default=2.0, type=float)
parser.add_argument('--w_unconfident', default=0.75, type=float)
# optimizer settings
parser.add_argument('--base_lr', default=0.001, type=float) 
parser.add_argument('--lr_network', default=5.0, type=float)                            # coefficient of the lr of other modules of the model
parser.add_argument('--lr_backbone', default=5.0, type=float)                           # coefficient of the lr of the backbone of the model
parser.add_argument('--mul_scheduler', default=0.9, type=float)                         # coefficient of the exp scheduler

args = parser.parse_args()

args.world_size = args.gpus * args.nodes
args.ddp = True if args.gpus > 1 else False

def main(gpu, ngpus_per_node, cfg, args):

    args.local_rank = gpu

    if args.local_rank <= 0:
        os.makedirs(args.save_path, exist_ok=True)
        
    logger = init_log_save(args.save_path, 'global', logging.INFO)
    logger.propagate = 0

    if args.local_rank <= 0:
        tb_dir = args.save_path
        tb = SummaryWriter(log_dir=tb_dir)

    if args.ddp:
        dist.init_process_group(backend='nccl', rank=args.local_rank, world_size=args.world_size)
        # dist.init_process_group(backend='nccl', init_method='env://', rank=args.local_rank, world_size=args.world_size)

    if args.local_rank <= 0:
        logger.info('{}\n'.format(pprint.pformat(cfg)))

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    model = Discrepancy_DeepLabV3Plus(args, cfg)
    if args.local_rank <= 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

    # pdb.set_trace()
    # TODO: check the parameter !!!

    optimizer = SGD([{'params': model.branch1.backbone.parameters(), 'lr': args.base_lr * args.lr_backbone}, 
                     {'params': model.branch2.backbone.parameters(), 'lr': args.base_lr * args.lr_backbone},
                     {'params': [param for name, param in model.named_parameters() if 'backbone' not in name],
                      'lr': args.base_lr * args.lr_network}], lr=args.base_lr, momentum=0.9, weight_decay=1e-4)

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(args.local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, broadcast_buffers=False, find_unused_parameters=False)

    # # try to load saved model
    # try:   
    #     model.load_model(args.load_path)
    #     if args.local_rank <= 0:
    #         logger.info('load saved model')
    # except:
    #     if args.local_rank <= 0:
    #         logger.info('no saved model')

    # ---- #
    # loss #
    # ---- #
    # CE loss for labeled data
    criterion_l = nn.CrossEntropyLoss(reduction='mean', ignore_index=255).cuda(args.local_rank)
    
    # consistency loss for unlabeled data
    criterion_u = nn.CrossEntropyLoss(reduction='none').cuda(args.local_rank)

    # ------- #
    # dataset #
    # ------- #
    trainset_u = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_u',
                             args.crop_size, args.unlabeled_id_path)
    trainset_l = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_l',
                             args.crop_size, args.labeled_id_path, nsample=len(trainset_u.ids))
    valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val')

    if args.ddp:
        trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    else:
        trainsampler_l = None
    trainloader_l = DataLoader(trainset_l, batch_size=args.batch_size,
                               pin_memory=True, num_workers=args.num_workers, drop_last=True, sampler=trainsampler_l)

    if args.ddp:
        trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    else:
        trainsampler_u = None
    
    trainloader_u = DataLoader(trainset_u, batch_size=args.batch_size,
                               pin_memory=True, num_workers=args.num_workers, drop_last=True, sampler=trainsampler_u)

    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=args.num_workers, drop_last=False)
    # if args.ddp:
    #     valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    # else:
    #     valsampler = None
    
    # valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=args.num_workers, drop_last=False, sampler=valsampler)

    total_iters = len(trainloader_u) * args.epochs
    previous_best = 0.0
    previous_best1 = 0.0
    previous_best2 = 0.0

    # can change with epochs, add SPL here
    conf_threshold = args.conf_threshold

    for epoch in range(args.epochs):
        if args.local_rank <= 0:
            logger.info('===========> Epoch: {:}, backbone1 LR: {:.4f}, backbone2 LR: {:.4f}, segmentation LR: {:.4f}'.format(
                epoch, optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'], optimizer.param_groups[-1]['lr']))
            logger.info('===========> Epoch: {:}, Previous best of ave: {:.2f}, Previous best of branch1: {:.2f}, Previous best of branch2: {:.2f}'.format(
                epoch, previous_best, previous_best1, previous_best2))

        total_loss, total_loss_CE, total_loss_con, total_loss_dis = 0.0, 0.0, 0.0, 0.0
        total_mask_ratio = 0.0

        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)

        loader = zip(trainloader_l, trainloader_u)

        total_labeled = 0
        total_unlabeled = 0

        for i, ((labeled_img, labeled_img_mask), (unlabeled_img, ignore_img_mask, cutmix_box)) in enumerate(loader):

            labeled_img, labeled_img_mask = labeled_img.cuda(args.local_rank), labeled_img_mask.cuda(args.local_rank)
            unlabeled_img, ignore_img_mask, cutmix_box = unlabeled_img.cuda(args.local_rank), ignore_img_mask.cuda(args.local_rank), cutmix_box.cuda(args.local_rank)

            model.train()

            optimizer.zero_grad()

            dist.barrier()

            num_lb, num_ulb = labeled_img.shape[0], unlabeled_img.shape[0]

            total_labeled += num_lb
            total_unlabeled += num_ulb

            # =========================================================================================
            # labeled data: labeled_img, labeled_img_mask
            # =========================================================================================
            labeled_logits = model(labeled_img)

            # =========================================================================================
            # unlabeled data: unlabeled_img, ignore_img_mask, cutmix_box
            # =========================================================================================
            unlabeled_logits = model(unlabeled_img)

            # to count the confident predictions
            unlabeled_pred_confidence1 = unlabeled_logits['pred1'].softmax(dim=1).max(dim=1)[0]
            unlabeled_pred_confidence2 = unlabeled_logits['pred2'].softmax(dim=1).max(dim=1)[0]

            # =========================================================================================
            # calculate loss
            # =========================================================================================

            # -------------
            # labeled
            # -------------
            # CE loss
            labeled_pred1 = labeled_logits['pred1']
            labeled_pred2 = labeled_logits['pred2']

            loss_CE1 = criterion_l(labeled_pred1, labeled_img_mask)
            loss_CE2 = criterion_l(labeled_pred2, labeled_img_mask)

            loss_CE = (loss_CE1 + loss_CE2) / 2
            loss_CE = loss_CE * args.w_CE

            # -------------
            # unlabeled
            # -------------
            # consistency loss
            unlabeled_pred1 = unlabeled_logits['pred1']
            unlabeled_pred2 = unlabeled_logits['pred2']

            if args.mode_confident == 'normal':
                loss_con1 = criterion_u(unlabeled_pred2, unlabeled_logits['pred1'].softmax(dim=1).max(dim=1)[1].detach().long()) * ((unlabeled_logits['pred1'].softmax(dim=1).max(dim=1)[0] > conf_threshold) & (ignore_img_mask != 255))
                loss_con1 = torch.sum(loss_con1) / torch.sum(ignore_img_mask != 255).item()
                loss_con2 = criterion_u(unlabeled_pred1, unlabeled_logits['pred2'].softmax(dim=1).max(dim=1)[1].detach().long()) * ((unlabeled_logits['pred2'].softmax(dim=1).max(dim=1)[0] > conf_threshold) & (ignore_img_mask != 255))
                loss_con2 = torch.sum(loss_con2) / torch.sum(ignore_img_mask != 255).item()
                
            elif args.mode_confident == 'soft':
                confident_pred1, confident_pred2, unconfident_pred1, unconfident_pred2 = soft_label_selection(unlabeled_pred1, unlabeled_pred2, conf_threshold)

                loss_con1_confident = criterion_u(unlabeled_pred2, unlabeled_logits['pred1'].softmax(dim=1).max(dim=1)[1].detach().long()) * (confident_pred1 & (ignore_img_mask != 255))
                loss_con2_confident = criterion_u(unlabeled_pred1, unlabeled_logits['pred2'].softmax(dim=1).max(dim=1)[1].detach().long()) * (confident_pred2 & (ignore_img_mask != 255))

                loss_con1_unconfident = args.w_unconfident * criterion_u(unlabeled_pred2, unlabeled_logits['pred1'].softmax(dim=1).max(dim=1)[1].detach().long()) * (unconfident_pred1 & (ignore_img_mask != 255))
                loss_con2_unconfident = args.w_unconfident * criterion_u(unlabeled_pred1, unlabeled_logits['pred2'].softmax(dim=1).max(dim=1)[1].detach().long()) * (unconfident_pred2 & (ignore_img_mask != 255))

                loss_con1 = (torch.sum(loss_con1_confident) + torch.sum(loss_con1_unconfident)) / torch.sum(ignore_img_mask != 255).item()
                loss_con2 = (torch.sum(loss_con2_confident) + torch.sum(loss_con2_unconfident)) / torch.sum(ignore_img_mask != 255).item()

            elif args.mode_confident == 'vote':
                same_pred, different_pred = vote_label_selection(unlabeled_pred1, unlabeled_pred2)

                loss_con1_same = criterion_u(unlabeled_pred2, unlabeled_logits['pred1'].softmax(dim=1).max(dim=1)[1].detach().long()) * (same_pred & (ignore_img_mask != 255))
                loss_con2_same = criterion_u(unlabeled_pred1, unlabeled_logits['pred2'].softmax(dim=1).max(dim=1)[1].detach().long()) * (same_pred & (ignore_img_mask != 255))

                loss_con1_different = args.w_confident * criterion_u(unlabeled_pred2, unlabeled_logits['pred1'].softmax(dim=1).max(dim=1)[1].detach().long()) * (different_pred & (ignore_img_mask != 255))
                loss_con2_different = args.w_confident * criterion_u(unlabeled_pred1, unlabeled_logits['pred2'].softmax(dim=1).max(dim=1)[1].detach().long()) * (different_pred & (ignore_img_mask != 255))

                loss_con1 = (torch.sum(loss_con1_same) + torch.sum(loss_con1_different)) / torch.sum(ignore_img_mask != 255).item()
                loss_con2 = (torch.sum(loss_con2_same) + torch.sum(loss_con2_different)) / torch.sum(ignore_img_mask != 255).item()

            elif args.mode_confident == 'vote_threshold':
                different1_confident, different1_else, different2_confident, different2_else = vote_threshold_label_selection(unlabeled_pred1, unlabeled_pred2, conf_threshold)

                loss_con1_else = criterion_u(unlabeled_pred2, unlabeled_logits['pred1'].softmax(dim=1).max(dim=1)[1].detach().long()) * (different1_else & (ignore_img_mask != 255))
                loss_con2_else = criterion_u(unlabeled_pred1, unlabeled_logits['pred2'].softmax(dim=1).max(dim=1)[1].detach().long()) * (different2_else & (ignore_img_mask != 255))

                loss_con1_cc = args.w_confident * criterion_u(unlabeled_pred2, unlabeled_logits['pred1'].softmax(dim=1).max(dim=1)[1].detach().long()) * (different1_confident & (ignore_img_mask != 255))
                loss_con2_cc = args.w_confident * criterion_u(unlabeled_pred1, unlabeled_logits['pred2'].softmax(dim=1).max(dim=1)[1].detach().long()) * (different2_confident & (ignore_img_mask != 255))

                loss_con1 = (torch.sum(loss_con1_else) + torch.sum(loss_con1_cc)) / torch.sum(ignore_img_mask != 255).item()
                loss_con2 = (torch.sum(loss_con2_else) + torch.sum(loss_con2_cc)) / torch.sum(ignore_img_mask != 255).item()

            elif args.mode_confident == 'vote_soft':
                same_pred, different_confident_pred1, different_confident_pred2, different_unconfident_pred1, different_unconfident_pred2 = vote_soft_label_selection(unlabeled_pred1, unlabeled_pred2, conf_threshold)

                loss_con1_same = criterion_u(unlabeled_pred2, unlabeled_logits['pred1'].softmax(dim=1).max(dim=1)[1].detach().long()) * (same_pred & (ignore_img_mask != 255))
                loss_con2_same = criterion_u(unlabeled_pred1, unlabeled_logits['pred2'].softmax(dim=1).max(dim=1)[1].detach().long()) * (same_pred & (ignore_img_mask != 255))

                loss_con1_different_confident = args.w_confident * criterion_u(unlabeled_pred2, unlabeled_logits['pred1'].softmax(dim=1).max(dim=1)[1].detach().long()) * (different_confident_pred1 & (ignore_img_mask != 255))
                loss_con2_different_confident = args.w_confident * criterion_u(unlabeled_pred1, unlabeled_logits['pred2'].softmax(dim=1).max(dim=1)[1].detach().long()) * (different_confident_pred2 & (ignore_img_mask != 255))

                loss_con1_different_unconfident = args.w_unconfident * criterion_u(unlabeled_pred2, unlabeled_logits['pred1'].softmax(dim=1).max(dim=1)[1].detach().long()) * (different_unconfident_pred1 & (ignore_img_mask != 255))
                loss_con2_different_unconfident = args.w_unconfident * criterion_u(unlabeled_pred1, unlabeled_logits['pred2'].softmax(dim=1).max(dim=1)[1].detach().long()) * (different_unconfident_pred2 & (ignore_img_mask != 255))

                loss_con1 = (torch.sum(loss_con1_same) + torch.sum(loss_con1_different_confident) + torch.sum(loss_con1_different_unconfident)) / torch.sum(ignore_img_mask != 255).item()
                loss_con2 = (torch.sum(loss_con2_same) + torch.sum(loss_con2_different_confident) + torch.sum(loss_con2_different_unconfident)) / torch.sum(ignore_img_mask != 255).item()

            else:
                loss_con1 = criterion_u(unlabeled_pred2, unlabeled_logits['pred1'].softmax(dim=1).max(dim=1)[1].detach().long()) * (ignore_img_mask != 255)
                loss_con1 = torch.sum(loss_con1) / torch.sum(ignore_img_mask != 255).item()
                loss_con2 = criterion_u(unlabeled_pred1, unlabeled_logits['pred2'].softmax(dim=1).max(dim=1)[1].detach().long()) * (ignore_img_mask != 255)
                loss_con2 = torch.sum(loss_con2) / torch.sum(ignore_img_mask != 255).item()

            loss_con = (loss_con1 + loss_con2) / 2
            loss_con = loss_con * args.w_con

            # -------------
            # both
            # -------------
            # discrepancy loss
            cos_dis = nn.CosineSimilarity(dim=1, eps=1e-6)

            # labeled
            labeled_feature1 = labeled_logits['feature1']
            labeled_feature2 = labeled_logits['feature2']
            loss_dis_labeled1 = 1 + cos_dis(labeled_feature1.detach(), labeled_feature2).mean()
            loss_dis_labeled2 = 1 + cos_dis(labeled_feature2.detach(), labeled_feature1).mean()
            loss_dis_labeled = (loss_dis_labeled1 + loss_dis_labeled2) / 2

            # unlabeled
            unlabeled_feature1 = unlabeled_logits['feature1']
            unlabeled_feature2 = unlabeled_logits['feature2']
            loss_dis_unlabeled1 = 1 + cos_dis(unlabeled_feature1.detach(), unlabeled_feature2).mean()
            loss_dis_unlabeled2 = 1 + cos_dis(unlabeled_feature2.detach(), unlabeled_feature1).mean()
            loss_dis_unlabeled = (loss_dis_unlabeled1 + loss_dis_unlabeled2) / 2

            loss_dis = (loss_dis_labeled + loss_dis_unlabeled) / 2
            loss_dis = loss_dis * args.w_dis

            # -------------
            # total
            # -------------
            loss = loss_CE
            if args.use_con:
                loss = loss + loss_con
            if args.use_dis:
                loss = loss + loss_dis

            dist.barrier()

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_loss_CE += loss_CE.item()
            total_loss_con += loss_con.item()
            total_loss_dis += loss_dis.item()

            total_confident = (((unlabeled_pred_confidence1 >= 0.95) & (ignore_img_mask != 255)).sum().item() + ((unlabeled_pred_confidence2 >= 0.95) & (ignore_img_mask != 255)).sum().item()) / 2
            total_mask_ratio += total_confident / (ignore_img_mask != 255).sum().item()

            iters = epoch * len(trainloader_u) + i

            # update lr

            backbone_lr = args.base_lr * (1 - iters / total_iters) ** args.mul_scheduler
            backbone_lr = backbone_lr * args.lr_backbone

            seg_lr = args.base_lr * (1 - iters / total_iters) ** args.mul_scheduler
            seg_lr = seg_lr * args.lr_network
                
            optimizer.param_groups[0]["lr"] = backbone_lr
            optimizer.param_groups[1]["lr"] = backbone_lr
            for ii in range(2, len(optimizer.param_groups)):
                optimizer.param_groups[ii]['lr'] = seg_lr

            if (i % (len(trainloader_u) // 8) == 0) and (args.local_rank <= 0):
                tb.add_scalar('train_loss_total', total_loss / (i+1), iters)
                tb.add_scalar('train_loss_CE', total_loss_CE / (i+1), iters)
                tb.add_scalar('train_loss_con', total_loss_con / (i+1), iters)
                tb.add_scalar('train_loss_dis', total_loss_dis / (i+1), iters)

            if (i % (len(trainloader_u) // 8) == 0) and (args.local_rank <= 0):
                logger.info('Iters: {:}, Total loss: {:.3f}, Loss CE: {:.3f}, '
                            'Loss consistency: {:.3f}, Loss discrepancy: {:.3f}, Mask: {:.3f}'.format(
                    i, total_loss / (i+1), total_loss_CE / (i+1), total_loss_con / (i+1), total_loss_dis / (i+1), 
                    total_mask_ratio / (i+1)))

        if args.use_SPL:
            conf_threshold += 0.01
            if conf_threshold >= 0.95:
                conf_threshold = 0.95

        if cfg['dataset'] == 'cityscapes':
            eval_mode = 'center_crop' if epoch < args.epochs - 20 else 'sliding_window'
        else:
            eval_mode = 'original'
        
        dist.barrier()

        # test with different branches
        if args.local_rank <= 0:
            if epoch == 4:
                evaluate_result = evaluate_save(cfg['dataset'], args.save_path, args.local_rank, model, valloader, eval_mode, args, cfg, idx_epoch=5)
            elif epoch == 9:
                evaluate_result = evaluate_save(cfg['dataset'], args.save_path, args.local_rank, model, valloader, eval_mode, args, cfg, idx_epoch=10)
            elif epoch == 19:
                evaluate_result = evaluate_save(cfg['dataset'], args.save_path, args.local_rank, model, valloader, eval_mode, args, cfg, idx_epoch=20)
            elif epoch == 39:
                evaluate_result = evaluate_save(cfg['dataset'], args.save_path, args.local_rank, model, valloader, eval_mode, args, cfg, idx_epoch=40)
            else:
                evaluate_result = evaluate(args.local_rank, model, valloader, eval_mode, args, cfg)


            mIOU1 = evaluate_result['IOU1']
            mIOU2 = evaluate_result['IOU2']
            mIOU_ave = evaluate_result['IOU_ave']

            tb.add_scalar('meanIOU_branch1', mIOU1, epoch)
            tb.add_scalar('meanIOU_branch2', mIOU2, epoch)
            tb.add_scalar('meanIOU_ave', mIOU_ave, epoch)

            logger.info('***** Evaluation with branch 1 {} ***** >>>> meanIOU: {:.2f}\n'.format(eval_mode, mIOU1))
            logger.info('***** Evaluation with branch 2 {} ***** >>>> meanIOU: {:.2f}\n'.format(eval_mode, mIOU2))
            logger.info('***** Evaluation with two branches {} ***** >>>> meanIOU: {:.2f}\n'.format(eval_mode, mIOU_ave))

            if mIOU1 > previous_best1:
                if previous_best1 != 0:
                    os.remove(os.path.join(args.save_path, 'branch1_%s_%.2f.pth' % (args.backbone, previous_best1)))
                previous_best1 = mIOU1
                torch.save(model.module.state_dict(),
                        os.path.join(args.save_path, 'branch1_%s_%.2f.pth' % (args.backbone, mIOU1)))
            
            if mIOU2 > previous_best2:
                if previous_best2 != 0:
                    os.remove(os.path.join(args.save_path, 'branch2_%s_%.2f.pth' % (args.backbone, previous_best2)))
                previous_best2 = mIOU2
                torch.save(model.module.state_dict(),
                        os.path.join(args.save_path, 'branch2_%s_%.2f.pth' % (args.backbone, mIOU2)))

            if mIOU_ave > previous_best:
                if previous_best != 0:
                    os.remove(os.path.join(args.save_path, 'ave_%s_%.2f.pth' % (args.backbone, previous_best)))
                previous_best = mIOU_ave
                torch.save(model.module.state_dict(),
                        os.path.join(args.save_path, 'ave_%s_%.2f.pth' % (args.backbone, mIOU_ave)))


if __name__ == '__main__':

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    cfg['n_gpu'] = args.gpus

    if args.ddp:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = str(args.port)
        # args.base_lr = args.base_lr * args.world_size

    if args.ddp:
        mp.spawn(main, nprocs=cfg['n_gpu'], args=(cfg['n_gpu'], cfg, args))
    else:
        main(-1, 1, cfg, args)
