import model.backbone.resnet as resnet
from model.backbone.xception import xception

import torch
from torch import nn
import torch.nn.functional as F


class DeepLabV3Plus(nn.Module):
    def __init__(self, args, cfg, use_MLP=False):
        super(DeepLabV3Plus, self).__init__()

        self.use_MLP = use_MLP
        self.use_dropout = args.use_dropout

        if 'resnet' in args.backbone:
            self.backbone = \
                resnet.__dict__[args.backbone](True, multi_grid=cfg['multi_grid'],
                                                 replace_stride_with_dilation=cfg['replace_stride_with_dilation'])
        else:
            assert args.backbone == 'xception'
            self.backbone = xception(True)

        low_channels = 256
        high_channels = 2048

        self.head = ASPPModule(high_channels, cfg['dilations'])

        self.reduce = nn.Sequential(nn.Conv2d(low_channels, 48, 1, bias=False),
                                    nn.BatchNorm2d(48),
                                    nn.ReLU(True))

        self.fuse = nn.Sequential(nn.Conv2d(high_channels // 8 + 48, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True),
                                  nn.Conv2d(256, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True))
        
        if self.use_MLP:
            # max the discrepancy between the output feature after concatenation of c1 and c4
            if args.use_norm:
                self.mapping = nn.Sequential(nn.Conv2d(high_channels // 8 + 48, high_channels // 8 + 48, 1, bias=False), nn.BatchNorm2d(high_channels // 8 + 48), nn.ReLU(True))
                if args.use_dropout:
                    self.dropout = nn.Dropout2d(p=args.dropout)
            else:
                self.mapping = nn.Conv2d(high_channels // 8 + 48, high_channels // 8 + 48, 1, bias=False)
                
        self.classifier = nn.Conv2d(256, cfg['nclass'], 1, bias=True)

    def forward(self, x):
        h, w = x.shape[-2:]

        feats = self.backbone.base_forward(x)
        c1, c4 = feats[0], feats[-1]

        feature, pred = self._decode(c1, c4)
        pred = F.interpolate(pred, size=(h, w), mode="bilinear", align_corners=True)

        return feature, pred

    def _decode(self, c1, c4):
        c4 = self.head(c4)
        c4 = F.interpolate(c4, size=c1.shape[-2:], mode="bilinear", align_corners=True)

        c1 = self.reduce(c1)

        feature = torch.cat([c1, c4], dim=1)

        # max discrepancy before 3*3 conv
        if self.use_MLP:
            feature = self.mapping(feature)
            if self.use_dropout:
                feature = self.dropout(feature)
            return_feature = feature
        else:
            return_feature = feature
        feature = self.fuse(return_feature)

        pred = self.classifier(feature)

        return return_feature, pred


def ASPPConv(in_channels, out_channels, atrous_rate):
    block = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate,
                                    dilation=atrous_rate, bias=False),
                          nn.BatchNorm2d(out_channels),
                          nn.ReLU(True))
    return block


class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__()
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU(True))

    def forward(self, x):
        h, w = x.shape[-2:]
        pool = self.gap(x)
        return F.interpolate(pool, (h, w), mode="bilinear", align_corners=True)


class ASPPModule(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPPModule, self).__init__()
        out_channels = in_channels // 8
        rate1, rate2, rate3 = atrous_rates

        self.b0 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                nn.BatchNorm2d(out_channels),
                                nn.ReLU(True))
        self.b1 = ASPPConv(in_channels, out_channels, rate1)
        self.b2 = ASPPConv(in_channels, out_channels, rate2)
        self.b3 = ASPPConv(in_channels, out_channels, rate3)
        self.b4 = ASPPPooling(in_channels, out_channels)

        self.project = nn.Sequential(nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
                                     nn.BatchNorm2d(out_channels),
                                     nn.ReLU(True))

    def forward(self, x):
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = self.b4(x)
        y = torch.cat((feat0, feat1, feat2, feat3, feat4), 1)
        return self.project(y)


class Discrepancy_DeepLabV3Plus(nn.Module):
    def __init__(self, args, cfg):
        super(Discrepancy_DeepLabV3Plus, self).__init__()
        # three branch, branch1 is the main branch without modification, the other two branches can add mapping layer
        if args.mode_mapping == 'both':
            self.branch1 = DeepLabV3Plus(args, cfg, use_MLP=args.use_MLP)
        else:
            self.branch1 = DeepLabV3Plus(args, cfg)
        self.branch2 = DeepLabV3Plus(args, cfg, use_MLP=args.use_MLP)

    def forward(self, x):
        logits = {}

        feature1, pred1 = self.branch1(x)
        feature2, pred2 = self.branch2(x)
        
        logits['pred1'] = pred1
        logits['feature1'] = feature1
        logits['pred2'] = pred2
        logits['feature2'] = feature2
        
        return logits
