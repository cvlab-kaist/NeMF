from functools import reduce
from operator import add

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet

class ResNet101(nn.Module):
    def __init__(self, feat_ids=tuple(range(4, 34))):
        super().__init__()
        self.backbone = resnet.resnet101(pretrained=True)
        self.feat_ids = feat_ids
        
        nbottlenecks = [3, 4, 23, 3]
        self.lids = reduce(add, [[i + 1] * x for i, x in enumerate(nbottlenecks)])
        self.bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), nbottlenecks)))

    def forward(self, img):
        r""" Extract intermediate features from ResNet"""
        feats = []

        # Layer 0
        feat = self.backbone.conv1.forward(img)
        feat = self.backbone.bn1.forward(feat)
        feat = self.backbone.relu.forward(feat)
        feat = self.backbone.maxpool.forward(feat)

        # Layer 1-4
        for hid, (bid, lid) in enumerate(zip(self.bottleneck_ids, self.lids)):
            res = feat
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].conv1.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].bn1.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].conv2.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].bn2.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].conv3.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].bn3.forward(feat)

            if bid == 0:
                res = self.backbone.__getattr__('layer%d' % lid)[bid].downsample.forward(res)

            feat += res

            if hid + 1 in self.feat_ids:
                feats.append(feat.clone())

            feat = self.backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)

        return feats