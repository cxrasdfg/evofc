import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

__all__ = ['densenet']

class Bottleneck(nn.Module):
    def __init__(self, inplanes, expansion=4, growthRate=12, dropRate=0):
        super(Bottleneck, self).__init__()
        planes = expansion * growthRate
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, growthRate, kernel_size=3, 
                               padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.dropRate = dropRate

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.dropRate > 0:
            out = F.dropout(out, p=self.dropRate, training=self.training)

        out = torch.cat((x, out), 1)

        return out

class BasicBlock(nn.Module):
    def __init__(self, inplanes, expansion=1, growthRate=12, dropRate=0):
        super(BasicBlock, self).__init__()
        planes = expansion * growthRate
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, growthRate, kernel_size=3, 
                               padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.dropRate = dropRate

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        if self.dropRate > 0:
            out = F.dropout(out, p=self.dropRate, training=self.training)

        out = torch.cat((x, out), 1)

        return out

class Transition(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=1,
                               bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = F.avg_pool2d(out, 2)
        return out

class DenseNet(nn.Module):

    def __init__(self, depth=22, block=BasicBlock, 
        dropRate=0, num_classes=10, growthRate=12, compressionRate=1, cfg=None):
        super(DenseNet, self).__init__()

        assert (depth - 4) % 3 == 0, 'depth should be 3n+4'
        n = (depth - 4) / 3 if block == BasicBlock else (depth - 4) // 6
        n = int(n)

        self.growthRate = growthRate
        
        # self.inplanes is a global variable used across multiple
        # helper functions
        self.inplanes = growthRate * 2 

        self.dropRate = dropRate

        in_planes_temp = self.inplanes
        if cfg is None:
            cfg = []
            if block is BasicBlock:
                for i in range(n):
                    cfg.append((in_planes_temp, self.growthRate))
                    in_planes_temp += self.growthRate
                cfg.append((in_planes_temp, int(math.floor(in_planes_temp // compressionRate))))
                for i in range(n):
                    cfg.append((in_planes_temp, self.growthRate))
                    in_planes_temp += self.growthRate
                cfg.append((in_planes_temp, int(math.floor(in_planes_temp // compressionRate))))
                for i in range(n):
                    cfg.append((in_planes_temp, self.growthRate))
                    in_planes_temp += self.growthRate
                cfg.append((in_planes_temp))
            else:
                assert 0
        
        self.cfg = cfg
        # import pdb;pdb.set_trace()

        self.conv1 = nn.Conv2d(3, cfg[0][0], kernel_size=3, padding=1,
                               bias=False)
        self.dense1 = self._make_denseblock(block, n, cfg = cfg[0:n])
        self.trans1 = self._make_transition(compressionRate, cfg = cfg[n])
        self.dense2 = self._make_denseblock(block, n, cfg = cfg[n+1: 2*n+1])
        self.trans2 = self._make_transition(compressionRate, cfg = cfg[2*n+1])
        self.dense3 = self._make_denseblock(block, n, cfg = cfg[2*n+2 : 3*n +2])
        self.bn = nn.BatchNorm2d(cfg[3*n +2])
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(cfg[3*n +2], num_classes)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_denseblock(self, block, blocks, cfg):
        layers = []
        for i in range(blocks):
            # Currently we fix the expansion ratio as the default value
            # layers.append(block(self.inplanes, growthRate=self.growthRate, dropRate=self.dropRate))
            # self.inplanes += self.growthRate
            layers.append(block(cfg[i][0], growthRate=cfg[i][1], dropRate=self.dropRate))

        return nn.Sequential(*layers)

    def _make_transition(self, compressionRate, cfg):
        inplanes = cfg[0]
        # outplanes = int(math.floor(cfg // compressionRate))
        # self.inplanes = outplanes
        return Transition(inplanes, cfg[1])


    def forward(self, x):
        x = self.conv1(x)

        x = self.trans1(self.dense1(x)) 
        x = self.trans2(self.dense2(x)) 
        x = self.dense3(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def densenet(**kwargs):
    """
    Constructs a ResNet model.
    """
    return DenseNet(**kwargs)


# python cifar_prune.py --arch densenet --depth 40 --dataset cifar10 --percent 0.3 --resume [PATH TO THE MODEL] --save_dir [DIRECTORY TO STORE RESULT]
# python cifar_prune.py --arch densenet --depth 100 --compressionRate 2 --dataset cifar10 --percent 0.3 --resume [PATH TO THE MODEL] --save_dir [DIRECTORY TO STORE RESULT]

def densenet40(**kwargs):
    return densenet(depth=40, **kwargs)

def densenet_100(**kwargs):
    return densenet(depth=100, compressionRate=2, **kwargs)

def densenet_bc_100(**kwargs):
    return densenet(depth=100, block=Bottleneck, compressionRate=2, **kwargs)