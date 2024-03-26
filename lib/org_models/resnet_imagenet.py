#coding=utf-8
import math
from typing import Any, Callable, List, Optional, Type, Union
from functools import partial

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
from torchvision.transforms._presets import ImageClassification
from torchvision.models._api import Weights, WeightsEnum
from torchvision.models._utils import _ovewrite_named_param, handle_legacy_interface
from torchvision.models._meta import _IMAGENET_CATEGORIES


model_urls = {
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, cfg, stride=1, downsample=None,bn_cls=None, all_and_sub_features=False):
        # cfg should be a number in this case
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, cfg, stride)
        self.bn1 = bn_cls(cfg)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(cfg, planes)
        self.bn2 = bn_cls(planes)
        self.downsample = downsample
        self.stride = stride
        self.all_and_sub_features = all_and_sub_features
        self.last_all_and_sub_features = None

    def set_all_and_sub(self, flag):
        self.all_and_sub_features = flag

    def forward(self, x):
        residual = x
        if self.all_and_sub_features:
            part_residual, full_residual = residual
            self.last_all_and_sub_features = []
        out_1 = self.conv1(x)
        out_2 = self.bn1(out_1)
       
        if self.all_and_sub_features:
            out = ( self.relu(out_2[0]), self.relu(out_2[1]) )
            # self.last_all_and_sub_features.append(out)
            # import pdb; pdb.set_trace()
        else:
            out = self.relu(out_2)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            if self.all_and_sub_features:
                part_residual = self.downsample(x[0])
                full_residual = self.downsample(x[1])
            else:
                residual = self.downsample(x)

        if self.all_and_sub_features:
            out_ref = out[0]
            out_ref += part_residual
            
            out_ref = out[1]
            out_ref += full_residual

            self.last_all_and_sub_features.append(out)
            # print(len(self.last_all_and_sub_features), self.last_all_and_sub_features[-1][0].shape)
        else:
            out += residual

        if self.all_and_sub_features:
            part_out = self.relu(out[0])
            full_out = self.relu(out[1])

            return (part_out, full_out)
        else:
            out = self.relu(out)

            return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, cfg, stride=1, downsample=None, bn_cls=None):
        assert cfg.__class__ == list
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, cfg[0], kernel_size=1, bias=False)
        self.bn1 = bn_cls(planes)
        self.conv2 = nn.Conv2d(cfg[0], cfg[1], kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = bn_cls(planes)
        self.conv3 = nn.Conv2d(cfg[1], planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = bn_cls(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    
    def set_all_and_sub(self, flag):
        self.all_and_sub_features = flag

    def forward(self, x):
        residual = x
        if self.all_and_sub_features:
            part_residual, full_residual = residual
            self.last_all_and_sub_features = []

        out_1 = self.conv1(x)
        out_2 = self.bn1(out_1)
        out = self.relu(out_2)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            if self.all_and_sub_features:
                part_residual = self.downsample(x[0])
                full_residual = self.downsample(x[1])
            else:
                residual = self.downsample(x)

        if self.all_and_sub_features:
            out_ref = out[0]
            out_ref += part_residual
            
            out_ref = out[1]
            out_ref += full_residual

            self.last_all_and_sub_features.append(out)
            # print(len(self.last_all_and_sub_features), self.last_all_and_sub_features[-1][0].shape)
        else:
            out += residual

        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, cfg=None, num_classes=1000, sync_train = False, all_and_sub_features=False):
        assert sync_train == False, "do not support the sync bn right now"
        self.bn_cls = nn.BatchNorm2d
        self.inplanes = 64
        super(ResNet, self).__init__()
        if block == BasicBlock:
            if cfg == None:
                cfg = [[64] * layers[0], [128]*layers[1], [256]*layers[2], [512]*layers[3]]
                cfg = [item for sub_list in cfg for item in sub_list]
        elif block == Bottleneck:
            if cfg == None:
                cfg = [*[[64, 64]] * layers[0], *[[128, 128]]*layers[1], *[[256, 256]]*layers[2], *[[512, 512]]*layers[3]]
        else:
            assert 0
        self.cfg = cfg
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = self.bn_cls(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        count = 0
        self.layer1 = self._make_layer(block, 64, layers[0], cfg[:layers[0]])
        count += layers[0]
        self.layer2 = self._make_layer(block, 128, layers[1], cfg[count:count+layers[1]], stride=2)
        count += layers[1]
        self.layer3 = self._make_layer(block, 256, layers[2], cfg[count:count+layers[2]], stride=2)
        count += layers[2]
        self.layer4 = self._make_layer(block, 512, layers[3], cfg[count:count+layers[3]], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        # self.fc = nn.Linear(cfg[-1], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, self.bn_cls):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        self.set_all_and_sub(all_and_sub_features)

    
    def set_all_and_sub(self, flag):
        self.all_and_sub_features = flag
        for m in self.modules():
            if m is self.conv1 or m is self.bn1:
                continue
            if hasattr(m,'set_all_and_sub') and m is not self:
                m.set_all_and_sub(flag)

    def _make_layer(self, block, planes, blocks, cfg, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                torch.nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        # import pdb;pdb.set_trace()
        layers.append(block(self.inplanes, planes, cfg[0],stride, downsample, bn_cls = self.bn_cls ))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cfg[i], bn_cls = self.bn_cls ))

        return nn.Sequential(*layers)
    
    def gather_last_all_and_sub_features(self):
        mlist = []
        for k, m in self.named_modules():
            if hasattr(m, 'last_all_and_sub_features'):
                cur_feats = m.last_all_and_sub_features
                if cur_feats is None:
                    # import pdb;pdb.set_trace()
                    assert 0
                mlist.extend(cur_feats)
        
        return mlist
    
    def forward(self, x):
        if self.all_and_sub_features:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            
            x = (x, x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            part_x, full_x = x
            part_x = self.avgpool(part_x)
            full_x = self.avgpool(full_x)

            part_x = part_x.view(part_x.size(0), -1)
            full_x = full_x.view(full_x.size(0), -1)

            part_x = self.fc(part_x)
            full_x = self.fc(full_x)

            as_feats = self.gather_last_all_and_sub_features()

            return (part_x, full_x, as_feats)
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)

            return x

_COMMON_META = {
    "min_size": (1, 1),
    "categories": _IMAGENET_CATEGORIES,
}

class ResNet50_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/resnet50-0676ba61.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 25557032,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnet",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 76.130,
                    "acc@5": 92.862,
                }
            },
            "_ops": 4.089,
            "_file_size": 97.781,
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        },
    )
    IMAGENET1K_V2 = Weights(
        url="https://download.pytorch.org/models/resnet50-11ad3fa6.pth",
        transforms=partial(ImageClassification, crop_size=224, resize_size=232),
        meta={
            **_COMMON_META,
            "num_params": 25557032,
            "recipe": "https://github.com/pytorch/vision/issues/3995#issuecomment-1013906621",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 80.858,
                    "acc@5": 95.434,
                }
            },
            "_ops": 4.089,
            "_file_size": 97.79,
            "_docs": """
                These weights improve upon the results of the original paper by using TorchVision's `new training recipe
                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V2



def resnet34(pretrained=False, sync_train = False, strict_weight=True, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3],  sync_train = sync_train, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']), strict=strict_weight)
    return model

def _resnet(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> ResNet:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ResNet(block, layers, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress), strict=False)

    return model


def resnet50(*, pretrained=False, weights: Optional[ResNet50_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-50 from `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`__.

    .. note::
       The bottleneck of TorchVision places the stride for downsampling to the second 3x3
       convolution while the original paper places it to the first 1x1 convolution.
       This variant improves the accuracy and is known as `ResNet V1.5
       <https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch>`_.

    Args:
        weights (:class:`~torchvision.models.ResNet50_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNet50_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ResNet50_Weights
        :members:
    """
    if pretrained:
        weights = ResNet50_Weights.verify(weights)
        weights = ResNet50_Weights.IMAGENET1K_V1

    return _resnet(Bottleneck, [3, 4, 6, 3], weights, progress, **kwargs)