from .resnet import *
from .vgg import *
from .resnet_imagenet import resnet34, resnet50
from .mobilenetv2 import mobilenet_v2
from .googlenet import GoogLeNet
from .densenet import densenet40

from lib.datasets import get_num_cls

def create_model(args, cfg):
    if args.arch =='resnet':
        if args.depth == 56:
            model = resnet(depth=args.depth, dataset=args.dataset, cfg=cfg)
        elif args.depth == 34:
            model = resnet34(pretrained=False, strict_weight=False, num_classes = get_num_cls(args), cfg=cfg)
            # assert 0
        elif args.depth == 50:
            # model = resnet50(pretrained=True)
            assert 0
    elif args.arch == 'vgg':
        model = vgg(depth=args.depth, dataset=args.dataset, cfg=cfg)
    elif args.arch == 'mobilenetv2':
        model = mobilenet_v2(pretrained=False, num_classes = get_num_cls(args), cfg=cfg)
    elif args.arch == 'inception':
        model = GoogLeNet(num_classes = get_num_cls(args), cfg=cfg)
    elif args.arch == 'densenet':
        if args.depth == 40:
            model = densenet40(num_classes = get_num_cls(args), cfg=cfg)
    else:
        assert 0, f'donot supprt this kind of architecture:{args.arch}'

    return model