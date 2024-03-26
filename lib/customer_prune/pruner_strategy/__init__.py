from .gn import *
from .mbv2 import *
from .r34 import *
from .r56 import *
from .vgg import *
from .dn import *


def get_pruning_strategy(args):
    arch = args.arch
    depth = args.depth
    # import pdb;pdb.set_trace()
    pruning_strategy = None
    if arch == 'resnet':
        if depth == 56:
            pruning_strategy = ResNet56Strategy(args)
        elif depth == 34:
            pruning_strategy = ResNet34Strategy(args)
    elif arch == 'vgg':
        # if depth == 16:
        pruning_strategy = VGGStrategy(args)
    elif arch == 'inception':
        pruning_strategy = GoogLeNetStrategy(args)
    elif arch == 'mobilenetv2':
        pruning_strategy = MobileNetV2Strategy(args)
    elif arch == 'densenet':
        if depth == 40:
            pruning_strategy = DenseNet40Strategy(args)
    assert pruning_strategy is not None 
    return pruning_strategy