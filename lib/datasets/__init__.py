from .dataloaders  import create_evolved_dataset, create_prune_dataset

def get_num_cls(args):
    dataset = args.dataset
    if dataset == 'cifar10':
        return 10
    elif dataset == 'cifar100':
        return 100
    elif dataset == 'imagenet':
        return 1000
    elif dataset == 'places365':
        return 365
    else:
        assert 0, f'Unknown dataset: {dataset}'