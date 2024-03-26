#coding=utf-8
import os
import torch
from torchvision import datasets, transforms
from nvidia.dali.plugin.pytorch import DALIGenericIterator, DALIClassificationIterator, LastBatchPolicy

from utils.utils import reset_random_seed, split_dataset
from lib.datasets import imagenet as ILSVRC_DATA
from lib.datasets import cifar as CIFAR_DATA
from lib.datasets import places365_standard_lmdb as PSL

IMAGENET_DIR = ILSVRC_DATA.IMAGENET_DIR


class DALIDataloader(DALIGenericIterator):
    def __init__(self, pipeline, size, batch_size, output_map=["data", "label"], auto_reset=True, 
                 onehot_label=False, prepare_first_batch=False, last_batch_policy = LastBatchPolicy.PARTIAL, **kwargs):
        self._size = size
        self.batch_size = batch_size
        self.onehot_label = onehot_label
        self.output_map = output_map
        super().__init__(pipelines=pipeline, size=size, auto_reset=auto_reset, output_map=output_map, \
                         prepare_first_batch=prepare_first_batch, last_batch_policy=last_batch_policy, **kwargs)

        self.dataset = [None]*self._size # create a fake dataset for external calls
         
    def __next__(self):
        if self._first_batch is not None:
            batch = self._first_batch[0]
            self._first_batch = None
            if self.onehot_label:
                return [batch[self.output_map[0]], batch[self.output_map[1]].squeeze().long()]
            else:
                return [batch[self.output_map[0]], batch[self.output_map[1]]]
        data = super().__next__()[0]
        if self.onehot_label:
            return [data[self.output_map[0]], data[self.output_map[1]].squeeze().long()]
        else:
            return [data[self.output_map[0]], data[self.output_map[1]].squeeze().long()]
    
    def __len__(self):
        if self._size%self.batch_size==0:
            return self._size//self.batch_size
        else:
            return self._size//self.batch_size+1
        
def create_prune_dataset_tv(args, prune_criterion_batch_size=None):
    kwargs = {'num_workers': args.num_workers, 'pin_memory': False} 
    train_kwargs = {'num_workers': 0, 'pin_memory': False} 

    if prune_criterion_batch_size is None:
        prune_criterion_batch_size = args.cri_batch_size
    # dataset for pruning decision
    if args.dataset == 'cifar10':
        train_set, test_set = CIFAR_DATA.create_prune_dataset_tv_cifar10(args)
    elif args.dataset == 'cifar100' :
        train_set, test_set = CIFAR_DATA.create_prune_dataset_tv_cifar100(args)
    elif args.dataset == 'imagenet':
        train_set, test_set = ILSVRC_DATA.create_prune_dataset_tv(args)
    elif args.dataset == 'places365':
        train_set, test_set = PSL.create_prune_dataset_tv(args)
    else:
        raise ValueError("No valid dataset is given.")
    
    train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=prune_criterion_batch_size, shuffle=True, **train_kwargs)
    
    test_loader = torch.utils.data.DataLoader(
        test_set,
            batch_size=args.test_batch_size, shuffle=False, **kwargs)
    
    return train_loader, test_loader


def create_evolved_dataset_tv(args):
    kwargs = {'num_workers': args.num_workers, 'pin_memory': False}

    reset_random_seed(args)
    if args.dataset == 'cifar10':
        train_set, val_set, test_set = CIFAR_DATA.create_evolved_dataset_tv_cifar10(args)
        
    elif args.dataset == 'cifar100':
        train_set, val_set, test_set = CIFAR_DATA.create_evolved_dataset_tv_cifar100(args)

    elif args.dataset == 'imagenet':
        train_set, val_set, test_set = ILSVRC_DATA.create_evolved_dataset_tv(args)
    
    elif args.dataset == 'places365':
        train_set, val_set, test_set = PSL.create_evolved_dataset_tv(args)
    else: 
        assert 0, 'dataset is not correct'

    train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=args.test_batch_size, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=args.test_batch_size, shuffle=False, **kwargs)
    
    return train_loader, val_loader, test_loader    

def create_prune_dataset_dali(args, prune_criterion_batch_size=None):
    if prune_criterion_batch_size is None:
        prune_criterion_batch_size = args.cri_batch_size
    pip_train, pip_test = None, None
    if args.dataset == 'cifar10':
        pip_train, pip_test = CIFAR_DATA.create_prune_dataset_dali_cifar10(args, prune_criterion_batch_size)
        
    elif args.dataset == 'cifar100':
        pip_train, pip_test = CIFAR_DATA.create_prune_dataset_dali_cifar100(args, prune_criterion_batch_size) 
    
    elif args.dataset == 'imagenet':
        pip_train, pip_test = ILSVRC_DATA.create_prune_dataset_dali(args, prune_criterion_batch_size) 
    elif args.dataset == 'places365':
        pip_train, pip_test = PSL.create_prune_dataset_dali(args, prune_criterion_batch_size)
    else: 
        assert 0, 'dataset is not correct'

    train_loader = DALIDataloader(pipeline=pip_train, size=len(pip_train), batch_size=prune_criterion_batch_size, onehot_label=False)
    test_loader = DALIDataloader(pipeline=pip_test, size=len(pip_test), batch_size=args.test_batch_size, onehot_label=False)

    return train_loader, test_loader

def create_evolved_dataset_dali(args):
    pip_train, pip_val, pip_test = None, None, None
    if args.dataset == 'cifar10':
        pip_train, pip_val, pip_test = CIFAR_DATA.create_evolved_dataset_dali_cifar10(args)
        
    elif args.dataset == 'cifar100':
        pip_train, pip_val, pip_test = CIFAR_DATA.create_evolved_dataset_dali_cifar100(args)
        
    elif args.dataset == 'imagenet':
        pip_train, pip_val, pip_test = ILSVRC_DATA.create_evolved_dataset_dali(args)
    elif args.dataset == 'places365':
        pip_train, pip_val, pip_test = PSL.create_evolved_dataset_dali(args)
    else: 
        assert 0, 'dataset is not correct'
    train_loader = DALIDataloader(pipeline=pip_train, size=len(pip_train), batch_size=args.batch_size, onehot_label=False)
    val_loader = DALIDataloader(pipeline=pip_val, size=len(pip_val), batch_size=args.test_batch_size, onehot_label=False)
    test_loader = DALIDataloader(pipeline=pip_test, size=len(pip_test), batch_size=args.test_batch_size, onehot_label=False)
        
    # train_loader = DALIClassificationIterator(pipelines=pip_train, last_batch_policy=LastBatchPolicy.PARTIAL)
    # val_loader = DALIClassificationIterator(pipelines=pip_val, last_batch_policy=LastBatchPolicy.PARTIAL)
    # test_loader = DALIClassificationIterator(pipelines=pip_test, last_batch_policy=LastBatchPolicy.PARTIAL)
    return train_loader, val_loader, test_loader    


def create_prune_dataset(args):
    using_dali = args.dali
    if using_dali:
        return create_prune_dataset_dali(args)
    return create_prune_dataset_tv(args)

def create_evolved_dataset(args):
    using_dali = args.dali
    if using_dali:
        return create_evolved_dataset_dali(args)
    return create_evolved_dataset_tv(args)

def create_full_train_dataset(args):
    """Create the original full training dataset
    """
    using_dali = args.dali
    if using_dali:
        return create_prune_dataset_dali(args, args.batch_size)
    return create_prune_dataset_tv(args, args.batch_size)
    