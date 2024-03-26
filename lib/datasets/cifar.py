#coding=utf-8
import pickle
import os, sys

import numpy as np
from sklearn.utils import shuffle
import torch as th
from torchvision import datasets, transforms
import nvidia.dali.ops as ops
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.types as types

from utils.utils import create_disjoint_indices, reset_random_seed, split_dataset

CIFAR10_DIR = os.path.expanduser('~/work/datasets/cifar10')
CIFAR100_DIR = os.path.expanduser('~/work/datasets/cifar100')

MEAN_TV = [0.4914, 0.4822, 0.4465]
STD_TV = [0.2023, 0.1994, 0.2010]
MEAN_DALI = th.tensor(MEAN_TV) * 255
STD_DALI = th.tensor(STD_TV) * 255
CROP_SIZE=32

TRAIN_TRANS_TV = transforms.Compose([
                            transforms.Pad(4),
                            transforms.RandomCrop(CROP_SIZE),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize(MEAN_TV, STD_TV)
                        ])

TEST_TRANS_TV = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(MEAN_TV, STD_TV)])



class HybridTrainPipe_CIFAR(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop=32, iterator=None, dali_cpu=False, local_rank=0,
                 world_size=1,
                 cutout=0):
        super(HybridTrainPipe_CIFAR, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        if iterator is not None:
            self.iterator = iter(iterator)
        else:
            assert 0, 'requires iterator'
            # self.iterator = iter(CIFAR_INPUT_ITER(batch_size, 'train', root=data_dir))
        dali_device = "gpu"
        self.input = ops.ExternalSource()
        self.input_label = ops.ExternalSource()
        self.pad = ops.Paste(device=dali_device, ratio=1.25, fill_value=0)
        self.uniform = ops.random.Uniform(range=(0., 1.))
        self.crop = ops.Crop(device=dali_device, crop_h=crop, crop_w=crop)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            # image_type=types.RGB,
                                            mean=MEAN_DALI,
                                            std=STD_DALI
                                            )
        self.coin = ops.random.CoinFlip(probability=0.5)

    def iter_setup(self):
        (images, labels) = self.iterator.next()
        self.feed_input(self.jpegs, images, layout="HWC")
        self.feed_input(self.labels, labels)

    def define_graph(self):
        rng = self.coin()
        self.jpegs = self.input()
        self.labels = self.input_label()
        output = self.jpegs
        output = self.pad(output.gpu())
        output = self.crop(output, crop_pos_x=self.uniform(), crop_pos_y=self.uniform())
        output = self.cmnp(output, mirror=rng)
        return [output, self.labels]

    def __len__(self):
        return len(self.iterator.data)

class HybridTestPipe_CIFAR(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, size, iterator, local_rank=0, world_size=1):
        super(HybridTestPipe_CIFAR, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        if iterator is None:
            assert 0, 'requires iterator'
            # self.iterator = iter(CIFAR_INPUT_ITER(batch_size, 'val', root=data_dir))
        else:
            self.iterator = iter(iterator)
        self.input = ops.ExternalSource()
        self.input_label = ops.ExternalSource()
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            # image_type=types.RGB,
                                            mean=MEAN_DALI,
                                            std=STD_DALI,
                                            )

    def iter_setup(self):
        (images, labels) = self.iterator.next()
        self.feed_input(self.jpegs, images, layout="HWC")  # can only in HWC order
        self.feed_input(self.labels, labels)

    def define_graph(self):
        self.jpegs = self.input()
        self.labels = self.input_label()
        output = self.jpegs
        output = self.cmnp(output.gpu())
        return [output, self.labels]

    def __len__(self):
        return len(self.iterator.data)

class CIFAR10_INPUT_ITER():
    base_folder = 'cifar-10-batches-py'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]

    def __init__(self, batch_size, type='train', root='./data.cifar10', indices = None):
        self.root = root
        self.batch_size = batch_size

        self.train = (type == 'train')
        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []
        for file_name, checksum in downloaded_list:
            
            file_path = os.path.join(self.root, self.base_folder, file_name)
            # file_path = os.path.join(self.root, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.targets = np.vstack(self.targets)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        if indices is not None:
            self.data = self.data[indices]
            self.targets = self.targets[indices].copy()
        self.data = self.data.copy() # contiguous
        # np.save("cifar.npy", self.data)
        # self.data = np.load('cifar.npy')  # to serialize, increase locality????
        # os.remove('cifar.npy')
        # self.i = 0
        # self.n = 0

    def __iter__(self):
        self.i = 0
        self.n = len(self.data)
        return self

    def __next__(self):
        batch = []
        labels = []
        for _ in range(self.batch_size):
            if self.train and self.i % self.n == 0:
                self.data, self.targets = shuffle(self.data, self.targets, random_state=0)
            img, label = self.data[self.i], self.targets[self.i]
            batch.append(img)
            labels.append(label)
            self.i = (self.i + 1) % self.n
        # print(labels)
        return (batch, labels)

    next = __next__

class CIFAR100_INPUT_ITER():
    base_folder = 'cifar-100-python'
    train_list = [
        ['train', None],
    ]

    test_list = [
        ['test', None],
    ]

    def __init__(self, batch_size, type='train', root='./data.cifar100', indices = None):
        self.root = root
        self.batch_size = batch_size

        self.train = (type == 'train')
        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []
        for file_name, checksum in downloaded_list:
            
            file_path = os.path.join(self.root, self.base_folder, file_name)
            # file_path = os.path.join(self.root, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.targets = np.vstack(self.targets)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        if indices is not None:
            self.data = self.data[indices]
            self.targets = self.targets[indices].copy()
        self.data = self.data.copy() # contiguous


    def __iter__(self):
        self.i = 0
        self.n = len(self.data)
        return self

    def __next__(self):
        batch = []
        labels = []
        for _ in range(self.batch_size):
            if self.train and self.i % self.n == 0:
                self.data, self.targets = shuffle(self.data, self.targets, random_state=0)
            img, label = self.data[self.i], self.targets[self.i]
            batch.append(img)
            labels.append(label)
            self.i = (self.i + 1) % self.n
        # print(labels)
        return (batch, labels)

    next = __next__

def cifar_prune_dataset_builder(args, data_root, prune_criterion_batch_size = None):
    if prune_criterion_batch_size is None:
        prune_criterion_batch_size = args.cri_batch_size
    if args.dataset == 'cifar10':
        train_iter = CIFAR10_INPUT_ITER(prune_criterion_batch_size, 'train', data_root)
        test_iter = CIFAR10_INPUT_ITER(args.test_batch_size, 'test', data_root)

    elif args.dataset == 'cifar100':
        train_iter = CIFAR100_INPUT_ITER(prune_criterion_batch_size, 'train', data_root)
        test_iter = CIFAR100_INPUT_ITER(args.test_batch_size, 'test', data_root)
    else:
        assert 0

    pip_train = HybridTrainPipe_CIFAR(batch_size=prune_criterion_batch_size, num_threads=args.num_workers, device_id=0, iterator=train_iter,
                                      data_dir=data_root, crop=CROP_SIZE, world_size=1, local_rank=0, cutout=0)
    pip_test = HybridTestPipe_CIFAR(batch_size=args.test_batch_size, num_threads=args.num_workers, device_id=0, iterator=test_iter,
                                    data_dir=data_root, crop=CROP_SIZE, size=CROP_SIZE, world_size=1, local_rank=0)
    # train_loader = DALIDataloader(pipeline=pip_train, size=CIFAR_IMAGES_NUM_TRAIN, batch_size=TRAIN_BS, onehot_label=True)
    # test_loader = DALIDataloader(pipeline=pip_test, size=CIFAR_IMAGES_NUM_TEST, batch_size=TEST_BS, onehot_label=True)

    return pip_train, pip_test

def create_prune_dataset_dali_cifar10(args, prune_criterion_batch_size = None):
    if prune_criterion_batch_size is None:
        prune_criterion_batch_size = args.cri_batch_size
    return cifar_prune_dataset_builder(args, CIFAR10_DIR, prune_criterion_batch_size)

def create_prune_dataset_dali_cifar100(args, prune_criterion_batch_size = None):
    if prune_criterion_batch_size is None:
        prune_criterion_batch_size = args.cri_batch_size
    return cifar_prune_dataset_builder(args, CIFAR100_DIR, prune_criterion_batch_size)

def create_evolved_dataset_dali_cifar10(args):
    data_root = CIFAR10_DIR
    for_num_dataset = CIFAR10_INPUT_ITER(args.batch_size, 'train', data_root)
    num_train_data = len(for_num_dataset.data)
    idx_train, idx_val = create_disjoint_indices(num_train_data, args.train_val_ratio, True, keep_same_seed=args.seed)
    pip_train_iter = CIFAR10_INPUT_ITER(args.batch_size, 'train', data_root, idx_train) 
    pip_val_iter = CIFAR10_INPUT_ITER(args.test_batch_size, 'train', data_root, idx_val) 
    pip_test_iter = CIFAR10_INPUT_ITER(args.test_batch_size, 'test', data_root)

    pip_train = HybridTrainPipe_CIFAR(batch_size=args.batch_size, num_threads=args.num_workers, device_id=0, 
                                      data_dir=data_root, crop=CROP_SIZE, world_size=1, local_rank=0, cutout=0, iterator=pip_train_iter)
    pip_val = HybridTestPipe_CIFAR(batch_size=args.test_batch_size, num_threads=args.num_workers, device_id=0,
                                    data_dir=data_root, crop=CROP_SIZE, size=CROP_SIZE, world_size=1, local_rank=0, iterator=pip_val_iter)
    pip_test = HybridTestPipe_CIFAR(batch_size=args.test_batch_size, num_threads=args.num_workers, device_id=0, iterator=pip_test_iter,
                                    data_dir=data_root, crop=CROP_SIZE, size=CROP_SIZE, world_size=1, local_rank=0)
    
    return pip_train, pip_val, pip_test


def create_evolved_dataset_dali_cifar100(args):
    prune_batch_size = args.cri_batch_size
    data_root = CIFAR100_DIR
    for_num_dataset = CIFAR100_INPUT_ITER(prune_batch_size, 'train', data_root)
    num_train_data = len(for_num_dataset.data)
    idx_train, idx_val = create_disjoint_indices(num_train_data, args.train_val_ratio, True, keep_same_seed=args.seed)
    pip_train_iter = CIFAR100_INPUT_ITER(args.batch_size, 'train', data_root, idx_train) 
    pip_val_iter = CIFAR100_INPUT_ITER(args.test_batch_size, 'train', data_root, idx_val) 
    pip_test_iter = CIFAR100_INPUT_ITER(args.test_batch_size, 'test', data_root)
    
    pip_train = HybridTrainPipe_CIFAR(batch_size=args.batch_size, num_threads=args.num_workers, device_id=0, 
                                      data_dir=data_root, crop=CROP_SIZE, world_size=1, local_rank=0, cutout=0, iterator=pip_train_iter)
    pip_val = HybridTestPipe_CIFAR(batch_size=args.test_batch_size, num_threads=args.num_workers, device_id=0,
                                    data_dir=data_root, crop=CROP_SIZE, size=CROP_SIZE, world_size=1, local_rank=0, iterator=pip_val_iter)
    pip_test = HybridTestPipe_CIFAR(batch_size=args.test_batch_size, num_threads=args.num_workers, device_id=0, iterator=pip_test_iter,
                                    data_dir=data_root, crop=CROP_SIZE, size=CROP_SIZE, world_size=1, local_rank=0)
    
    return pip_train, pip_val, pip_test

def create_prune_dataset_tv_cifar10(args):
    train_set = datasets.CIFAR10(CIFAR10_DIR, train=True, download=True,
                        transform=TRAIN_TRANS_TV)
    test_set = datasets.CIFAR10(CIFAR10_DIR, train=False, transform=TEST_TRANS_TV)

    return train_set, test_set

def create_prune_dataset_tv_cifar100(args):
    train_set = datasets.CIFAR100(CIFAR100_DIR, train=True, download=True,
                        transform=TRAIN_TRANS_TV) 
    test_set = datasets.CIFAR100(CIFAR100_DIR, train=False, transform=TEST_TRANS_TV)
    
    return train_set, test_set

def create_evolved_dataset_tv_cifar10(args):
    train_dataset_1 = datasets.CIFAR10(CIFAR10_DIR, train=True, download=True,
                        transform=TRAIN_TRANS_TV)
    train_dataset_2 = datasets.CIFAR10(CIFAR10_DIR, train=True, download=True,
                    transform=TEST_TRANS_TV)

    test_set = datasets.CIFAR10(CIFAR10_DIR, train=False, transform=TEST_TRANS_TV)

    train_set, val_set = split_dataset(train_dataset_1, train_dataset_2, args.train_val_ratio)

    return train_set, val_set, test_set

def create_evolved_dataset_tv_cifar100(args):
    train_dataset_1 = datasets.CIFAR100(CIFAR100_DIR, train=True, download=True,
                        transform=TRAIN_TRANS_TV)

    train_dataset_2 = datasets.CIFAR100(CIFAR100_DIR, train=True, download=True,
                    transform=TEST_TRANS_TV)

    test_set = datasets.CIFAR100(CIFAR100_DIR, train=False, transform=TEST_TRANS_TV)

    train_set, val_set = split_dataset(train_dataset_1, train_dataset_2, args.train_val_ratio)
    
    return train_set, val_set, test_set