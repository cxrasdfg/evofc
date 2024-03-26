#coding=utf-8
import pickle
import os, sys
import multiprocessing

import numpy as np
from sklearn.utils import shuffle
import torch as th
from torchvision import datasets, transforms
import nvidia.dali.ops as ops
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.types as types

from utils.utils import create_disjoint_indices, split_dataset as split_dataset_tv
from lib.datasets.lmdb.caffe_lmdb import CaffeLMDBDataset

PLACES365_LMDB_DIR = os.path.expanduser('~/work/datasets/lmdb_places365standard') 
PLACES365_DIR = os.path.expanduser('~/work/datasets/places365_standard')

MEAN_TV = [0.485, 0.456, 0.406]
STD_TV = [0.229, 0.224, 0.225]
MEAN_DALI = th.tensor(MEAN_TV) * 255
STD_DALI = th.tensor(STD_TV) * 255
VAL_SIZE = 256
CROP_SIZE = 224

TRAIN_TRANS_TV = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN_TV,
                                    std=STD_TV),
                ])

TEST_TRANS_TV = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN_TV,
                                     std=STD_TV),
                ])

class Places365StandardLMDB(CaffeLMDBDataset):
    num_classes = 365
    def __init__(self, lmdb_path, transform=None):
        super().__init__(lmdb_path, transform)

    
class HybridTrainPipeLMDB(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, num_sample, dali_cpu=False, local_rank=0, world_size=1, file_list = None):
        super(HybridTrainPipeLMDB, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        dali_device = "gpu" # can change to "cpu" if using too much vram.
        if dali_device == 'cpu':
            image_decoder = 'cpu'
        else:
            image_decoder = 'mixed'
        assert file_list is None, 'DALI caffe api donot support file_list'
        self.input = ops.readers.Caffe(path=data_dir, shard_id=local_rank, num_shards=world_size, random_shuffle=True)
        self.decode = ops.decoders.Image(device=image_decoder, output_type=types.RGB)
        self.res = ops.RandomResizedCrop(device=dali_device, size=crop, random_area=[0.08, 1.25])
        self.cmnp = ops.CropMirrorNormalize(device=dali_device,
                                            dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            # image_type=types.RGB,
                                            mean=MEAN_DALI,
                                            std=STD_DALI)
        self.coin = ops.random.CoinFlip(probability=0.5)
        print('DALI "{0}" variant'.format(dali_device))
        self.num_sample = num_sample

    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images, mirror=rng)
        return [output, self.labels]

    def __len__(self):
        return self.num_sample

class HybridTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, num_sample, dali_cpu=False, local_rank=0, world_size=1, file_list = None):
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        dali_device = "gpu" # can change to "cpu" if using too much vram.
        if dali_device == 'cpu':
            image_decoder = 'cpu'
        else:
            image_decoder = 'mixed'
        self.input = ops.readers.File(file_root=data_dir, shard_id=local_rank, num_shards=world_size, random_shuffle=True, file_list=file_list)
        self.decode = ops.decoders.Image(device=image_decoder, output_type=types.RGB)
        self.res = ops.RandomResizedCrop(device=dali_device, size=crop, random_area=[0.08, 1.25])
        self.cmnp = ops.CropMirrorNormalize(device=dali_device,
                                            dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            # image_type=types.RGB,
                                            mean=MEAN_DALI,
                                            std=STD_DALI)
        self.coin = ops.random.CoinFlip(probability=0.5)
        print('DALI "{0}" variant'.format(dali_device))
        self.num_sample = num_sample

    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images, mirror=rng)
        return [output, self.labels]

    def __len__(self):
        return self.num_sample

class HybridValPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, size, num_sample, local_rank=0, world_size=1, file_list = None):
        dali_device = "gpu"
        if dali_device == 'cpu':
            image_decoder = 'cpu'
        else:
            image_decoder = 'mixed'

        super(HybridValPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        self.input = ops.readers.File(file_root=data_dir, shard_id=local_rank, num_shards=world_size,
                                    random_shuffle=False, file_list=file_list)
        self.decode = ops.decoders.Image(device=image_decoder, output_type=types.RGB)
        self.res = ops.Resize(device=dali_device, resize_shorter=size, interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device=dali_device,
                                            dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            # image_type=types.RGB,
                                            mean=MEAN_DALI,
                                            std=STD_DALI)

        self.num_sample = num_sample

    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images)
        return [output, self.labels]
    
    def __len__(self):
        return self.num_sample

class HybridValPipeLMDB(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, size, num_sample, local_rank=0, world_size=1, file_list = None):
        dali_device = "gpu"
        if dali_device == 'cpu':
            image_decoder = 'cpu'
        else:
            image_decoder = 'mixed'
        assert file_list is None, 'DALI caffe api donot support file_list'

        super(HybridValPipeLMDB, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        self.input = ops.readers.Caffe(path=data_dir, shard_id=local_rank, num_shards=world_size,
                                    random_shuffle=False)
        self.decode = ops.decoders.Image(device=image_decoder, output_type=types.RGB)
        self.res = ops.Resize(device=dali_device, resize_shorter=size, interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device=dali_device,
                                            dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            # image_type=types.RGB,
                                            mean=MEAN_DALI,
                                            std=STD_DALI)

        self.num_sample = num_sample

    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images)
        return [output, self.labels]
    
    def __len__(self):
        return self.num_sample
    
def scan_dir_func(args):
    idx, root_dir, img_dir  = args
    img_dir = os.path.join(root_dir, img_dir)
    if os.path.isdir(img_dir):
        return os.listdir(img_dir), idx
    return None 

def filter_dir(dir_list, root_path):
    new_dir_list = []
    for dir_name in dir_list:
        if os.path.isdir(os.path.join(root_path, dir_name)):
            new_dir_list.append(dir_name)
    return new_dir_list

def scan_ilsvrc_dir(args, num_process = 4):
    """
    scan the directories and return the following list like:
    [ ('dog.jpg', 0),
      ('cute kitten.jpg' 1),
      ('doge.png', 0)]
    """

    train_file_list = []
    test_file_list = []

    trn_dir = os.path.join(PLACES365_DIR, 'train')
    val_dir = os.path.join(PLACES365_DIR, 'val')
    trn_cls_dirs = sorted(os.listdir(trn_dir), reverse=False)
    val_cls_dirs = sorted(os.listdir(val_dir), reverse=False)

    trn_cls_dirs = filter_dir(trn_cls_dirs, trn_dir)
    val_cls_dirs = filter_dir(val_cls_dirs, val_dir)

    # import pdb;pdb.set_trace()
    id_to_cls_name = {}
    for idx, cls_name in enumerate(trn_cls_dirs):
        id_to_cls_name[idx] = cls_name

    def combine_to_list1(list1, list2):
        for v in list2:
            if v is not None:
                v1, idx = v
                for v2 in v1:
                    list1.append((os.path.join(id_to_cls_name[idx], v2), idx))

    with multiprocessing.Pool(num_process) as pool:
        temp_list = pool.map(scan_dir_func, [ (i, trn_dir, img_dir) for i, img_dir in enumerate(trn_cls_dirs)])
    combine_to_list1(train_file_list, temp_list)

    with multiprocessing.Pool(num_process) as pool:
        temp_list = pool.map(scan_dir_func, [ (i, val_dir, img_dir) for i, img_dir in enumerate(val_cls_dirs)])
    combine_to_list1(test_file_list, temp_list)

    return train_file_list, test_file_list

def write_to_file(list1, fpath):
    """
    Example::

      dog.jpg 0
      cute kitten.jpg 1
      doge.png 0
    """
    with open(fpath,'w+') as f:
        for i, (img_name, label) in enumerate(list1):
            if i == len(list1) - 1:
                f.write(f'{img_name} {label}')
            else:
                f.write(f'{img_name} {label}\n')

def split_dataset(args):
    exp_root = args.save
    train_val_ratio = args.train_val_ratio
    train_file_save = os.path.join(exp_root,'ilsvrc_train.txt')
    val_file_save = os.path.join(exp_root,'ilsvrc_val.txt')
    
    trn_list, tst_list = scan_ilsvrc_dir(args)
    num_tst = len(tst_list)
    trn_list = np.array(trn_list)

    total_trn_sample = len(trn_list)
    idx_train, idx_val = create_disjoint_indices(total_trn_sample, train_val_ratio, True, keep_same_seed=args.seed)

    slc_trn_list  = trn_list[idx_train]
    slc_val_list = trn_list[idx_val]

    write_to_file(slc_trn_list, train_file_save)
    write_to_file(slc_val_list, val_file_save)

    num_trn = len(slc_trn_list)
    num_val = len(slc_val_list)

    return (num_trn, num_val, num_tst), (train_file_save, val_file_save)
    
def create_prune_dataset_dali_lmdb(args, prune_criterion_batch_size = None):
    if prune_criterion_batch_size is None:
        prune_criterion_batch_size = args.cri_batch_size
    # assert 0, "dali is not supported now"
    trn_dir = os.path.join(PLACES365_LMDB_DIR, 'train_image_lmdb')
    val_dir = os.path.join(PLACES365_LMDB_DIR, 'val_image_lmdb')
    # r1, r2 = scan_ilsvrc_dir(args)
    num_trn = CaffeLMDBDataset(trn_dir,None).__len__()
    num_tst = CaffeLMDBDataset(val_dir,None).__len__()

    pip_train = HybridTrainPipeLMDB(batch_size=prune_criterion_batch_size, num_threads=args.num_workers, device_id=0, data_dir=trn_dir, \
                                crop=CROP_SIZE, world_size=1, local_rank=0, num_sample=num_trn)
    pip_test = HybridValPipeLMDB(batch_size=args.test_batch_size, num_threads=args.num_workers, device_id=0, data_dir=val_dir, \
                             crop=CROP_SIZE, size=VAL_SIZE, world_size=1, local_rank=0, num_sample= num_tst)

    return pip_train, pip_test

def create_evolved_dataset_dali_lmdb(args):
    assert 0, 'dali is not support file_list for caffe'
    trn_dir = os.path.join(PLACES365_LMDB_DIR, 'train_image_lmdb')
    (num_trn, num_val, num_tst), (train_file_save, val_file_save) = split_dataset(args)
    
    pip_train = HybridTrainPipeLMDB(batch_size=args.batch_size, num_threads=args.num_workers, device_id=0, data_dir=trn_dir, \
                                crop=CROP_SIZE, world_size=1, local_rank=0, num_sample=num_trn, file_list=train_file_save)
    pip_val = HybridValPipeLMDB(batch_size=args.test_batch_size, num_threads=args.num_workers, device_id=0, data_dir=trn_dir, \
                             crop=CROP_SIZE, size=VAL_SIZE, world_size=1, local_rank=0, num_sample= num_val, file_list=val_file_save)
    
    tst_dir = os.path.join(PLACES365_LMDB_DIR, 'val_image_lmdb')
    pip_test = HybridValPipeLMDB(batch_size=args.test_batch_size, num_threads=args.num_workers, device_id=0, data_dir=tst_dir, \
                             crop=CROP_SIZE, size=VAL_SIZE, world_size=1, local_rank=0, num_sample= num_tst)
    return pip_train, pip_val, pip_test


def create_prune_dataset_tv_lmdb(args):
    trn_dir = os.path.join(PLACES365_LMDB_DIR, 'train_image_lmdb')
    val_dir = os.path.join(PLACES365_LMDB_DIR, 'val_image_lmdb')
    
    train_dataset = Places365StandardLMDB(trn_dir, TRAIN_TRANS_TV)
    test_dataset = Places365StandardLMDB(val_dir, TEST_TRANS_TV)

    return train_dataset, test_dataset

def create_evolved_dataset_tv_lmdb(args):
    trn_dir = os.path.join(PLACES365_LMDB_DIR, 'train_image_lmdb')
    val_dir = os.path.join(PLACES365_LMDB_DIR, 'val_image_lmdb')

    train_val_ratio = args.train_val_ratio

    train_dataset = Places365StandardLMDB(trn_dir, TRAIN_TRANS_TV)
    val_dataset = Places365StandardLMDB(trn_dir, TEST_TRANS_TV)
    test_dataset = Places365StandardLMDB(val_dir, TEST_TRANS_TV)

    train_dataset, val_dataset = split_dataset_tv(train_dataset, val_dataset, train_val_ratio, shuffle=True)
    
    return train_dataset, val_dataset, test_dataset

def create_prune_dataset_tv(args):
    trn_dir = os.path.join(PLACES365_DIR, 'train')
    val_dir = os.path.join(PLACES365_DIR, 'val')
    
    train_dataset = datasets.ImageFolder(trn_dir, TRAIN_TRANS_TV)
    test_dataset = datasets.ImageFolder(val_dir, TEST_TRANS_TV)

    return train_dataset, test_dataset

def create_evolved_dataset_tv(args):
    trn_dir = os.path.join(PLACES365_DIR, 'train')
    val_dir = os.path.join(PLACES365_DIR, 'val')

    train_val_ratio = args.train_val_ratio

    train_dataset = datasets.ImageFolder(trn_dir, TRAIN_TRANS_TV)
    val_dataset = datasets.ImageFolder(trn_dir, TEST_TRANS_TV)
    test_dataset = datasets.ImageFolder(val_dir, TEST_TRANS_TV)

    train_dataset, val_dataset = split_dataset_tv(train_dataset, val_dataset, train_val_ratio, shuffle=True)
    
    return train_dataset, val_dataset, test_dataset

def create_prune_dataset_dali(args, prune_criterion_batch_size = None):
    if prune_criterion_batch_size is None:
        prune_criterion_batch_size = args.cri_batch_size
    trn_dir = os.path.join(PLACES365_DIR, 'train')
    val_dir = os.path.join(PLACES365_DIR, 'val')
    r1, r2 = scan_ilsvrc_dir(args)
    
    num_trn = len(r1)
    num_tst = len(r2)
    # import pdb;pdb.set_trace()
    pip_train = HybridTrainPipe(batch_size=prune_criterion_batch_size, num_threads=args.num_workers, device_id=0, data_dir=trn_dir, \
                                crop=CROP_SIZE, world_size=1, local_rank=0, num_sample=num_trn)
    pip_test = HybridValPipe(batch_size=args.test_batch_size, num_threads=args.num_workers, device_id=0, data_dir=val_dir, \
                             crop=CROP_SIZE, size=VAL_SIZE, world_size=1, local_rank=0, num_sample= num_tst)

    return pip_train, pip_test

def create_evolved_dataset_dali(args):
    trn_dir = os.path.join(PLACES365_DIR, 'train')
    (num_trn, num_val, num_tst), (train_file_save, val_file_save) = split_dataset(args)
    
    pip_train = HybridTrainPipe(batch_size=args.batch_size, num_threads=args.num_workers, device_id=0, data_dir=trn_dir, \
                                crop=CROP_SIZE, world_size=1, local_rank=0, num_sample=num_trn, file_list=train_file_save)
    pip_val = HybridValPipe(batch_size=args.test_batch_size, num_threads=args.num_workers, device_id=0, data_dir=trn_dir, \
                             crop=CROP_SIZE, size=VAL_SIZE, world_size=1, local_rank=0, num_sample= num_val, file_list=val_file_save)
    
    tst_dir = os.path.join(PLACES365_DIR, 'val')
    pip_test = HybridValPipe(batch_size=args.test_batch_size, num_threads=args.num_workers, device_id=0, data_dir=tst_dir, \
                             crop=CROP_SIZE, size=VAL_SIZE, world_size=1, local_rank=0, num_sample= num_tst)
    return pip_train, pip_val, pip_test