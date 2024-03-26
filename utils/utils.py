# coding=utf-8 

from io import RawIOBase
from logging import root
import os, sys, time, subprocess
import socket 
import datetime
import random
import filelock
import json
from ast import literal_eval
from collections import OrderedDict

import numpy as np
import pymoo
import torch as th
from torch.functional import block_diag
from torch.utils.data import Subset
from pymoo.rand.random import seed as pymoo_seed

from torch._C import _get_cudnn_allow_tf32


from utils.nv_parser import parse_nvidia_smi

def reset_random_seed_no_args(seed):
    random.seed(seed)
    np.random.seed(seed)
    pymoo_seed(seed)
    th.manual_seed(seed)
    th.random.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic=True

def reset_random_seed(args):
    reset_random_seed_no_args(args.seed)

def create_exp_dir(root_name, exp_name, save_name, args=None):
    mname = socket.gethostname()
    timestamp = str(datetime.datetime.now().timestamp())
    exp_name = mname + '-' + exp_name 

    dir_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
                             root_name, exp_name +'-'+ save_name + '-' + timestamp)
    
    if not os.path.exists(dir_path):
        oldmask = os.umask(000)
        os.makedirs(dir_path, mode=0o777, exist_ok=True)
        os.umask(oldmask)
    
    if args is not None:
        args_path = os.path.join(dir_path, 'args.config')
        with open(args_path, 'w+') as f:
            for k, v in vars(args).items():
                f.write(f'{k}={str(v)}\n')

    return dir_path

def args_dict_pair_to_str(argv_dict):
    res = ''
    for k, v in argv_dict.items():
        if v is not None:
            res += f'{k} {v} '
        else:
            res += f'{k} '
    return res

def get_sys_argv(argv_list = sys.argv):
    argv_dict = OrderedDict()
    argv_list = argv_list[1:] # discard the file name
    state=0 # get the key
    # state=1 # get the val
    last_in = None
    for v in argv_list:
        if state == 0:
            if v.startswith('--'):
                last_in = v
                state = 1
            else:
                assert 0, 'state error'
        else:
            if v.startswith('--'):
                argv_dict[last_in] = None
                last_in = v
            else:
                argv_dict[last_in] = v
                last_in = v
                state = 0

    if state == 1:
        argv_dict[last_in] = None
    return argv_dict
    
class GlobalNetRecorder(object):
    
    def __init__(self, root, fname='net') -> None:
        self.root = root
        self.fname = fname

    def get_file_path(self):
        # return os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), GlobalNetRecorder.fname)
        return os.path.join(self.root, self.fname+'.records')


    def read_all(self):
        fpath = self.get_file_path()
        with filelock.FileLock(fpath+'.lock'):
            try:
                res = None
                with open(fpath,'r') as f:
                    raw_content = f.read()
                    res = json.loads(raw_content)
            except FileNotFoundError as e:
                res = None
            except json.JSONDecodeError as e:
                res = None
        return res
            

    def write_dict(self, res):
        fpath = self.get_file_path()
        with filelock.FileLock(fpath+'.lock'):
            with open(fpath,'w') as f:
                res_str = json.dumps(res)
                f.write(res_str)


    def read(self, key, ** extra_keys):
        if isinstance(key, np.ndarray):
            key = np.array2string(key,threshold=sys.maxsize,floatmode='fixed',precision=6,max_line_width=sys.maxsize,separator=',')
        
        for k,v in extra_keys.items():
            key += '|'+str(k)+':' + str(v) + '|' 

        fpath = self.get_file_path()
        with filelock.FileLock(fpath+'.lock'):
            try:
                with open(fpath,'r') as f:
                    raw_content = f.read()
                    try:
                        res = json.loads(raw_content)
                        if key in res:
                            res = res[key]
                        else:
                            res = None
                    except json.JSONDecodeError as e:
                        res = None

            except FileNotFoundError as e:
                res = None
        
        return res
    
    def write(self, key, val, **extra_keys):
        if isinstance(key, np.ndarray):
            key = np.array2string(key,threshold=sys.maxsize,floatmode='fixed',precision=6,max_line_width=sys.maxsize,separator=',')

        for k,v in extra_keys.items():
            key += '|'+str(k)+':' + str(v) + '|' 

        fpath = self.get_file_path()
        with filelock.FileLock(fpath+'.lock'):
            if os.path.exists(fpath):
                with open(fpath,'r') as f:
                    raw_content = f.read()
                    try:
                        res = json.loads(raw_content)
                    except json.JSONDecodeError as e:
                        res = {}
            else:
                res = {}
            res[key] = val

            with open(fpath,'w') as f:
                res_str = json.dumps(res)
                f.write(res_str)

def create_disjoint_indices(num_data, train_val_ratio, shuffle, keep_same_seed=None):
    train_num = int(train_val_ratio / (train_val_ratio + 1) * num_data)
    # valid_size = int(num_data * valid_ratio)
    all_idx = np.arange(num_data)

    if shuffle:
        if keep_same_seed is not None:
            reset_random_seed_no_args(keep_same_seed)
        all_idx = np.random.permutation(all_idx)
    # idx_train = all_idx[valid_size:]
    # idx_val = all_idx[:valid_size]
    idx_train = all_idx[:train_num]
    idx_val = all_idx[train_num:]

    return idx_train, idx_val

def split_dataset(d1, d2, train_val_ratio, shuffle=True):
    """
    Split the dataset by the specified number of data per class
    :param d1:
    :param d2:
    :param train_val_ratio: ratio between training samples number and validation samples number
    :param shuffle:
    :return subset1: contains the `1 - valid_ratio` data in `d1`
    :return subset2: contains the data which is exclusive of `subset1`
    """
    assert len(d1) == len(d2), 'the two dataset must be consistent'
    num_data = len(d1)

    idx_train, idx_val = create_disjoint_indices(num_data, train_val_ratio, shuffle)

    subset1 = Subset(d1, idx_train)
    subset2 = Subset(d2, idx_val)

    return subset1, subset2

def parse_cuda_visible_devices(cuda_visible_devices):
    if cuda_visible_devices is not None:
        cuda_visible_devices = cuda_visible_devices.split(',')
        temp_list = []
        for x in cuda_visible_devices:
            temp_list.append(int(x))
        cuda_visible_devices = temp_list

    return cuda_visible_devices
    
class GPUTool(object):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        cuda_visible_devices = args.cuda_visible_devices

        self.cuda_visible_devices = parse_cuda_visible_devices(cuda_visible_devices)

    def get_gpus(self):
        nvidia_str = os.popen('nvidia-smi -q -x').read()
        # nvidia_str = g.decode('utf-8')
        parse_res = parse_nvidia_smi(nvidia_str)

        gpu_dict = []
        for i, res in enumerate(parse_res):
            gpu_id = res['gpu_id']
            free_mem = res['memory']['free_memory']
            free_mem = free_mem[0:len(free_mem)-3]
            free_mem = int(free_mem)
            if self.cuda_visible_devices is not None:
                if gpu_id not in self.cuda_visible_devices:
                    continue
            gpu_dict.append([gpu_id, free_mem])
        
        # import pdb;pdb.set_trace()
        gpu_dict = sorted(gpu_dict, key= lambda x: x[1], reverse=True)
        return gpu_dict

    def adequate_gpu(self, min_momery, blocking=False):
        adequate_gpu_id = None

        while 1:
            gpu_status = self.get_gpus()
            if gpu_status[0][1] > min_momery:
                adequate_gpu_id=gpu_status[0][0]
                break
            elif not blocking:
                break
            time.sleep(1)

        return adequate_gpu_id

class TrainingNetTaskManager(object):
    def __init__(self, args) -> None:
        super().__init__()

        self.args = args
        self.proc_list=[]
        nppg = args.num_process_per_gpu
        self.gpu_tool = GPUTool(args)

        num_gpu = len(self.gpu_tool.get_gpus())
        self.max_proc = num_gpu * nppg


    def add_task(self, sub_work_dir, sciprt_path='lib/prune_net.py', debug=False):
        while 1:
            cudid=self.task_slot_is_available(5000)
            if cudid is not None:
                break
            self.clean_proc_list()
            time.sleep(1)

        python_exec = sys.executable
        args = self.args
#--------------------------------------------------------------------------------
        # import pdb;pdb.set_trace()
        # _cmd = 'CUDA_VISIBLE_DEVICES={} {} {} --dataset {} --save {} --seed {} \
        #     --batch_size {} --test-batch-size {} --lr {} --momentum {} --weight-decay {} \
        #     --epochs {} --log-interval {} --arch {} --depth {} --train_val_ratio {}'.format(cudid,python_exec, sciprt_path,
        #     args.dataset, sub_work_dir, args.seed, args.batch_size,
        #     args.test_batch_size, args.lr, args.momentum, args.weight_decay, args.epochs,
        #     args.log_interval,args.arch, args.depth, args.train_val_ratio)
#--------------------------------------------------------------------------------
        sys_argv_dict = get_sys_argv()
        sys_argv_dict['--save'] = sub_work_dir
        extra_params_str = args_dict_pair_to_str(sys_argv_dict)

        _cmd = 'CUDA_VISIBLE_DEVICES={} {} {} {}'.format(cudid, python_exec, sciprt_path, extra_params_str)
         
        if debug:
            if hasattr(self.args, 'logging'):
                self.args.logging.info(f'\tTask CMD:\n-->\t{_cmd}')
            else:
                print(_cmd)
        if args.fast_train_debug:
            _cmd += ' --fast_train_debug'
        # import pdb;pdb.set_trace()
        p = subprocess.Popen(_cmd, # process example
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL, shell=True)
        
        self.proc_list.append(p)

    def task_slot_is_available(self, min_mem=None):
        if min_mem is None:
            min_mem=-1
        gpu_id = self.gpu_tool.adequate_gpu(min_mem)
        if len(self.proc_list) >= self.max_proc:
            gpu_id = None
        return gpu_id
    
    def clean_proc_list(self):
        remove_list = []
        for proc in self.proc_list:
            if  not self.check_alive(proc):
                remove_list.append(proc)
        
        for v in remove_list:
            self.proc_list.remove(v)

    def check_alive(self, proc):
        if proc.poll() is None:
            return True 
        else:
            return False

    def wait(self):
        for proc in self.proc_list:
            proc.wait()
        self.kill_all()

    def kill_all(self):
        for proc in self.proc_list:
            proc.kill()
        self.proc_list = []


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    # ouput is [batch, 1000]
    # target is [batch]
    batch_size = target.size(0)
    num = output.size(1)
    target_topk = []
    appendices = []
    for k in topk:
        if k <= num:
            target_topk.append(k)
        else:
            appendices.append([0.0])
    topk = target_topk
    maxk = max(topk)
    # k=maxk dim=1
    _, pred = output.topk(maxk, 1, True, True)
    # pred size is [100, 5]
    # transpose
    pred = pred.t()
    # pred size is [5,100]
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    # correct is true/false

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))
    return res + appendices

def select_interval(x, seg_num):
    res= []
    for v in x:
            for i in range(seg_num):
                if i*(1.0/seg_num) < v and v < (i+1)*(1.0/seg_num):
                    res.append(i)
                    break
    return np.array(res) 