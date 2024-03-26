# coding=utf-8
#coding=utf-8
import os
import abc
from ast import literal_eval
from functools import partial
from typing import Callable, Any

import numpy as np
import torch
from torch import nn

from lib.prune_criterion import get_all_criterion
from utils.utils import select_interval


class BaseStrategy(object):
    def __init__(self, args) -> None:
        self.args = args
        self.n_depth = None # the number of convolutional layer
    
    def set_subnet_config(self, model, args, cfg_mask, return_masked_weight = False):
        raise NotImplementedError() 

    def super2sub(self, supernet, newmodel, cfg, cfg_mask):
        # raise NotImplementedError() 
        return self.exchange_weight_super_and_sub(supernet, newmodel, cfg, cfg_mask, self.copy_weight_super_to_sub_func())

    def sub2super(self, supernet, newmodel, cfg, cfg_mask):
        # raise NotImplementedError() 
        return self.exchange_weight_super_and_sub(supernet, newmodel, cfg, cfg_mask, self.copy_weight_sub_to_super_func())

    def prune_model(self, model, args, pruner, **kwargs):
        raise NotImplementedError() 

    def get_mask(self, snip_scores, prune_probs, prune_criterions, idx, out_channels):
        """Get mask based on each convolutional layer
        """
        prune_prob_stage = prune_probs[idx]
        # weight_copy = m.weight.data.abs().clone().cpu().numpy()
        # L1_norm = np.sum(weight_copy, axis=(1,2,3))
        select_cri = prune_criterions[idx]
        snip_norm = snip_scores[select_cri][idx].cpu().numpy()
        
        num_keep = int(np.ceil(out_channels * (1 - prune_prob_stage)))
        if num_keep < 1:
            num_keep = 1
        arg_max = np.argsort(snip_norm)
        arg_max_rev = arg_max[::-1][:num_keep]
        mask = torch.zeros(out_channels)
        mask[arg_max_rev.tolist()] = 1

        return select_cri, num_keep, mask.bool()
    
    def get_genome_from_file(self, args):
        with open(os.path.join(args.save,'genome.txt'),'r') as f:
            raw_cnt = f.read()
            genome_org = np.array(literal_eval(raw_cnt))
            genome = genome_org.reshape(-1, 2)

        return genome

    def expose_dp_mdoel(self, model):
        if hasattr(model, 'module'):
            return model.module
        return model
    
    def exchange_weight_super_and_sub(self, supernet: torch.nn.Module, newmodel :torch.nn.Module, cfg: list, cfg_mask: list,
                                      exchange_func: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, int], Any], **kwargs):
        raise NotImplementedError()
    
    def copy_weights_auxiliary_func(self, super_parms: torch.Tensor, sub_params: torch.Tensor, 
                                    mask:torch.Tensor, dim:int = None, direction: int = 0):
        if dim is None:
            dim = 0
        
        if mask is not None:
            mask_with_slice = (slice(None),)*dim +(mask,)
            output_shape = super_parms[mask_with_slice].shape
            if direction == 0: # from supernet to sub
                sub_params.view(*output_shape).copy_(super_parms[mask_with_slice])
            elif direction == 1:
                super_parms[mask_with_slice] = sub_params.view(*output_shape)
            else:
                assert 0
        else:
            if direction == 0: # from supernet to sub
                sub_params.copy_(super_parms)
            elif direction == 1:
                super_parms.copy_(sub_params)
            else:
                assert 0

    def copy_weight_super_to_sub_func(self):
        
        return partial(self.copy_weights_auxiliary_func, direction=0)
    
    def copy_weight_sub_to_super_func(self):
    
        return partial(self.copy_weights_auxiliary_func, direction=1)
    