# coding=utf-8

from typing import Any, Callable
from torch._tensor import Tensor
from torch.nn.modules import Module
from .base_strategy import *

class VGGStrategy(BaseStrategy):
    
    def __init__(self, args) -> None:
        super().__init__(args)
        self.n_depth = self.args.depth-2

    def prune_model(self, model, args, pruner, **kwargs):
        layer_id = 0
        # cfg = [32, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 256, 256, 256, 'M', 256, 256, 256]
        cfg = []
        # assert self.args.depth == 16
        cfg_mask = []
        genome = self.get_genome_from_file(args)
        prune_probs = genome[:,0]
        prune_criterions = genome[:,1]
        prune_criterions = select_interval(prune_criterions, 4) # discrete

        snip_scores = get_all_criterion(model, pruner.pruned_train_loader, 'cuda', prune_probs)
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                out_channels = m.weight.data.shape[0]
                select_cri, num_keep, mask = self.get_mask(snip_scores, prune_probs, prune_criterions, layer_id, out_channels)
                
                cfg.append(num_keep)
                cfg_mask.append(mask)
                if 'cfg_mask_with_extra_infos' in kwargs:
                    cfg_mask_with_extra_infos = kwargs['cfg_mask_with_extra_infos']
                    cfg_mask_with_extra_infos.append([select_cri, num_keep, out_channels])

                layer_id += 1

            elif isinstance(m, nn.MaxPool2d):
                cfg.append('M')
        return cfg, cfg_mask
    
    def exchange_weight_super_and_sub(self, supernet: Module, newmodel: Module, cfg: list, cfg_mask: list, 
                                      exchange_func: Callable[[Tensor, Tensor, Tensor, int], Any], **kwargs):
        layer_id_in_cfg = 0
        end_mask = cfg_mask[layer_id_in_cfg]
        start_mask = torch.ones(3).bool()
    
        supernet = self.expose_dp_mdoel(supernet)
        newmodel = self.expose_dp_mdoel(newmodel)

        for [m0, m1] in zip(supernet.modules(), newmodel.modules()):
            if isinstance(m0, nn.BatchNorm2d):
                exchange_func(m0.weight.data, m1.weight.data, end_mask)
                exchange_func(m0.bias.data, m1.bias.data, end_mask)
                exchange_func(m0.running_mean, m1.running_mean, end_mask)
                exchange_func(m0.running_var, m1.running_var, end_mask)

                layer_id_in_cfg += 1
                start_mask = end_mask
                if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                    end_mask = cfg_mask[layer_id_in_cfg]

            elif isinstance(m0, nn.Conv2d):
                cmask = (end_mask[:, None] * start_mask[None])
                exchange_func(m0.weight.data, m1.weight.data, cmask)
                
                if 'cfg_mask_with_extra_infos' in kwargs:
                    cfg_mask_with_extra_infos = kwargs['cfg_mask_with_extra_infos']
                    output_channel_with_id = kwargs['output_channel_with_id']   
                    output_channel_with_id.append([layer_id_in_cfg, cfg_mask_with_extra_infos[layer_id_in_cfg]])
                    
            elif isinstance(m0, nn.Linear):
                if layer_id_in_cfg == len(cfg_mask):
                    mask = cfg_mask[-1]
                    exchange_func(m0.weight.data, m1.weight.data, mask, dim=1)
                    exchange_func(m0.bias.data, m1.bias.data, mask=None)

                    layer_id_in_cfg += 1
                    continue

                exchange_func(m0.weight.data, m1.weight.data, mask = None)
                exchange_func(m0.bias.data, m1.bias.data, mask=None)

            elif isinstance(m0, nn.BatchNorm1d):
                exchange_func(m0.weight.data, m1.weight.data, mask=None)
                exchange_func(m0.bias.data, m1.bias.data, mask=None)
                exchange_func(m0.running_mean, m1.running_mean, mask=None)
                exchange_func(m0.running_var, m1.running_var, mask=None)