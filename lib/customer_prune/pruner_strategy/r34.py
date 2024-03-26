from typing import Any, Callable
from torch._tensor import Tensor
from torch.nn.modules import Module
from .base_strategy import *

class ResNet34Strategy(BaseStrategy):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.n_depth = args.depth + 2
        self.downsample_skip_idx = [10,19,32] # resnet 34 downsample layer index

    def set_subnet_config(self, model, args, cfg_mask, return_masked_weight=False):
        return super().set_subnet_config(model, args, cfg_mask, return_masked_weight)
    
    def exchange_weight_super_and_sub(self, supernet: Module, newmodel: Module, cfg: list, cfg_mask: list,
                                      exchange_func: Callable[[Tensor, Tensor, Tensor, int], Any], **kwargs):
        # import pdb; pdb.set_trace()
        start_mask = torch.ones(3)
        layer_id_in_cfg = 0
        conv_count = 1
        ds_flag = 0
        
        supernet = self.expose_dp_mdoel(supernet)
        newmodel = self.expose_dp_mdoel(newmodel)

        for [m0, m1] in zip(supernet.modules(), newmodel.modules()):
            if isinstance(m0, nn.Conv2d):
                if conv_count in self.downsample_skip_idx:
                    exchange_func(m0.weight.data, m1.weight.data, mask=None)

                    conv_count += 1 
                    ds_flag = 1- ds_flag
                    continue

                if conv_count == 1:
                    exchange_func(m0.weight.data, m1.weight.data, mask=None)
                    conv_count += 1

                    continue
                if conv_count % 2 == ds_flag:
                    mask = cfg_mask[layer_id_in_cfg]
                    exchange_func(m0.weight.data, m1.weight.data, mask)
                    
                    if 'cfg_mask_with_extra_infos' in kwargs:
                        cfg_mask_with_extra_infos = kwargs['cfg_mask_with_extra_infos']
                        output_channel_with_id = kwargs['output_channel_with_id']   
                        output_channel_with_id.append([conv_count, cfg_mask_with_extra_infos[layer_id_in_cfg]])

                    layer_id_in_cfg += 1
                    conv_count += 1
                    continue
                if conv_count % 2 == 1-ds_flag:
                    mask = cfg_mask[layer_id_in_cfg-1]
                    exchange_func(m0.weight.data, m1.weight.data, mask, dim=1)

                    conv_count += 1
                    continue

            elif isinstance(m0, nn.BatchNorm2d):
                if self.args.indi_bn:
                    continue
                if conv_count % 2 == 1-ds_flag and conv_count-1 not in self.downsample_skip_idx:
                    mask = cfg_mask[layer_id_in_cfg-1]

                    exchange_func(m0.weight.data, m1.weight.data, mask)
                    exchange_func(m0.bias.data, m1.bias.data, mask)
                    exchange_func(m0.running_mean, m1.running_mean, mask)
                    exchange_func(m0.running_var, m1.running_var, mask)
                    continue

                exchange_func(m0.weight.data, m1.weight.data, mask=None)
                exchange_func(m0.bias.data, m1.bias.data, mask=None)
                exchange_func(m0.running_mean, m1.running_mean, mask=None)
                exchange_func(m0.running_var, m1.running_var, mask=None)
            elif isinstance(m0, nn.Linear):
                exchange_func(m0.weight.data, m1.weight.data, mask=None)
                exchange_func(m0.bias.data, m1.bias.data, mask=None)

    def prune_model(self, model, args, pruner, **kwargs):
        layer_id = 1
        cfg = []
        cfg_mask = []
        genome = self.get_genome_from_file(args)
        prune_probs = genome[:,0]
        prune_criterions = genome[:,1]
        prune_criterions = select_interval(prune_criterions, 4) # discrete

        # self.pruned_train_loader, self.pruned_test_loader = self.create_prune_dataset() 
        snip_scores = get_all_criterion(model, pruner.pruned_train_loader, 'cuda', prune_probs)
        
        ds_flag = 0
        model = self.expose_dp_mdoel(model)

        # import pdb;pdb.set_trace()
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                out_channels = m.weight.data.shape[0]
                if layer_id in self.downsample_skip_idx:
                    layer_id += 1
                    ds_flag = 1- ds_flag
                    continue
                if layer_id % 2 == ds_flag:
                    select_cri, num_keep, mask = self.get_mask(snip_scores, prune_probs, prune_criterions, layer_id-1, out_channels)
                    
                    cfg_mask.append(mask)
                    cfg.append(num_keep)
                    if 'cfg_mask_with_extra_infos' in kwargs:
                        cfg_mask_with_extra_infos = kwargs['cfg_mask_with_extra_infos']
                        cfg_mask_with_extra_infos.append([select_cri, num_keep, out_channels])

                    layer_id += 1
                    continue
                layer_id += 1

        return cfg, cfg_mask