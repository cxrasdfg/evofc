# coding=utf-8
import os, sys
import shutil
from torch.nn.functional import margin_ranking_loss

import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms

from lib.org_models.models import *
from utils import compute_flops
from utils.utils import parse_cuda_visible_devices
from lib.prune_criterion import get_all_criterion, get_all_criterion_wm
from utils.logger import Logger
from utils.utils import GlobalNetRecorder, reset_random_seed, split_dataset
from lib.customer_prune.base_pruner import BasePruningTrainer

class WSPruningTrainer(BasePruningTrainer):
    def __init__(self, args):
        super().__init__(args)        
        
        self._init_super_net()

    def _init_super_net(self):
        reset_random_seed(self.args)
        model = self.create_model()
        self.load_pretrain(model)

        self.args.cuda = torch.cuda.is_available()
        device = torch.device("cuda" if self.args.cuda else "cpu")
        model.cuda(device)

        self.super_net = model

        self.args.start_epoch=0
        self._share_train_dataloader_iter = iter(self.train_loader) 
        self.global_epoch_idx = 0
        self.global_iterations = 0

        if self.args.resume is not None:
            self.resume(self.args)

    def prune(self, args):
        # acc = test(model)
        cfg, cfg_mask = self.pruning_strategy.prune_model(self.super_net, args, self)
        
        newmodel = self.create_model_with_cfg(cfg)
        self.pruning_strategy.super2sub(self.super_net, newmodel, cfg, cfg_mask)

        with open(os.path.join(args.save, 'pruned.config'),'w') as f:
            f.write(json.dumps({'cfg': cfg}))

        newmodel = nn.DataParallel(newmodel, device_ids=parse_cuda_visible_devices(args.cuda_visible_devices))
        newmodel.cuda()

        num_parameters = sum([param.nelement() for param in newmodel.parameters()])
        self.logging.info(newmodel.__str__())
        old_model = self.super_net
        model = newmodel
        acc, top5 = self.simple_test_after_prune(model)


        self.logging.info("number of parameters: "+str(num_parameters))
        with open(os.path.join(args.save, "prune.txt"), "w") as fp:
            fp.write("Number of parameters: \n"+str(num_parameters)+"\n")
            fp.write("Test accuracy: \n"+str(acc)+"\n")


        self.logging.info('Before Prune->:\n')
        model_flops = compute_flops.print_model_param_flops(old_model, input_res=next(iter(self.pruned_train_loader))[0].shape[-1])
        model_params_num = compute_flops.print_model_param_nums(old_model)
        self.logging.info('  + Number of FLOPs: %.5fG' % (model_flops))
        self.logging.info('  + Number of params: %.2fM' % (model_params_num))
        
        self.logging.info('After Prune->:\n')
        model_flops = compute_flops.print_model_param_flops(newmodel, input_res=next(iter(self.pruned_train_loader))[0].shape[-1])
        self.logging.info('  + Number of FLOPs: %.5fG' % (model_flops))
        model_params_num = compute_flops.print_model_param_nums(newmodel)
        self.logging.info('  + Number of params: %.2fM' % (model_params_num))

        return cfg, cfg_mask,model_flops, model_params_num, newmodel
            
    def train_share_subnet(self, args):
        self.logging = Logger(0, 'prune_net'+args.save, args.save)
        self.logging.info(args.__str__())
        genome_org = args.genome
        cfg, cfg_mask, model_flops, model_params_num, newmodel = self.prune(args)
        torch.cuda.empty_cache()
        optimizer = optim.SGD(newmodel.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        lr_scheduler = self.create_lr_scheduler(optimizer)

        best_prec1_val = 0.
        best_prec1_test = 0.
        epoch = self.global_epoch_idx
        
        # def fool(epoch, epoch_list):
        #     temp_list = [0] + epoch_list+[1e100] # add a place holder
        #     idx = 0
        #     for i, epoch_idi in enumerate(temp_list):
        #         if epoch_idi <= epoch and epoch <= temp_list[i+1]:
        #             idx = i
        #             break
        #     return idx 

        # for param_group in optimizer.param_groups:
        #     param_group['lr'] *= 0.1**fool(epoch, [args.epochs*0.5, args.epochs*0.75])
        
        self.share_train_once(newmodel, optimizer, lr_scheduler)
        
        torch.cuda.empty_cache()
        prec1_val, top5_val = self.test(newmodel,self.val_loader, 'validation')
        torch.cuda.empty_cache()
        prec1_test, top5_test = self.test(newmodel, self.test_loader, 'test')
        torch.cuda.empty_cache()
        
        self.pruning_strategy.sub2super(self.super_net, newmodel, cfg, cfg_mask)
        
        is_best = prec1_test > best_prec1_test
        best_prec1_test = max(prec1_test, best_prec1_test)
        best_prec1_val = max(prec1_val, best_prec1_val)
        # save checkpoint
        
        self.save_checkpoint({
            'epoch': epoch + 1,
            'global_epoch_index': self.global_epoch_idx+1,
            'state_dict': newmodel.module.state_dict() if hasattr(newmodel,'module') else newmodel.state_dict(),
            'state_dict_super_net': self.super_net.state_dict(),
            'best_prec1': best_prec1_test,
            'optimizer': optimizer.state_dict(),
            'cfg': newmodel.module.cfg if hasattr(newmodel,'module') else  newmodel.cfg 
        }, False, filepath=args.save)
       
        self.logging.info('Best on Validation Set: {:.4f}'.format(best_prec1_val))
        self.logging.info('Best on Test Set: {:.4f}'.format(best_prec1_test))
        val_err = 1 - best_prec1_val

        gnr = GlobalNetRecorder(os.path.dirname(os.path.dirname(args.save)))
    
        # cfg, flops, params = prune(args)
        self.logging.info('*'*20+'Complete Pruning'+'*'*20)
        # torch.cuda.empty_cache()
        # import pdb; pdb.set_trace()

        # val_err = train_from_scratch(args,cfg)
        # self.logging.info('*'*20+'Complete Training from Scratch'+'*'*20)

        performance = {'err':val_err.item(),'flops':model_flops, 'params':model_params_num}
        gnr.write(genome_org, performance)
        self.logging.info('Write to network record file:'+  str(performance))
        self.logging.shutdown()
        # return val_err.item()
    
    def share_train_once(self, model, optimizer, lr_scheduler):
        model.train()
        avg_loss = 0.
        train_acc = 0.
        iter_per_net = self.train_loader.__len__()*self.args.epochs// (self.args.n_offsprings* self.args.n_gens)
       
        for _ in range(self.global_epoch_idx):
            lr_scheduler.step()
        self.logging.info(f'LR:{lr_scheduler.get_last_lr()[-1]:.6f}')

        for i in range(iter_per_net):
            try:
                data, target = next(self._share_train_dataloader_iter)
                self.global_iterations += 1
            except StopIteration:
                self._share_train_dataloader_iter = iter(self.train_loader)
                data, target = next(self._share_train_dataloader_iter)
                self.global_epoch_idx += 1
                self.global_iterations = 1

                lr_scheduler.step()
                self.logging.info(f'LR:{lr_scheduler.get_last_lr()[-1]:.6f}')

            data, target = data.to('cuda'), target.to('cuda')
            loss, pred = self.train_batch(model, optimizer, self.global_iterations, data, target, self.train_loader, self.global_epoch_idx +1 )

            avg_loss += loss
            train_acc += pred.eq(target.data.view_as(pred)).cpu().sum()

            if self.args.fast_train_debug:
                break

    def fintune_train_once_epoch(self, model, optimizer, epoch):
        self.train(model, optimizer, self.full_org_train_loader, epoch)
            
    def resume(self, args):
        assert args.resume
        if os.path.isfile(args.resume):
            self.logging.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            self.global_epoch_idx = checkpoint['global_epoch_index']
            best_prec1 = checkpoint['best_prec1']
            self.super_net.load_state_dict(checkpoint['state_dict_super_net'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            self.logging.info("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
                .format(args.resume, checkpoint['epoch'], best_prec1))
        else:
            self.logging.info("=> no checkpoint found at '{}'".format(args.resume))
            assert 0
