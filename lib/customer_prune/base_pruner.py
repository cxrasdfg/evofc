# coding=utf-8
import os
import shutil
from torch.nn.functional import margin_ranking_loss
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiplicativeLR, MultiStepLR, StepLR, CosineAnnealingLR
from torch.autograd import Variable

from utils import compute_flops
from utils.utils import GlobalNetRecorder, reset_random_seed, split_dataset, select_interval
from utils.utils import accuracy as accuracy_metric
from lib.lr_scheduler.WarmupLR import WarmupLR
from lib.org_models.models import *
from lib.customer_prune.pruner_strategy import get_pruning_strategy
from lib.datasets import dataloaders, get_num_cls

class BasePruningTrainer(object):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args 
        self.logging = args.logging

        self.create_dataset()
        self.create_pruning_strategy()

    def create_model(self, cfg=None):
        args = self.args
        model = create_model(args, cfg)

        if args.pretrain is not None and args.pretrain != '':
            init_archive =  torch.load(args.pretrain)
            init_weight = init_archive['state_dict']
            model.load_state_dict(init_weight)
            self.logging.info ('load pretrained model -> ' + args.pretrain)

        args.cuda = torch.cuda.is_available()
        # device = torch.device("cuda" if args.cuda else "cpu")
        # model = torch.nn.DataParallel(model)
        model.cuda()

        return model
    
    def create_model_with_cfg(self, cfg):
        model =self.create_model(cfg)
        return model 
    
    def create_dataset(self):
        reset_random_seed(self.args)

        self.logging.info('Create dataset for pruning...')
        self.pruned_train_loader, self.pruned_test_loader = dataloaders.create_prune_dataset(self.args)

        # for evolved dataset
        self.logging.info('Create dataset for evolution...')
        train_loader, val_loader, test_loader = dataloaders.create_evolved_dataset(self.args)
        full_train_loader = self.pruned_train_loader
        self.train_loader = train_loader
        self.full_org_train_loader = full_train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.logging.info('Dataset creation completion...')


        self.logging.info('num of train: {}, num of val: {}, num of test: {}'.format(len(self.train_loader.dataset),
                                                                                          len(self.val_loader.dataset),
                                                                                          len(self.test_loader.dataset) ))

        self.logging.info('Loading dataset Successful!')

    def create_pruning_strategy(self):
        self.pruning_strategy = get_pruning_strategy(self.args)

    def test(self, model, dataloader, name):
        args = self.args

        model.eval()
        test_loss = 0
        num_dataset = len(dataloader.dataset)
        output_list = [None] * len(dataloader)
        target_list = [None] * len(dataloader)

        with torch.no_grad():
            for i, (data, target) in enumerate(dataloader):
                if i == len(dataloader) - 1:
                    if num_dataset % dataloader.batch_size != 0:
                        remain_num = num_dataset % dataloader.batch_size
                        data = data[:remain_num]
                        target = target[:remain_num]
                if target.dim() == 2:
                    assert target.shape[1] == 1
                    target = target[:,0]

                target_list[i] = target
                data, target = data.cuda(), target.cuda()
                # data, target = Variable(data, volatile=True), Variable(target)
                # import pdb;pdb.set_trace()
                output = model(data)
                output_list[i] = output.cpu()
                # import pdb;pdb.set_trace()
                test_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss

                if (i+1) % args.log_interval == 0:
                    self.logging.info('Test Iterations: [{}/{}]\t'.format(
                        i+1, len(dataloader)))
                
        output_list  = torch.cat(output_list, dim=0)
        target_list = torch.tensor(np.concatenate(target_list))
        top1, top5 = accuracy_metric(output_list, target_list, [1,5])
        test_loss /= num_dataset

        eval_num = len(output_list)

        self.logging.info('\n{} set: Average loss: {:.4f}, top1: {}/{} ({:.1f}%), top5: {}/{} ({:.1f}%)\n'.format(
            name, test_loss, top1*eval_num, eval_num, top1*100, top5*eval_num, eval_num, top5*100))
        return top1, top5
    
    def simple_test_after_prune(self, model):
        return self.test(model, self.pruned_test_loader, 'Testing (pruning)')
    
    def prune(self, args):
        reset_random_seed(args)
        model = self.create_model()

        if args.pretrain is not None:
            flag, start_epoch, best_prec1 = self.resume(model, optimizer=None,file_path=args.pretrain)
            if flag:
                args.start_epoch = start_epoch
                start_epoch = start_epoch

        self.logging.info('Create original model successfully!')
        # simple test model after Pre-processing prune (simple set BN scales to zeros)
        acc, top5 = self.test(model, self.pruned_test_loader, 'Test (Before Pruning)')

        cfg, cfg_mask = self.pruning_strategy.prune_model(model, args, self)

        newmodel = self.create_model_with_cfg(cfg)
        if args.cuda:
            newmodel.cuda()

        # import pdb;pdb.set_trace()
        self.pruning_strategy.super2sub(model, newmodel, cfg, cfg_mask)

        with open(os.path.join(args.save, 'pruned.config'),'w') as f:
            f.write(json.dumps({'cfg': cfg}))

        num_parameters = sum([param.nelement() for param in newmodel.parameters()])
        self.logging.info(newmodel.__str__())
        old_model = model
        model = newmodel
        acc, top5 = self.test(model, self.pruned_test_loader, 'Test (Pruned Model)')


        self.logging.info("number of parameters: "+str(num_parameters))
        with open(os.path.join(args.save, "prune.txt"), "w") as fp:
            fp.write("Number of parameters: \n"+str(num_parameters)+"\n")
            fp.write("Test accuracy: \n"+str(acc)+"\n")

        self.logging.info('Before Prune->:\n')
        model_flops = compute_flops.print_model_param_flops(old_model, input_res=next(iter(self.train_loader))[0].shape[-1])
        model_params_num = compute_flops.print_model_param_nums(old_model)
        self.logging.info('  + Number of FLOPs: %.5fG' % (model_flops))
        self.logging.info('  + Number of params: %.2fM' % (model_params_num))
        
        self.logging.info('After Prune->:\n')
        model_flops = compute_flops.print_model_param_flops(newmodel, input_res=next(iter(self.train_loader))[0].shape[-1])
        self.logging.info('  + Number of FLOPs: %.5fG' % (model_flops))
        model_params_num = compute_flops.print_model_param_nums(newmodel)
        self.logging.info('  + Number of params: %.2fM' % (model_params_num))

        return cfg, model_flops, model_params_num

    def resume(self, model, optimizer, file_path):
        if os.path.isfile(file_path):
            self.logging.info("=> loading checkpoint '{}'".format(file_path))
            checkpoint = torch.load(file_path)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            self.logging.info("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
                .format(file_path, checkpoint['epoch'], best_prec1))
            return True, start_epoch, best_prec1
        else:
            self.logging.info("=> no checkpoint found at '{}'".format(file_path))
        return False, None, None
    
    def load_pretrain(self, model):
        if self.args.pretrain is not None and self.args.pretrain != '':
           init_archive =  torch.load(self.args.pretrain)
           init_weight = init_archive['state_dict']
           model.load_state_dict(init_weight)
           self.logging.info ('load pretrained model -> ' + self.args.pretrain)

    def save_training_step(self, model, optimizer, epoch, is_best, best_prec1_test):
        args = self.args
        self.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1_test,
                'optimizer': optimizer.state_dict(),
                'cfg': model.cfg
            }, is_best, filepath=args.save)
        
    def save_checkpoint(self, state, is_best, filepath):
        torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))
        if is_best:
            shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath, 'model_best.pth.tar'))
    
    def train(self, model, optimizer, train_loader, epoch):
        args = self.args
        model.train()
        avg_loss = 0.
        train_acc = 0.
        for batch_idx, (data, target) in enumerate(train_loader):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            loss, pred = self.train_batch(model, optimizer, batch_idx, data, target, train_loader, epoch)
            
            avg_loss += loss
            train_acc += pred.eq(target.data.view_as(pred)).cpu().sum()
            
            if args.fast_train_debug:
                break
        return avg_loss, train_acc
    
    def train_batch(self, model, optimizer, batch_idx, data, target, train_loader, epoch):
        args = self.args
        data, target = Variable(data), Variable(target)
        
        optimizer.zero_grad()
        output = model(data)
        
        loss = F.cross_entropy(output, target)
        pred = output.data.max(1, keepdim=True)[1]
        
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            self.logging.info('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
            
        return loss.item(), pred
    
    def create_lr_scheduler(self, optimizer):
        args = self.args
        if args.lr_scheduler == 'multi_step':
            scheduler = MultiStepLR(optimizer, milestones=[args.epochs*0.5, args.epochs*0.75], gamma=args.lr_gamma)
            assert 'imagenet' not in args.dataset

        elif args.lr_scheduler == 'step':
            scheduler = StepLR(optimizer, step_size= args.lr_step_size, gamma=args.lr_gamma)
        elif args.lr_scheduler == 'cos':
            scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

        # cancle warmup this time
        # if args.warmup is not None and args.warmup > 0:
            # scheduler = WarmupLR(scheduler, init_lr=args.warmup_init_lr, num_warmup=args.warmup, warmup_strategy='linear')
        return scheduler