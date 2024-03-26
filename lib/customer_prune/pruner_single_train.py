# coding=utf-8
import logging
import os, sys
from torch.nn.functional import margin_ranking_loss
sys.path.append(os.path.dirname(
                os.path.dirname(
                os.path.dirname(os.path.realpath(__file__)))))

import json
import torch
import torch.optim as optim

from utils import compute_flops
from utils.logger import Logger
from utils.utils import GlobalNetRecorder, reset_random_seed, split_dataset, select_interval
from utils.opt import get_opt
from lib.lr_scheduler.WarmupLR import WarmupLR
from lib.org_models.models import *
from lib.customer_prune.pruner_strategy import get_pruning_strategy
from lib.customer_prune.base_pruner import BasePruningTrainer

class SingleTrainingPruner(BasePruningTrainer):
    def __init__(self, args) -> None:
        super().__init__(args)

    def train_from_scratch(self, cfg, pd = False):
        args = self.args
        model = self.create_model_with_cfg(cfg=cfg)
        if args.cuda:
            model.cuda()
            if pd:
                 model = torch.nn.DataParallel(model)

        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        lr_scheduler = self.create_lr_scheduler(optimizer)
        args.start_epoch=0

        best_prec1_val = 0.
        best_prec1_test = 0.
        for epoch in range(args.start_epoch, args.epochs):
            self.train(model, optimizer, self.train_loader, epoch)
            
            lr_scheduler.step()
            self.logging.info('lr schedule!')

            torch.cuda.empty_cache()
            prec1_val, prec5_val = self.test(model, self.val_loader , 'Evo(validation)')
            torch.cuda.empty_cache()
            prec1_test, prec5_test = self.test(model, self.test_loader, 'Evo(test)')
            torch.cuda.empty_cache()

            is_best = prec1_test > best_prec1_test
            best_prec1_test = max(prec1_test, best_prec1_test)
            best_prec1_val = max(prec1_val, best_prec1_val)
            self.save_training_step(model.module if hasattr(model, 'module') else model, optimizer, epoch, is_best, best_prec1_test)
            if args.fast_train_debug:
                    break
        self.logging.info('Best on Validation Set: {:.4f}'.format(best_prec1_val))
        self.logging.info('Best on Test Set: {:.4f}'.format(best_prec1_test))
        val_err = 1 - best_prec1_val
        return val_err.item()
    
if __name__ == "__main__":
    # Prune settings
    parser = get_opt()
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    assert args.pretrain is None  # discard this var

    logging = Logger(0, 'prune_net', args.save)
    logging.info(args.__str__())
    args.logging = logging 
    # import pdb;pdb.set_trace()
    STP = SingleTrainingPruner(args)
    genome = STP.pruning_strategy.get_genome_from_file(args)
    genome_org = genome.reshape(-1)

    gnr = GlobalNetRecorder(os.path.dirname(os.path.dirname(args.save)))
    
    cfg, flops, params = STP.prune(args)
    logging.info('*'*20+'Complete Pruning'+'*'*20)
    torch.cuda.empty_cache()
    # import pdb; pdb.set_trace()

    val_err = STP.train_from_scratch(cfg)
    logging.info('*'*20+'Complete Training from Scratch'+'*'*20)

    performance = {'err':val_err,'flops':flops, 'params':params}
    gnr.write(genome_org, performance)
    logging.info('Write to network record file:'+  str(performance))
    logging.shutdown()
