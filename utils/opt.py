
import argparse
import os, sys, time, argparse, subprocess

def get_opt():
    parser = argparse.ArgumentParser("EvoCrit")
    # directory
    parser.add_argument('--root', type=str, default='exp_dir',
                        help='experiment directory')
    # parser.add_argument('--save', type=str, default=time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time())), help='experiment name')
    parser.add_argument('--save', type=str, default=None, help='experiment name')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'mnist', 'celeba_64', 'celeba_256',
                                 'imagenet_32', 'ffhq', 'lsun_bedroom_128', 'imagenet', 'places365'], help='which dataset to use')

    # DDP.
    parser.add_argument('--num_process_per_gpu', type=int, default=5,
                        help='number of gpus')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed used for initialization')

    # Train Arch
    # optimizer of high-level training
    parser.add_argument('--pop_size', type=int, default=10, help='population size of individuals')
    parser.add_argument('--n_gens', type=int, default=20, help='number of generations')
    parser.add_argument('--n_offsprings', type=int, default=10, help='number of offspring created per generation')
    # parser.add_argument('--crossover', type=float, default=.9, help='probability of crossover')
    # parser.add_argument('--mutation', type=float, default=.1, help='probability of mutation')
    parser.add_argument('--resolution_scale', type=float, default=1.0, help='resolution scale of the input image when searching the architectures')
    parser.add_argument('--train_remained', type=float, default=1.0, help='to reduce the number of the training data')
    parser.add_argument('--gpu_wait_time', type=float, default=10.0, help='wait for gpu changes')

    # Train Net
    # some hyper parameters of low-level training which is not considered in the high-level optimization
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size per GPU')
    parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
    parser.add_argument('--cri_batch_size', type=int, default=128,
                    help='input batch size for computing the criterion scores')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--lr-step-size', default=30, type=int, help='lr step size (default: 30)')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='lr gamma for scheduler (default: 0.1)')
    parser.add_argument('--epochs', type=int, default=300,
                        help='num of training epochs')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
    parser.add_argument('--arch', default='resnet', type=str, 
                    help='architecture to use')
    parser.add_argument('--depth', default=56, type=int,
                    help='depth of the neural network')               
    parser.add_argument('--load_state', default=False, type=bool,
                    help='state loading flags')
    parser.add_argument('--train_val_ratio', default=19, type=int,
                    help='train vs val')
    parser.add_argument('--fast_train_debug', action='store_true',
                    help='fast debug flag')
    parser.add_argument('--resume', type=str,
                    help='resume path')
    parser.add_argument('--ws', action='store_true', help='weight sharing flags')
    parser.add_argument('--num_workers', type=int, default=1, help='num of the workers')
    parser.add_argument('--min_ratio', type=float, default=.0, help='minimal pruning rate for each layer')
    parser.add_argument('--cosine_anneal_bound', type=float, default=None, help='consine_annealing_bound')
    parser.add_argument('--anneal_bound', type=float, default=None, help='consine_annealing_bound')
    parser.add_argument('--indi_bn', action='store_true', help='indepedent batch norm')
    parser.add_argument('--pretrain', type=str, default=None, help='pretrain model path')
    parser.add_argument('--ws_layer_cnt', type=int, default=None, help='layer counter in layer sharing')
    parser.add_argument('--criterion', type=str, default='default',choices=['default','weight_magnitude', 'front'], 
    help='criterion')
    parser.add_argument('--warmup', type=int, default=20,help='default warmup epochs')
    parser.add_argument('--warmup_init_lr', type=float, default=0.01,help='default warmup init lr')
    parser.add_argument('--distill', type=float, default=None, help='coefficient for distillation, disabled if none')
    parser.add_argument('--alpha_ff', type=float, default=0.1, help='coefficient for full networks training')
    parser.add_argument('--distill_loss', type=str, default='l1', help='distillation loss type')
    parser.add_argument('--detach', default=False, action='store_true', help='detach the distillation loss for teacher network')
    parser.add_argument('--distill_earlystop', default=None, type=int, help='stop distill after the required epochs')
    parser.add_argument('--lr_scheduler', default='multi_step', type=str, choices=['multi_step', 'step', 'cos'], help='learning rate scheduler')
    
    parser.add_argument('--super_net_weight_dir', type=str, default='', help='supernet dir')
    parser.add_argument('--search_work_dir', type=str, default='', help='pruning configuration dir')
    # parser.add_argument('--extract_pareto_points', default=False, action='store_true', help = 'flag of extracting the points of the pareto front')
    parser.add_argument('--pareto_dir', type=str, default='', help='pruning configuration dir')
    parser.add_argument('--dali', default=True, type=bool, help='use the nvidia dali framework')

    parser.add_argument('--cuda_visible_devices', default=None, type=str, help='CUDA_VISIBLE_DEVICES, e.g.: `0,2,3`')
    # ------------ For Visualization -------------
    parser.add_argument('--dynamic', action='store_true', default=False, help='dynamic graph')
    parser.add_argument('--save_fig', action='store_true', default=False, help='save flag')
    parser.add_argument('--only_pareto', action='store_true', default=False)
    
    return parser