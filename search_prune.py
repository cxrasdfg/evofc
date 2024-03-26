# coding=utf-8
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import os, sys, time, argparse, subprocess
from copy import copy
import numpy as np
import torch as th

from pymop.problem import Problem
from pymoo.optimize import minimize

# relative files
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from utils.logger import Logger, Writer
from utils import utils
from utils.opt import get_opt
from utils.utils import GlobalNetRecorder as GNR, reset_random_seed, TrainingNetTaskManager

from lib.ga import evoprune
from lib.customer_prune.pruner_ws import WSPruningTrainer
from lib.customer_prune.pruner_strategy import get_pruning_strategy

class EvoPruneProblem(Problem):
    def __init__(self, n_vars, n_obj, n_constr, lb, ub,
                 config,):
        super(EvoPruneProblem, self).__init__(n_var=n_vars, n_obj=n_obj, n_constr=n_constr, type_var=np.int)
        self.xl = lb
        self.xu = ub
        self.config = config
        self._save_dir = self.config.save
        self._n_evaluated = 0
        self._cuda_device = 0

    def _evaluate(self, POP, F, *vargs, **kwargs):
        n_pop = len(POP)
        objs = np.full((n_pop, self.n_obj), np.nan)
        
        tnt = TrainingNetTaskManager(args)
        for idx, genome in enumerate(POP):
            arch_id = self._n_evaluated + 1
            logging.info('Evaluate, Network id = {}'.format(arch_id))
            

            # decode the network and evaluate...
            # dummy test
            # performance = {'err':np.random.rand(),'flops':321.2 + np.random.normal()}
            # performance = gnr.read(genome)
            # if performance is not None:
            #     logging.info('load from cache, Network id = {}'.format(arch_id))
            #     {'err':np.random.rand(),'flops':np.random.rand()} # for dummy test can be reproducible
            # else:
            #     performance = {'err':np.random.rand(),'flops':np.random.rand()}
            #     gnr.write(genome, performance)
            
            indi_dir = os.path.join(args.save,'pop', 'gen%03d-indi%05d'%(GEN_ITER, idx))
            if not os.path.exists(indi_dir):
                os.makedirs(indi_dir)
            genome_file_path = os.path.join(indi_dir, 'genome.txt')
            with open(genome_file_path,'w') as f:
                f.write(genome.tolist().__str__())
            
            if not args.ws:
                if gnr.read(genome) is None:
                    tnt.add_task(indi_dir, sciprt_path='lib/customer_prune/pruner_single_train.py', debug=True)
                    time.sleep(args.gpu_wait_time) # wait gpu change
                else:
                    logging.info('load from cache, Network id = {}'.format(arch_id))
            else: 
                new_args = copy(args)
                new_args.genome = genome
                new_args.save = indi_dir
                swn.train_share_subnet(new_args)
            self._n_evaluated += 1
            # p.wait()
        
        tnt.wait()
        for idx, genome in enumerate(POP):
            performance = gnr.read(genome)
            
            objs[idx, 0] = performance['err']
            objs[idx, 1] = performance['flops']

            
            # time.sleep(1) # wait to release the gpu resources
        F['F'] = objs
        

def minimize_hook(algo):
    gen = algo.n_gen
    global GEN_ITER
    GEN_ITER = gen
    pop_var = algo.pop.get('X')
    pop_obj = algo.pop.get('F')
    
    # for indi in pop_var:
    #     for code in indi:
    #         print('%.2f'%(code), end=',')
    #     print()
    logging.info('generation = {}'.format(gen))
    logging.info('population error: best = {}, mean = {}, '
                 'median = {}, worst = {}'.format(np.min(pop_obj[:,0]), np.mean(pop_obj[:,0]),
                                                  np.median(pop_obj[:,0]), np.max(pop_obj[:,0])))

    logging.info('population complexity: best = {}, mean = {}, '
                 'median = {}, worst = {}'.format(np.min(pop_obj[:,1]), np.mean(pop_obj[:,1]),
                                                  np.median(pop_obj[:,1]), np.max(pop_obj[:,1])))


def main(args):
    # 1. reset the random seed
    utils.reset_random_seed(args)
    logging.info(args.__str__())
    # 2. initialize the lower bound and upper bound
    # l1-norm, l2-norm, geometry median, emprical senstivity
    num_criterion = 4
    n_depth = get_pruning_strategy(args).n_depth
    n_var = int((1+1)*n_depth)
    # lb = np.zeros(n_var)
    lb_prob = np.ones((n_depth,1))*args.min_ratio
    
    if args.cosine_anneal_bound is not None: 
        def func(min_,init_,max_t,t):
            return min_ + .5* (init_ - min_) * (1+ np.cos(t/max_t*np.pi))
        
        lb_prob = np.array([ func(0.0,args.cosine_anneal_bound,n_depth,i) for i in range(n_depth) ][::-1])[:,None]
    
    if args.anneal_bound is not None:
        if n_depth == 55:
            lb_prob = np.array([*[.2]*18,*[.4]*18,*[.6]*18,.0])[:,None]
        else: 
            assert 0
    lb_cri = np.zeros((n_depth,1))
    lb = np.concatenate((lb_prob,lb_cri),axis=1).reshape(-1)
    ub = np.ones(n_var)

    # 3. define the problem and optimize it
    problem = EvoPruneProblem(n_vars=n_var, n_obj=2, n_constr=0, lb=lb, ub=ub, config=args)
    method = evoprune(pop_size=args.pop_size, n_offsprings=args.n_offsprings, eliminate_duplicates=True)
    res = minimize(problem, method, callback=minimize_hook, termination=('n_gen', args.n_gens), seed=args.seed)


if __name__ == '__main__':
    parser = get_opt()
    args = parser.parse_args()

    # extra manipulation on args
    # args.save = utils.create_exp_dir(args.root, 'pruning-search', args.save)
    if args.save is None:
        args.save = utils.create_exp_dir(args.root, 'pruning-search', '-'.join(
            [v for v in [args.dataset, args.arch+str(args.depth), 'pop'+str(args.pop_size), 
            'gen'+str(args.n_gens), 'weight_sharing' if args.ws else '', f'min_ratio_{args.min_ratio}',
            f'epochs{args.epochs}', 'indi_bn' if args.indi_bn else '', 'pretrain' if args.pretrain is not None else '',
            f'cosine_anneal_bound{args.cosine_anneal_bound}' if args.cosine_anneal_bound is not None else "",
            'anneal_bound' if args.anneal_bound is not None else "",
            f'ws_layer_cnt{args.ws_layer_cnt}' if args.ws_layer_cnt is not None else "",
            f'cri_{args.criterion}' if args.criterion != 'default' else "",
            ] if v != '']), args)

    gnr = GNR(args.save)


    # global logger
    logging = Logger(0, 'evolutionary pruning rate assignment', args.save)
    
    # global writter
    writer = Writer(0, args.save)
    
    args.writer = writer
    args.logging = logging

    args.distributed = False
    if args.ws is True:
        # import pdb;pdb.set_trace()
        swn = WSPruningTrainer(args)

    GEN_ITER=0
    # main entry
    main(args)