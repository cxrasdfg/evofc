# coding=utf-8

import sys, os, time
from tensorboardX import SummaryWriter
import logging


class Logger(object):
    def __init__(self, rank, name, save):
        self.rank = rank
        if self.rank == 0:
            log_format = '%(asctime)s %(message)s'
            logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                                format=log_format, datefmt='%m/%d %I:%M:%S %p')
            fh = logging.FileHandler(os.path.join(save, 'log.txt'), mode='w')
            fh.setFormatter(logging.Formatter(log_format))

            self.logger = logging.getLogger(name)
            self.logger.addHandler(fh)

            self.start_time = time.time()

    def info(self, string, *args):
        if self.rank == 0:
            elapsed_time = time.time() - self.start_time
            elapsed_time = time.strftime(
                '(Elapsed: %H:%M:%S) ', time.gmtime(elapsed_time))
            if isinstance(string, str):
                string = elapsed_time + string
            else:
                self.logger.info(elapsed_time)
            self.logger.info(string, *args)
    
    def shutdown(self):
        x = self.logger.handlers.copy()
        for i in x:
            self.logger.removeHandler(i)
            i.flush()
            i.close()
        self.logger.handlers.clear()

    def __del__(self):
        self.shutdown()

class Writer(object):
    def __init__(self, rank, save):
        self.rank = rank
        if self.rank == 0:
            save = os.path.join(save, 'tensorboard-log')
            self.writer = SummaryWriter(log_dir=save, flush_secs=20)

    def add_scalar(self, *args, **kwargs):
        if self.rank == 0:
            self.writer.add_scalar(*args, **kwargs)

    def add_figure(self, *args, **kwargs):
        if self.rank == 0:
            self.writer.add_figure(*args, **kwargs)

    def add_image(self, *args, **kwargs):
        if self.rank == 0:
            self.writer.add_image(*args, **kwargs)

    def add_histogram(self, *args, **kwargs):
        if self.rank == 0:
            self.writer.add_histogram(*args, **kwargs)

    def add_histogram_if(self, write, *args, **kwargs):
        if write and False:   # Used for debugging.
            self.add_histogram(*args, **kwargs)

    def close(self, *args, **kwargs):
        if self.rank == 0:
            self.writer.close()