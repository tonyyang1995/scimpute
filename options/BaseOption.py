import argparse
import os
import torch

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False
    
    def initialize(self):
        self.parser.add_argument('--gpu_ids', type=str, default='0,1')
    
    def print_options(self, opt):
        message = ''
        message += '------------------------ OPTIONS -----------------------------\n'
        for k,v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}\n'.format(str(k), str(v), comment)
        message += '------------------------  END   ------------------------------\n'
        print(message)
        expr_dir = os.path.join(opt.checkpoint_dir, opt.name)
        if not os.path.exists(expr_dir):
            os.makedirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):
        # parse the options, create checkpoint, and set up device
        if not self.initialized:
            self.initialize()

        self.opt = self.parser.parse_args()

        os.environ['CUDA_VISIBLE_DEVICES'] = self.opt.gpu_ids
        # set gpu ids
        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)
        if len(self.opt.gpu_ids) > 0:
            self.opt.device = torch.device('cuda')
        else:
            self.opt.device = torch.device('cpu')

        self.print_options(self.opt)
        return self.opt