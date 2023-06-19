import numpy as np
import os, sys, time
import torch
import torch.multiprocessing as mp

import utils.options as options
from utils.util import log, is_port_in_use
import model.runner

def main_worker(rank, world_size, port, opt):

    opt.device = rank
    opt.world_size = world_size
    opt.port = port
    
    trainer = model.runner.Runner(opt)

    trainer.load_dataset(opt)
    trainer.build_networks(opt)
    trainer.setup_optimizer(opt)
    trainer.restore_checkpoint(opt)
    trainer.setup_visualizer(opt)

    trainer.train(opt)

def main():
    log.process(os.getpid())
    log.title("[{}] (training)".format(sys.argv[0]))

    opt_cmd = options.parse_arguments(sys.argv[1:])
    opt = options.set(opt_cmd=opt_cmd)
    options.save_options_file(opt)

    port = 34567
    while is_port_in_use(port):
        port += 1
    world_size = torch.cuda.device_count()
    if world_size == 1:
        main_worker(0, world_size, port, opt)
    else:
        mp.spawn(main_worker, nprocs=world_size, args=(world_size, port, opt))

if __name__ == "__main__":
    main()