import os, sys
import torch

import utils.options as options
from utils.util import log
import model.pretrainer

log.process(os.getpid())
log.title("[{}] (training)".format(sys.argv[0]))

opt_cmd = options.parse_arguments(sys.argv[1:])
opt = options.set(opt_cmd=opt_cmd)
options.save_options_file(opt)

with torch.cuda.device(opt.device):

    trainer = model.pretrainer.Runner(opt)

    trainer.load_dataset(opt)
    trainer.build_networks(opt)
    trainer.setup_optimizer(opt)

    trainer.train(opt)
