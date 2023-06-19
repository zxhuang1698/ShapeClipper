import os, sys
import torch

import utils.options as options
from utils.util import log, is_port_in_use
import model.runner

log.process(os.getpid())
log.title("[{}] (evaluating)".format(sys.argv[0]))

opt_cmd = options.parse_arguments(sys.argv[1:])
opt = options.set(opt_cmd=opt_cmd)
port = 34567
while is_port_in_use(port):
    port += 1
opt.device = 0
opt.world_size = 1
opt.port = port

with torch.cuda.device(opt.device):

    evaluator = model.runner.Runner(opt)
    split = "test"
    evaluator.load_dataset(opt, eval_split=split)
    evaluator.test_data.id_filename_mapping(opt, os.path.join(opt.output_path, 'data_list.txt'))
    evaluator.build_networks(opt)
    evaluator.restore_checkpoint(opt, best=True, evaluate=True)
    evaluator.setup_visualizer(opt)

    evaluator.evaluate(opt, ep=0)
