# Author: Weinan He
# Mail: Sapphire9877@gmail.com
# ----------------------------------------------


import argparse
import os
import yaml
import easydict

# config
parser = argparse.ArgumentParser(description="config")
parser.add_argument('--cfg', type=str, required=True, default="configs/unida.yaml", help="Relative path of config file, such as 'configs/unida.yaml'")
parser.add_argument('--gpu', default=None, help="Overwrite the specified gpu device in `--cfg`")
args = parser.parse_args()

# path
dir_root = os.path.dirname(os.path.abspath(__file__))
data_root = os.path.join(dir_root, 'data')
config_file = os.path.join(dir_root, args.cfg)

conf = yaml.load(open(config_file), Loader=yaml.FullLoader)
conf = easydict.EasyDict(conf)

devices = args.gpu if args.gpu is not None else conf.gpu
os.environ["CUDA_VISIBLE_DEVICES"] = devices
print(f"set CUDA_VISIBLE_DEVICES={devices}")


import ssl
import sys
import time
import shutil
import torch
import numpy as np
import random
from lightning import Fabric
from torch.utils.tensorboard import SummaryWriter
import logging


from train_sapphire_TASC import TASCTrainer


METHOD = {
    'TASC': TASCTrainer,
}



if __name__ == '__main__':

    # pretrained models path
    os.environ['TORCH_HOME'] = os.path.join(dir_root, '..', 'pretrained_models')
    ssl._create_default_https_context = ssl._create_unverified_context

    # gpu
    if not torch.cuda.is_available():
        raise Exception('CUDA is not available!!!')

    # dataset
    dataset_name = conf.dataset.name  
    task = conf.dataset.task

    shared = conf.dataset.shared
    source_private = conf.dataset.source_private
    target_private = conf.dataset.target_private

    split_str = '_{}-{}-{}_{}'.format(shared, source_private, target_private, conf.method)

    # seed
    if conf.control.seed.flag == True:
        SEED = conf.control.seed.seed
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)
        torch.backends.cudnn.deterministic = True

    # Tensorboard writer
    print("Build writer and copy source code")
    writer_name = conf.exp_name
    if writer_name == 'x':
        writer_name = ''
    if writer_name != '':
        writer_name = '_' + writer_name
    writer_name = split_str+writer_name
    timestamp = time.strftime('%Y_%m%d_%H%M%S', time.localtime())
    logdir = os.path.join('output', dataset_name, 'runs_'+timestamp+'_'+dataset_name+'_'+task+writer_name)
    writer = SummaryWriter(log_dir=logdir)
    # copy source code and configs to logdir
    if not os.path.exists(logdir+'/config'):
        os.makedirs(logdir+'/config')
    shutil.copy(config_file, logdir+'/config/conf.yaml')
    shutil.copytree('./sapphire', logdir+'/code/sapphire')
    base_files = [filename for filename in os.listdir('.') 
                  if os.path.isfile(filename)] #and "Base" in filename]
    shutil.copy(f'./train_sapphire.py', logdir+f'/code/train_sapphire.py')
    for filename in base_files:
        shutil.copy(f'./{filename}', logdir+f'/code/{filename}')
    shutil.copy(f'./train_sapphire_{conf.method}.py', logdir+f'/code/train_sapphire_{conf.method}.py')
    print(f"Copy train_sapphire_{conf.method}.py")

    # Redirect Log
    class Logger(object):
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, "w")
    
        def write(self, message):
            self.log.write(message)
            self.terminal.write(message)
            self.log.flush()
        
        def flush(self):
            pass

    log_out = Logger(logdir+'/log.out')
    log_err = Logger(logdir+'/log.err')
    sys.stdout = log_out
    sys.stderr = log_err

    dir_dict = {
        'data': data_root,
        'project': dir_root,
        'log': logdir,
    }

    # logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging._nameToLevel[conf.logger])
    
    file_handler = logging.FileHandler(logdir+'/log.log', mode='w')
    
    logger.addHandler(file_handler)
    formatter = logging.Formatter(fmt='%(asctime)s %(name)s [%(levelname)s]: %(message)s', \
                                  datefmt='%d-%b-%y %H:%M:%S')
    file_handler.setFormatter(formatter)
    logger.info("Start logging")

    if conf.logger_console == True:
        console_handler = logging.StreamHandler()
        logger.addHandler(console_handler)
        console_handler.setFormatter(formatter)

    # Trainer
    print('Build Trainer')
    trainer = METHOD[conf.method](conf, writer, dir_dict, logger)

    # model
    print('Build model')
    trainer.build_model()

    # dataset
    print('Build Dataset')
    trainer.build_data_loaders()               

    # optimizer
    print('Build optimizer')
    trainer.build_optimizer()

    # lr
    print('Build lr_schedule')
    trainer.build_lr_schedule()

    # pytorch lightning
    if hasattr(conf, 'lightning') and conf.lightning == True:
        precision = "bf16-mixed" if torch.cuda.is_bf16_supported() else "16-mixed"
        print(f'Starting pytorch lightning, precision: {precision}')
        fabric = Fabric(accelerator="cuda", devices='auto', precision=precision)  # '16-mixed', 'bf16-mixed'
        try: torch.set_float32_matmul_precision("high")  # 'medium' | 'high'
        except: print("'torch' has no attribute 'set_float32_matmul_precision' (torch <1.12)")
        fabric.launch()

    # **********
    print()
    print('-------------------')
    print()
    print('Start Time: '+timestamp)
    print('Job: {} {} {}/{}/{}'.format(dataset_name, task, shared, source_private, target_private)) 
    print(f"Method: {conf.method}")
    print('GPU: {}'.format(devices))
    print('Logdir: {}'.format(logdir))
    print()
    print('-------------------')
    print()

    # train
    print('Start training!')

    trainer.train()
            
    writer.close()
    print()

    timestamp = time.strftime('%Y_%m%d_%H%M%S', time.localtime())
    print('Time: '+timestamp)
    print('torch.cuda.max_memory_allocated: ', torch.cuda.max_memory_allocated())
    print('Done!')
