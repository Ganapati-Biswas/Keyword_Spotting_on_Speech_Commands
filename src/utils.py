import os
import random
import numpy as np
import torch
import yaml
import subprocess
import logging
from datetime import datetime

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def save_config(config: dict, path: str):
    with open(path, 'w') as f:
        yaml.safe_dump(config, f)

def get_git_hash() -> str:
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()
    except Exception:
        return ""

def create_experiment_dir(base_dir: str, exp_name: str) -> str:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    dir_path = os.path.join(base_dir, f"{exp_name}_{timestamp}")
    os.makedirs(dir_path, exist_ok=True)
    return dir_path

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self, name: str, fmt: str = ':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_logger(name: str, log_file: str = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
    return logger
