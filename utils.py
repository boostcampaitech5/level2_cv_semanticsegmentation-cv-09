import random
import torch
import numpy as np
import os

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def get_gpu():
    GB = 1024.*1024.0*1024.0
    return round(torch.cuda.max_memory_allocated() / GB, 1)

def save_model(model, save_dir, file_name='fcn_resnet50_best_model.pt'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    output_path = os.path.join(save_dir, file_name)
    print(f"Save model in {output_path}")
    torch.save(model, output_path)