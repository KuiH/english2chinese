import torch
import numpy as np
import random
import pickle


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_data(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data
