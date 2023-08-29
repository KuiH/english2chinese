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


def append_log_train(train_loss, epoch, writer):
    writer.add_scalar('1_train/train_loss', train_loss, epoch)


def append_log_valid_test(v_loss,  t_loss,  epoch, writer):
    writer.add_scalar('2_valid/loss', v_loss, epoch)
    writer.add_scalar('3_test/loss', t_loss, epoch)
