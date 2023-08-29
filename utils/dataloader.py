import random
from torch.utils import data
from typing import *
import torch
from utils.common import load_data, setup_seed
from utils.tokenizer import Tokenizer
from torch.nn.utils.rnn import pad_sequence
from config.model_config import CommonConfig

setup_seed(2333)
tokenizer = Tokenizer()


class MyDataset(data.Dataset):
    def __init__(self, data: List[Tuple]):
        self.x = [d[0] for d in data]
        self.y = [d[1] for d in data]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        """这里返回tokenized的结果"""
        en = tokenizer.encode(self.x[index], lang_type="en")
        zh = tokenizer.encode(self.y[index], lang_type="zh")
        zh = [tokenizer.bos_ids] + zh + [tokenizer.eos_ids]  # 中文(decoder相关部分)加上bos和eos
        en = torch.tensor(en)
        zh = torch.tensor(zh)
        return en, zh


def collate_fn(batch: List[Tuple]):
    #  batch是一个列表，其中是一个一个的元组，每个元组是dataset中_getitem__的结果
    src_ids = [b[0] for b in batch]
    dec_input_ids = [b[1] for b in batch]  # decoder的输入，带有bos
    tgt_ids = [b[1][1:] for b in batch]  # decoder的label，不带bos
    pad_src = pad_sequence(src_ids, batch_first=True, padding_value=tokenizer.pad_ids)  # 原句子
    pad_dec_input = pad_sequence(dec_input_ids, batch_first=True, padding_value=tokenizer.pad_ids)  # decoder的输入
    pad_tgt = pad_sequence(tgt_ids, batch_first=True, padding_value=tokenizer.pad_ids)  # 目标句子

    # print(tokenizer.decode(pad_src[2].numpy(), lang_type="en"))
    # print(tokenizer.decode(pad_dec_input[2].numpy(), lang_type="zh"))
    # print(tokenizer.decode(pad_tgt[2].numpy(), lang_type="zh"))

    # 找pad_tgt的有效长度，即没有pad的部分的长度
    valid_lens = []
    for tgt in pad_tgt:
        for i in range(len(tgt) - 1, -1, -1):
            if tgt[i] != tokenizer.pad_ids:
                valid_lens.append(i + 1)
                break
    valid_lens = torch.tensor(valid_lens,dtype=torch.long)
    if CommonConfig.cuda:
        pad_src = pad_src.cuda()
        pad_dec_input = pad_dec_input.cuda()
        pad_tgt = pad_tgt.cuda()
        valid_lens = valid_lens.cuda()
    return pad_src, pad_dec_input, pad_tgt, valid_lens


# 切分数据集
def split_dataset():
    total_num = 23455
    test_part = valid_part = 0.15
    valid_num = int(total_num * valid_part)
    test_num = int(total_num * test_part)
    train_num = total_num - valid_num - test_num
    total_data = load_data(r"dataset/en_zh.pkl")
    random.shuffle(total_data)
    train_data = total_data[:train_num]
    valid_data = total_data[train_num:valid_num + train_num]
    test_data = total_data[valid_num + train_num:test_num + valid_num + train_num]
    train_set = MyDataset(train_data)
    valid_set = MyDataset(valid_data)
    test_set = MyDataset(test_data)
    return train_set, valid_set, test_set


train_set, valid_set, test_set = split_dataset()
train_loader = data.DataLoader(dataset=train_set, batch_size=CommonConfig.batch_size, shuffle=True,
                               collate_fn=collate_fn)
valid_loader = data.DataLoader(dataset=valid_set, batch_size=CommonConfig.batch_size, shuffle=False,
                               collate_fn=collate_fn)
test_loader = data.DataLoader(dataset=test_set, batch_size=CommonConfig.batch_size, shuffle=False,
                              collate_fn=collate_fn)

if __name__ == '__main__':
    for a,b,c,d in test_loader:
        print(c)
        print(d)
        break
