import time
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils import data
from typing import *

from config.model_config import CommonConfig, LSTMConfig
from model.LSTM import Seq2SeqLSTM
from utils.dataloader import train_loader, valid_loader, test_loader
from utils.mask import MaskedCrossEntropyLoss
from utils.common import *
from utils.tokenizer import Tokenizer
from utils.metric import bleu

setup_seed(2333)
epoch = CommonConfig.epoch
lr = CommonConfig.lr
cuda = CommonConfig.cuda
batch_size = CommonConfig.batch_size
model_name = "LSTM"
tokenizer = Tokenizer()
vocab_size_en, vocab_size_zh = tokenizer.vocab_size("en"), tokenizer.vocab_size("zh")
embed_size_en, embed_size_zh = LSTMConfig.embed_size_en, LSTMConfig.embed_size_zh
num_layers_en, num_layers_zh = LSTMConfig.num_layers_en, LSTMConfig.num_layers_zh
hidden_size, pad_ids, dropout = LSTMConfig.hidden_size, tokenizer.pad_ids, CommonConfig.dropout

model = Seq2SeqLSTM(vocab_size_en=vocab_size_en, vocab_size_zh=vocab_size_zh, embed_size_en=embed_size_en,
                    embed_size_zh=embed_size_zh, num_hiddens=hidden_size, num_layers_zh=num_layers_zh,
                    num_layers_en=num_layers_en, pad_ids=pad_ids, dropout=dropout)
if cuda:
    model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = MaskedCrossEntropyLoss()
log_name = f"{model_name}_epoch{epoch}_lr{lr}_hid{hidden_size}_bs{batch_size}_drop{dropout}"
writer = SummaryWriter(log_dir=rf'logs/{log_name}')


def test_model(dataloader: data.DataLoader):
    """返回loss"""
    start = time.time()
    model.eval()
    with torch.no_grad():
        loss = 0.0
        total = 0
        for batch_src, batch_dec_input, batch_tgt, valid_lens in dataloader:
            total += batch_src.shape[0]
            out = model(batch_src, batch_dec_input)  # out: [bs, seq_len, vocab_size]
            # print(out.shape, batch_tgt.shape, valid_lens.shape)
            batch_loss = criterion(out[:, :-1, :], batch_tgt,
                                   valid_lens)  # out的seq_len包含开头的bos，但是实际上算损失时没有包含bos。所以这里少取一个
            # print(batch_loss.shape)
            loss += batch_loss.sum()
    end = time.time()
    print(f"test model uses time: {end - start}")
    return loss / total


def report_res(dataloader: data.DataLoader):
    """计算指标"""

    def batch_str(batch_ids_list) -> List[str]:
        """获取一个batch ids对应的str"""
        out_strs = []
        for batch_ids in batch_ids_list:  # 这样可以遍历batch
            out_str = tokenizer.decode(batch_ids, lang_type="zh")
            eos_index = out_str.find(tokenizer.eos)
            out_str = out_str[:len(out_str) if eos_index < 0 else eos_index]
            out_strs.append(out_str)
        return out_strs

    start = time.time()
    model.eval()
    with torch.no_grad():
        total = 0
        sum_bleu = 0.0
        for batch_src, batch_dec_input, batch_tgt, valid_lens in dataloader:
            total += batch_src.shape[0]
            out = model(batch_src, batch_dec_input)  # out: [bs, seq_len, vocab_size]
            out_seqs = torch.argmax(out, dim=2)  # out_seq: [bs, seq_len]
            out_strs: List[str] = batch_str(out_seqs)
            tgt_strs: List[str] = batch_str(batch_tgt)
            bleu_scores = bleu(reference=tgt_strs, candidate=out_strs)

            # TODO: bleu求和
            sum_bleu += sum(bleu_scores)
    end = time.time()
    print(f"report result uses time: {end - start}")
    return sum_bleu / total


def main():
    print("begin training...")
    start = time.time()
    for epoch_ in range(epoch):
        ticket = time.time()
        test_loss = test_model(test_loader)
        valid_loss = test_model(valid_loader)
        append_log_valid_test(valid_loss, test_loss, epoch_, writer)
        model.train()
        epoch_loss = 0
        total = 0
        for batch_src, batch_dec_input, batch_tgt, valid_lens in train_loader:
            total += batch_src.shape[0]
            batch_out = model(batch_src, batch_dec_input)
            batch_loss = criterion(batch_out[:, :-1, :], batch_tgt,
                                   valid_lens)  # 这里batch_loss是一个一维tensor，不是数值。因为设置了reduction = 'none'
            optimizer.zero_grad()
            batch_loss.sum().backward()
            optimizer.step()
            epoch_loss += batch_loss.sum()
        end = time.time()
        append_log_train(epoch_loss / total, epoch, writer)
        print(f"epoch {epoch_}, loss: {epoch_loss / total}, time: {end - ticket}")
    end = time.time()
    print(f"train uses time: {end - start}")
    torch.save(model.state_dict(), rf'checkpoints/{log_name}.ckpt')

    print("train set result:")
    report_res(train_loader)
    print("valid set result:")
    report_res(valid_loader)
    print("test set result:")
    report_res(test_loader)


if __name__ == '__main__':
    main()
