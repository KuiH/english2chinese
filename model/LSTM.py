import torch
import torch.nn as nn
from typing import *


class Seq2SeqEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, pad_ids, dropout=0.):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size, padding_idx=pad_ids)
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=num_hiddens, num_layers=num_layers, dropout=dropout,
                            batch_first=True)

    def forward(self, x):
        x = self.emb(x)  # x: bs, embed_size
        output, (hn, cn) = self.lstm(x)  # output: (bs, seq_len, num_hiddens), hn:(num_layers, bs, num_hiddens)
        return output, (hn, cn)


class Seq2SeqDecoder(nn.Module):
    # num_hiddens要和Encoder相同
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, pad_ids, dropout=0.):
        super().__init__()
        self.num_layers = num_layers
        self.emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size, padding_idx=pad_ids)
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=num_hiddens, num_layers=num_layers, dropout=dropout,
                            batch_first=True)
        self.fc = nn.Linear(in_features=num_hiddens, out_features=vocab_size)

    def forward(self, x, enc_state):
        """enc_state是encoder的(hn,cn)"""
        x = self.emb(x)
        enc_hn = enc_state[0][-self.num_layers:]  # 用切片可以保留维度
        enc_cn = enc_state[1][-self.num_layers:]
        output, state = self.lstm(x, (enc_hn, enc_cn))
        output = self.fc(output)  # [bs, seq_len, vocab_size]
        return output, state


class Seq2SeqLSTM(nn.Module):
    def __init__(self, vocab_size_en, vocab_size_zh, embed_size_en, embed_size_zh, num_layers_en, num_layers_zh,
                 num_hiddens, pad_ids, dropout=0.):
        super().__init__()
        assert num_layers_en >= num_layers_zh
        self.encoder = Seq2SeqEncoder(vocab_size=vocab_size_en, embed_size=embed_size_en, num_hiddens=num_hiddens,
                                      num_layers=num_layers_en, pad_ids=pad_ids, dropout=dropout)
        self.decoder = Seq2SeqDecoder(vocab_size=vocab_size_zh, embed_size=embed_size_zh, num_hiddens=num_hiddens,
                                      num_layers=num_layers_zh, pad_ids=pad_ids, dropout=dropout)

    def forward(self, en_x, zh_x):
        _, state = self.encoder(en_x)
        output, _ = self.decoder(zh_x, state)
        return output


if __name__ == '__main__':
    vocab_size_en, vocab_size_zh = 10, 20
    embed_size_en, embed_size_zh = 5, 8
    num_layers_en, num_layers_zh = 2, 2
    hidden_size, pad_ids, dropout = 3, 0, 0.1
    seq2seq = Seq2SeqLSTM(vocab_size_en=vocab_size_en, vocab_size_zh=vocab_size_zh, embed_size_en=embed_size_en,
                          embed_size_zh=embed_size_zh, num_hiddens=hidden_size, num_layers_zh=num_layers_zh,
                          num_layers_en=num_layers_en, pad_ids=pad_ids, dropout=dropout)
    en_x = torch.tensor([[1,2,3,4,0],[5,2,6,0,0],[4,5,5,6,7]])
    zh_x = torch.tensor([[5,15,2,0],[14,13,0,0],[1,2,3,3]])
    out = seq2seq(en_x, zh_x)
    print(out.shape)
