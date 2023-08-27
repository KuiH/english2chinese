import torch
import torch.nn as nn


class Seq2SeqEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, pad_ids, dropout=0):
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
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, pad_ids, dropout=0):
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
        output = self.fc(output)
        return output


if __name__ == '__main__':
    encoder = Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=3, num_layers=2, pad_ids=1)
    X = torch.zeros((12, 7), dtype=torch.long)
    output, state = encoder(X)
    # print(output.size(), state[0].size(), state[1].size())
    decoder = Seq2SeqDecoder(vocab_size=20, embed_size=13, num_hiddens=3, num_layers=1, pad_ids=1)
    X = torch.zeros((12, 7), dtype=torch.long)
    output = decoder(X, state)
    print(output.shape)
