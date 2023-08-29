class CommonConfig:
    batch_size = 64 # 128
    cuda = True
    lr = 1e-4
    dropout = 0.1
    epoch = 50


class LSTMConfig:
    embed_size_en = 64  # 1024
    embed_size_zh = 64  # 2048
    num_layers_en = 2
    num_layers_zh = 2
    hidden_size = 64  # 512
