# 因为不想租服务器，所以就只能设置得小一点，在自己电脑上跑..

class CommonConfig:
    batch_size = 16  # 128
    cuda = True
    lr = 1e-4
    dropout = 0.1
    epoch = 30


class LSTMConfig:
    embed_size_en = 768  # 1024
    embed_size_zh = 512  # 2048
    num_layers_en = 2
    num_layers_zh = 2
    hidden_size = 384  # 512
