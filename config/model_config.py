class CommonConfig:
    def __init__(self):
        self.batch_size = 128
        self.cuda = True
        self.lr = 1e-4
        self.dropout = 0.1


class LSTMConfig:
    def __init__(self):
        self.embed_size_en = 1024
        self.embed_size_zh = 2048
        self.num_layers_en = 2
        self.num_layers_zh = 2
        self.hidden_size = 512


common_config = CommonConfig()
lstm_config = LSTMConfig()
