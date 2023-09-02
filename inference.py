from utils.tokenizer import Tokenizer
from model.LSTM import Seq2SeqLSTM
import torch

from utils.common import segment_en
from config.model_config import LSTMConfig, CommonConfig

tokenizer = Tokenizer()
vocab_size_en, vocab_size_zh = tokenizer.vocab_size("en"), tokenizer.vocab_size("zh")
embed_size_en, embed_size_zh = LSTMConfig.embed_size_en, LSTMConfig.embed_size_zh
num_layers_en, num_layers_zh = LSTMConfig.num_layers_en, LSTMConfig.num_layers_zh
hidden_size, pad_ids, dropout = LSTMConfig.hidden_size, tokenizer.pad_ids, CommonConfig.dropout

model = Seq2SeqLSTM(vocab_size_en=vocab_size_en, vocab_size_zh=vocab_size_zh, embed_size_en=embed_size_en,
                    embed_size_zh=embed_size_zh, num_hiddens=hidden_size, num_layers_zh=num_layers_zh,
                    num_layers_en=num_layers_en, pad_ids=pad_ids, dropout=dropout)
cuda = CommonConfig.cuda
device = "cpu"
if cuda:
    model = model.cuda()
    device = "cuda"

checkpoint = r"checkpoint path"
model.load_state_dict(torch.load(checkpoint))

model.eval()


def do_infer(input_en: str):
    token_list = segment_en(input_en.lower())
    src_ids = torch.tensor([tokenizer.bos_ids] + tokenizer.encode(token_list, lang_type="en"), dtype=torch.long,
                           device=device)
    _, enc_state = model.encoder(src_ids)
    dec_input = torch.tensor([tokenizer.bos_ids], dtype=torch.long, device=device)
    dec_state = enc_state
    # print(src_ids.shape, dec_input.shape)
    out_zh = ""
    while len(out_zh) < 50:
        dec_out, dec_state = model.decoder(dec_input, dec_state)
        pred_id = torch.argmax(dec_out, dim=1)
        out_token = tokenizer.decode(pred_id, lang_type="zh")
        if out_token == tokenizer.eos:
            break
        out_zh += out_token
        dec_input = pred_id
    return out_zh


if __name__ == '__main__':
    in_str = "I think so."
    pred = do_infer(in_str)
    print(pred)
