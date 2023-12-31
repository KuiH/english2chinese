import json
from typing import *

import torch


class Tokenizer:
    token_ids_table = None
    special_tokens = None

    def __init__(self):
        with open(r"config/token_ids_table.json", 'r', encoding='utf-8') as f:
            Tokenizer.token_ids_table = json.load(f)
        with open(r"config/special_tokens.json", 'r', encoding='utf-8') as f:
            Tokenizer.special_tokens = json.load(f)
        self.bos = self.special_tokens["bos"]
        self.pad = self.special_tokens["pad"]
        self.eos = self.special_tokens["eos"]
        self.unk = self.special_tokens["unk"]
        self.bos_ids = self.token_ids_table[self.bos]
        self.pad_ids = self.token_ids_table[self.pad]
        self.eos_ids = self.token_ids_table[self.eos]
        self.unk_ids = self.token_ids_table[self.unk]
        self.__en_ids_token_table = self.__ids_token_table("en")
        self.__zh_ids_token_table = self.__ids_token_table("zh")

    def encode(self, token_list: List[str], lang_type: str, to_length: int = -1) -> List[int]:
        """
        因为数据集本身已经是List了，所以这里就传入list. 按道理是传入str.
        to_length: encode之后的长度。=-1表示不限制长度
        """
        assert lang_type in ("en", "zh")
        to_length = len(token_list) if to_length == -1 else to_length
        temp_list = token_list.copy()
        temp_list = temp_list[:to_length]
        temp_list += [self.pad] * (to_length - len(temp_list))
        ids_list: List[int] = []
        for i, token in enumerate(temp_list):
            if token in (self.bos, self.pad, self.eos, self.unk):
                ids_list.append(self.token_ids_table[token])
            else:  # 非特殊字符
                if token in self.token_ids_table[lang_type]:
                    ids_list.append(self.token_ids_table[lang_type][token])
                else:  # 没见过的token
                    ids_list.append(self.token_ids_table[self.unk])
        return ids_list

    def decode(self, ids_list: Union[torch.Tensor, List], lang_type: str):
        """ids list解码成str"""
        decoded_list = []
        res_str = ""
        assert lang_type in ("en", "zh")
        if isinstance(ids_list, torch.Tensor):
            if ids_list.device.type == 'cuda':  # 在gpu
                copy_list = ids_list.cpu().numpy().tolist()
            else:
                copy_list = ids_list.numpy().tolist()
        else:  # 是List
            copy_list = ids_list

        if lang_type == "en":
            for i in copy_list:
                decoded_list.append(self.__en_ids_token_table[i])
        else:
            for i in copy_list:
                decoded_list.append(self.__zh_ids_token_table[i])
        res_str = " ".join(decoded_list)
        return res_str

    def vocab_size(self, lang_type: str):
        assert lang_type in ("en", "zh")
        return len(self.__en_ids_token_table) if lang_type == "en" else len(self.__zh_ids_token_table)

    def __ids_token_table(self, lang_type: str):
        assert lang_type in ("en", "zh")
        lang_token_ids = self.token_ids_table[lang_type]
        lang_token_ids[self.bos] = self.token_ids_table[self.bos]
        lang_token_ids[self.pad] = self.token_ids_table[self.pad]
        lang_token_ids[self.eos] = self.token_ids_table[self.eos]
        lang_token_ids[self.unk] = self.token_ids_table[self.unk]
        return {k: v for v, k in lang_token_ids.items()}


if __name__ == '__main__':
    tokenizer = Tokenizer()
    token_list = ["i", "am", "a", "aabbccddee", "student", "."]
    s1 = tokenizer.encode(token_list, lang_type="en", to_length=4)
    s2 = tokenizer.encode(token_list, lang_type="en")
    s3 = tokenizer.encode(token_list, lang_type="en", to_length=10)
    print(tokenizer.decode(s1, lang_type="en"))
    print(tokenizer.decode(s2, lang_type="en"))
    print(tokenizer.decode(s3, lang_type="en"))

    token_list = ['是', '湯', '姆', '救', '了', '這', '個', '二', 'hi', '发', '货', '小', '女', '孩', '。']
    s1 = tokenizer.encode(token_list, lang_type="zh", to_length=7)
    s2 = tokenizer.encode(token_list, lang_type="zh")
    s3 = tokenizer.encode(token_list, lang_type="zh", to_length=18)
    s1 = torch.tensor(s1)
    s2 = torch.tensor(s2)
    s3 = torch.tensor(s3)
    print(tokenizer.decode(s1, lang_type="zh"))
    print(tokenizer.decode(s2, lang_type="zh"))
    print(tokenizer.decode(s3, lang_type="zh"))

    print(tokenizer.vocab_size("en"))
    print(tokenizer.vocab_size("zh"))
