import json

from utils.load_data import load_data

"""
给中英语言构造token->ids映射，保存在config中。
"""
data = load_data(r"../dataset/en_zh.pkl")
lang = {"en": 0, "zh": 1}


def token_ids_table(lang_type: str):
    assert lang_type in lang
    tokens = set()
    for d in data:
        sentence = d[lang[lang_type]]
        for token in sentence:
            tokens.add(token)
    tokens = sorted(tokens, key=lambda x: len(x))  # tokens会变为List
    ids = 4  # 保留bos, eos, pad, unk的位置
    res = {}
    for token in tokens:
        res[token] = ids
        ids += 1
    return res


if __name__ == '__main__':
    token_ids = {"<bos>": 0, "<pad>": 1, "<eos>": 2, "<unk>": 3}
    en_table = token_ids_table("en")
    zh_table = token_ids_table("zh")
    token_ids["en"] = en_table
    token_ids["zh"] = zh_table
    res_path = r"../config/token_ids_table.json"
    with open(res_path, "w", encoding='utf-8') as f:
        json.dump(token_ids, f, ensure_ascii=False, indent=2)
