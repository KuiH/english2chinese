from typing import *
import pickle

"""
制作pickle之后的数据。格式:
[(en0,zh0),(en1,zh1),...]
"""


def en_zh_pairs():
    """从数据集构造(英文，中文)对。如果英文相同，只取第一次出现的"""
    with open("../dataset/en_zh.txt", 'r', encoding='utf-8') as f:
        data = f.readlines()
        data = [d.split('\t') for d in data]
        pairs: List[Tuple] = [(d[0], d[1]) for d in data]
    res_pairs = []
    en = set()
    for p in pairs:
        if p[0] not in en:
            res_pairs.append(p)
            en.add(p[0])
    return res_pairs


def segment_word(pairs: List[Tuple]):
    def segment_en(sentence: str):
        """以空格分隔，标点符号与句子分开，单引号不分"""
        punctuation = set(',.;:\"?!')
        new_sen = ""
        for i, c in enumerate(sentence):
            if c in punctuation:
                new_sen += f"{c} " if i == 0 else f" {c}"
            else:
                new_sen += c
        return new_sen.lower().strip().split(' ')

    def segment_zh(sentence: str):
        """按字分隔"""
        return list(sentence)

    new_pairs = [(segment_en(p[0]), segment_zh(p[1])) for p in pairs]
    del pairs
    return new_pairs


def save_pickle(data, path: str):
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print("成功保存数据!")


if __name__ == '__main__':
    segmented_pairs = segment_word(en_zh_pairs())
    save_pickle(segmented_pairs, r'../dataset/en_zh.pkl')
    print(segmented_pairs[:10])