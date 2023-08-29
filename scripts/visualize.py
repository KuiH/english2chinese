from utils.common import load_data
import matplotlib.pyplot as plt
from typing import *


def token_number_dist(data: List[Tuple]):
    """句子中token的数量分布"""
    en_dist = [0] * 40
    zh_dist = [0] * 40
    en_max_len = -1
    zh_max_len = -1
    for d in data:
        en_len, zh_len = len(d[0]), len(d[1])
        en_dist[en_len] += 1
        zh_dist[zh_len] += 1
        en_max_len, zh_max_len = max(en_max_len,en_len), max(zh_max_len,zh_len)
    fig = plt.figure(figsize=(10, 8), dpi=160)
    plt.plot(range(0, en_max_len+5), en_dist[:en_max_len+5])
    plt.title("En sentence token distribution")
    plt.xlabel("sentence token number")
    plt.ylabel("sentence number")
    plt.savefig(r"../pic/en_sectence_token_dist.jpg")
    fig = plt.figure(figsize=(10, 8), dpi=160)
    plt.plot(range(0, zh_max_len + 5), zh_dist[:zh_max_len + 5])
    plt.title("Zh sentence token distribution")
    plt.xlabel("sentence token number")
    plt.ylabel("sentence number")
    plt.savefig(r"../pic/zh_sectence_token_dist.jpg")


if __name__ == '__main__':
    data = load_data(r'../dataset/en_zh.pkl')
    token_number_dist(data)
