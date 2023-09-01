from typing import *
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

smoothing = SmoothingFunction()


def bleu(references: List[str], candidates: List[str]):
    """
    传进来的是str，是表示一整个以空格分隔的句子，而不是一个token.
    所以先拆成一个个token再交给nltk计算
    """
    references = [ref.split(' ') for ref in references]
    candidates = [cand.split(' ') for cand in candidates]
    scores = [sentence_bleu([ref], cand, smoothing_function=smoothing.method1) for ref, cand in
              zip(references, candidates)]
    return scores


if __name__ == '__main__':
    references = ["测 试 。", "是 湯 姆 救 了 這 個 小 女 孩 。"]
    candidates = ["这 是 测 试", "湯 姆 救 這 個 小 女 孩 。"]
    scores = bleu(references=references, candidates=candidates)
    print(scores)
    print(sum(scores))
