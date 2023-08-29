import torch
import torch.nn as nn


def tensor_mask(x, valid_lens, value=0):
    """
    给x进行mask, 有效长度由valid_len指定。
    x = [[1,2,3],[4,5,6]], valid_len = [1,2] ---> [[1,0,0],[4,5,0]]
    """
    assert x.shape[0] == len(valid_lens), "valid_len must have the same length as the first dimension of x"
    maxlen = x.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32, device=x.device)[None, :] < valid_lens[:, None]
    x[~mask] = value
    return x


class MaskedCrossEntropyLoss(nn.CrossEntropyLoss):
    """带遮蔽的softmax交叉熵损失函数,让pad的部分不计算损失"""

    def forward(self, pred, label, valid_lens):
        weights = torch.ones_like(label)
        weights = tensor_mask(weights, valid_lens)
        self.reduction = 'none'
        # torch.nn.CrossEntropyLoss()在第二个维度求损失，因此这里pred换一下维度
        unweighted_loss = super().forward(pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)  # mean: 对每个样本求平均，也就是算一个句子的损失
        return weighted_loss


if __name__ == '__main__':
    loss = MaskedCrossEntropyLoss()
    res = loss(torch.ones(3, 4, 10), torch.ones((3, 4), dtype=torch.long),
               torch.tensor([4, 2, 0]))
    print(res)
