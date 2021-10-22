# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
from paddle import nn
import paddle.nn.functional as F
# from paddle.nn.modules.loss import _Loss
from torch_to_paddle_api.api import gather,scatter_add_,masked_fill_


class LabelSmoothingLoss(nn.Layer):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """

    def __init__(self, label_smoothing=0, tgt_vocab_size=0, ignore_index=0, size_average=None, reduce=None, reduction='mean'):
        assert 0.0 < label_smoothing <= 1.0
        self.ignore_index = ignore_index
        super(LabelSmoothingLoss, self).__init__()

        assert label_smoothing > 0
        assert tgt_vocab_size > 0

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        self.one_hot = paddle.full((tgt_vocab_size, 1), smoothing_value)
        self.one_hot[self.ignore_index] = 0
        self.linear = paddle.nn.Linear(10, 3)
        self.linear.register_buffer('one_hot', self.one_hot.unsqueeze(0))
        self.confidence = 1.0 - label_smoothing
        self.tgt_vocab_size = tgt_vocab_size

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size * num_pos * n_classes
        target (LongTensor): batch_size * num_pos
        """
        assert self.tgt_vocab_size == output.shape[2]
        batch_size, num_pos = target.shape[0], target.shape[1]
        output = output.reshape([-1, self.tgt_vocab_size])
        target = target.reshape([-1])
        model_prob = paddle.fluid.layers.expand(self.one_hot, (target.shape[0], 1))
        model_prob = gather(paddle.to_tensor([self.confidence]), 1, target.unsqueeze(1))
        scatter_add_(model_prob, 1, target.unsqueeze(1), paddle.to_tensor([self.confidence]))
        masked_fill_(model_prob, (target == self.ignore_index).unsqueeze(1), 0)

        return F.kl_div(output, model_prob, reduction='none').shape([batch_size, num_pos, -1]).sum(2)
