import torch.nn.functional as F
import torch
from torch.nn.modules.loss import _Loss
from collections import Counter
import torch.nn as nn
import numpy as np
from util.loss_utils import *
import math


class BaseLoss(_Loss):
    def __init__(self,):
        super(BaseLoss, self).__init__()
        self.cum_loss = 0
        self.cum_acc = 0

    def clear_loss(self):
        self.cum_loss = 0
        self.cum_acc = 0

    @staticmethod
    def get_acc(logits,y):
        _, predicted = torch.max(logits.data, 1)
        correct = (predicted == y).sum().item()
        return correct / y.numel()


class PlainLoss(BaseLoss):
    def __init__(self, padding_idx:int):
        super(PlainLoss, self).__init__()
        self.padding_idx = padding_idx
        self.criteria = torch.nn.CrossEntropyLoss(ignore_index=padding_idx)

    def forward(self, y_hat, y):
        y = y.contiguous().view(-1)
        l = self.criteria(y_hat,y)
        self.cum_loss +=l
        return l

    def get_description(self, step):
        tok_loss = self.cum_loss
        desc = " token loss : %f, token ppl : %f, acc : %f" % (
            tok_loss / step, math.exp(tok_loss / step), self.cum_acc / step)
        return desc





