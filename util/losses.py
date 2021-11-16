import torch.nn.functional as F
import torch
from torch.nn.modules.loss import _Loss
from collections import Counter
import torch.nn as nn
import numpy as np
import math


class BaseLoss(_Loss):
    def __init__(self,):
        super(BaseLoss, self).__init__()
        self.cum_loss = 0

    def clear_loss(self):
        self.cum_loss = 0


class PretrainLoss(BaseLoss):
    def __init__(self,):
        super(PretrainLoss, self).__init__()
        self.criteria = torch.nn.MSELoss()

    def forward(self, y_hat, y):
        l = self.criteria(y_hat,y)
        self.cum_loss +=l
        return l

    def get_description(self, step):
        tok_loss = self.cum_loss
        desc = " token loss : %f" % (
            tok_loss / step)
        return desc


class ClassificationLoss(BaseLoss):
    def __init__(self,):
        super(ClassificationLoss, self).__init__()
        self.criteria = torch.nn.CrossEntropyLoss()

    def forward(self, y_hat, y):
        y_hat = y_hat[:, 0]
        l = self.criteria(y_hat,y)
        self.cum_loss +=l
        return l

    def get_description(self, step):
        tok_loss = self.cum_loss
        desc = " token loss : %f" % (
            tok_loss / step)
        return desc
