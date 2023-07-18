import torch
import torch.nn as nn
import torch.nn.functional as F


"""
This is an implementation of
"Content and Colour Distillation for Learning Image Translations with the Spatial Profile Loss"
(https://arxiv.org/abs/1908.00274).
It is taken from here:
https://github.com/ssarfraz/SPL/blob/master/SPL_Loss/pytorch_spl_loss.py
"""


class GPLoss(nn.Module):

    def __init__(self):
        super(GPLoss, self).__init__()
        self.loss = SPLoss()

    def get_gradients(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, c, h, w].
        Returns:
            v: a float tensor with shape [b, c, h - 1, w].
            h: a float tensor with shape [b, c, h, w - 1].
        """
        v = x[:, :, 1:, :] - x[:, :, :-1, :]
        h = x[:, :, :, 1:] - x[:, :, :, :-1]
        return v, h

    def forward(self, x, y):
        """
        Arguments:
            x, y: float tensors with shape [b, c, h, w].
        Returns:
            a float tensor with shape [].
        """
        x_v, x_h = self.get_gradients(x)
        y_v, y_h = self.get_gradients(y)

        return self.loss(x_v, y_v) + self.loss(x_h, y_h)


class CPLoss(nn.Module):

    def __init__(self):
        super(CPLoss, self).__init__()

        self.loss = SPLoss()
        self.gp_loss = GPLoss()

    def to_YUV(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, 3, h, w],
            it represents a batch of RGB images
            with pixel values in the range [0, 1].
        Returns:
            a float tensor with shape [b, 3, h, w].
        """
        R, G, B = torch.split(x, 1, dim=1)

        Y = 0.299 * R + 0.587 * G + 0.114 * B
        U = 0.492 * (B - Y)
        V = 0.877 * (R - Y)

        return torch.cat([Y, U, V], dim=1)

    def forward(self, x, y):
        """
        It returns a value in [-12, 12] range.
        Arguments:
            x, y: float tensors with shape [b, 3, h, w].
        Returns:
            a float tensor with shape [].
        """
        loss = self.loss(x, y)

        x_yuv = self.to_YUV(x)
        y_yuv = self.to_YUV(y)
        loss += self.loss(x_yuv, y_yuv)

        loss += self.gp_loss(x_yuv, y_yuv)
        return loss


class SPLoss(nn.Module):

    def __init__(self):
        super(SPLoss, self).__init__()

    def forward(self, x, y):
        """
        It returns a value in [-c, c] range.
        Arguments:
            x, y: float tensors with shape [b, c, h, w].
        Returns:
            a float tensor with shape [].
        """
        b, _, h, w = x.size()

        cols = F.normalize(x, p=2, dim=2) * F.normalize(y, p=2, dim=2)
        rows = F.normalize(x, p=2, dim=3) * F.normalize(y, p=2, dim=3)
        # they have shape [b, c, h, w]

        cols = cols.sum(2)  # shape [b, c, w]
        rows = rows.sum(3)  # shape [b, c, h]

        # `cols` represent cosine similarities between columns,
        # `rows` represent cosine similarities between rows

        return (-1.0/b) * ((1.0/w) * cols.sum() + (1.0/h) * rows.sum())