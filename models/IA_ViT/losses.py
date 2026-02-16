"""
Copyright 2023 Yao Qiang, Chengyin Li, Hui Zhu,
Prashant Khanduri, Dongxiao Zhu

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

Modifications:
- Modified by Julian Fernando Weber, 2026: modified class for regression task, added MMDLossRegression class, and updated compute_hinton_loss to use SmoothL1Loss for regression distillation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MMDLoss(nn.Module):
    def __init__(self, sigma, kernel="rbf"):
        super(MMDLoss, self).__init__()
        self.sigma = sigma
        self.kernel = kernel

    def forward(self, old_atts, new_atts,):

        mmd_loss = self.pdist(old_atts, new_atts)[0].mean()

        return mmd_loss
    
    @staticmethod
    def pdist(e1, e2, eps=1e-12, kernel='rbf', sigma_base=1.0, sigma_avg=None):
        if len(e1) == 0 or len(e2) == 0:
            res = torch.zeros(1)
        else:
            if kernel == 'rbf':
                e1_square = e1.pow(2).sum(dim=1)
                e2_square = e2.pow(2).sum(dim=1)
                prod = e1 @ e2.t()
                res = (e1_square.unsqueeze(1) + e2_square.unsqueeze(0) - 2 * prod).clamp(min=eps)
                res = res.clone()

                sigma_avg = res.mean().detach() if sigma_avg is None else sigma_avg
                res = torch.exp(-res / (2*(sigma_base**2)*sigma_avg))
            elif kernel == 'poly':
                res = torch.matmul(e1, e2.t()).pow(2)
                
        return res, sigma_avg

class MMDLossRegression(nn.Module):
    """
    Biased MMD^2 with RBF kernel.

    old_atts: [N, D]
    new_atts: [M, D]
    returns: scalar
    """
    def __init__(self, sigma: float | None = None, sigma_base: float = 1.0, eps: float = 1e-12):
        super().__init__()
        self.sigma = sigma
        self.sigma_base = sigma_base
        self.eps = eps

    def _sq_dist(self, e1: torch.Tensor, e2: torch.Tensor) -> torch.Tensor:
        # e1: [N,D], e2: [M,D] -> dist2: [N,M]
        e1_sq = (e1 ** 2).sum(dim=1, keepdim=True)          # [N,1]
        e2_sq = (e2 ** 2).sum(dim=1, keepdim=True)          # [M,1]
        dist2 = (e1_sq + e2_sq.t() - 2.0 * (e1 @ e2.t())).clamp(min=self.eps)
        return dist2

    def _rbf(self, dist2: torch.Tensor, sigma2: torch.Tensor) -> torch.Tensor:
        return torch.exp(-dist2 / (2.0 * sigma2))

    def forward(self, old_atts: torch.Tensor, new_atts: torch.Tensor) -> torch.Tensor:
        if old_atts.numel() == 0 or new_atts.numel() == 0:
            return torch.zeros((), device=old_atts.device, dtype=old_atts.dtype)

        dxx = self._sq_dist(old_atts, old_atts)  # [N,N]
        dyy = self._sq_dist(new_atts, new_atts)  # [M,M]
        dxy = self._sq_dist(old_atts, new_atts)  # [N,M]


        if self.sigma is not None:
            sigma2 = torch.tensor((self.sigma_base * self.sigma) ** 2,
                                  device=old_atts.device, dtype=old_atts.dtype)
        else:
            sigma_avg = dxy.mean().detach()
            sigma2 = (self.sigma_base ** 2) * sigma_avg

        kxx = self._rbf(dxx, sigma2)
        kyy = self._rbf(dyy, sigma2)
        kxy = self._rbf(dxy, sigma2)

        mmd2 = kxx.mean() + kyy.mean() - 2.0 * kxy.mean()
        return mmd2

    

def mse(feature1, feature2):
    return (feature1 - feature2).pow(2).mean()

def at(x):
    return F.normalize(x.pow(2).view(x.size(0), -1))

def at_loss(x, y):
    return (at(x) - at(y)).pow(2).mean()

def compute_at_loss(feature1, feature2):
    attention_loss = (1 / 2) * (at_loss(feature1, feature2))
    return attention_loss

def compute_hinton_loss(outputs, t_outputs,  kd_temp=3):
    #destillation loss but only for classification

    # soft_label = F.softmax(t_outputs / kd_temp, dim=1)
    # kd_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs / kd_temp, dim=1),
    #                                               soft_label) * (kd_temp * kd_temp)
    # return kd_loss
    
    #student = outputs, teacher = t_outputs
    loss = nn.SmoothL1Loss()(outputs, t_outputs) #use normalization if no normalization is applied in data
    return loss