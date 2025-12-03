import torch
import torch.nn as nn
import torch.nn.functional as F


class F3locLoss(nn.Module):
    def __init__(self, shape_loss_weight=None):
        """
        shape_loss_weight: float 或 None
        """
        super().__init__()
        self.shape_loss_weight = shape_loss_weight

    def forward(self, pred_rays, gt_rays):
        """
        pred_rays: [B, N]
        gt_rays:   [B, N]
        """
        # 基础 L1 损失
        l1 = F.l1_loss(pred_rays, gt_rays)

        # 如果不需要 shape loss，就只返回 L1
        if self.shape_loss_weight is None:
            return l1

        # shape loss = 1 - cosine_similarity
        cosine_sim = F.cosine_similarity(pred_rays, gt_rays).mean()
        shape_loss = self.shape_loss_weight * (1 - cosine_sim)

        # 总损失
        return l1 + shape_loss
