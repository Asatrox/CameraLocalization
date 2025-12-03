import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPPredictor(nn.Module):
    """
    输入:  bev_feat [B, 1024, 40, 128]
    输出:  depth_pred [B, 40]   （40×1 的深度）
    """

    def __init__(self, in_channels=2048, hidden_channels=256, mlp_hidden_dim=128):
        super().__init__()
        # 1. 先做一些卷积，保留空间结构，同时降通道维
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels // 2, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )

        # 2. 只沿着 W 方向池化，保留 H=40 这一维
        # 输入:  [B, hidden_channels, 40, 128]
        # 输出:  [B, hidden_channels, 40, 1]
        self.pool_along_width = nn.AdaptiveAvgPool2d((40, 1))

        # 3. 对每一行（40 行）用 MLP 回归一个深度
        # 先把 [B, hidden_channels, 40, 1] → [B, 40, hidden_channels]
        # 然后 Linear(hidden_channels → 1)，最后 squeeze 成 [B, 40]
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, mlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_dim, 1)
        )

    def forward(self, bev_feat):
        """
        bev_feat: [B, 1024, 40, 128]
        return:   depth_pred: [B, 40]
        """
        B, C, H, W = bev_feat.shape
        assert C == 2048 and H == 40

        x = self.conv(bev_feat)                    
        x = self.pool_along_width(x)              
        x = x.squeeze(-1)                          
        x = x.permute(0, 2, 1)                  

        B, H, C_feat = x.shape                    
        x_flat = x.reshape(B * H, C_feat)         
        depth_flat = self.mlp(x_flat)            
        depth = depth_flat.view(B, H)              

        return depth
