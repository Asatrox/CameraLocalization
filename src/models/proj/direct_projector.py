import torch
import torch.nn as nn
import numpy as np


class DirectBEVProjector(nn.Module):
    """
    从 RGB 特征 feat_rgb 投影/重采样成 BEV 特征。
    - 输入:  feat_rgb: [B, C, H, W]
    - 输出:  bev_feat: [B, C, H, W]   (不返回 mask)
    """

    def __init__(self, grd_height=-2.0, rotation_range=360.0, bev_h=40, bev_w=128):
        """
        参数:
            grd_height:     地面相机高度（和你原代码一样，默认 -2）
            rotation_range: rot 的角度范围（度），和原 BEV_corr 里的保持一致
        """
        super().__init__()
        self.grd_height = float(grd_height)
        self.rotation_range = float(rotation_range)
        self.bev_h = bev_h
        self.bev_w = bev_w

    def compute_uv(self, rot, shift_u, shift_v, H, W, meter_per_pixel):
        """
        生成每个 BEV 像素在地面特征图上的采样坐标 (u, v)。

        rot:             [B]   航向角参数（网络估计），一般在 [-1,1]
        shift_u,shift_v: [B]   平移参数
        H, W:            int   地面特征图大小 (feat_rgb 的 H, W)
        meter_per_pixel: 标量或 [B]，每像素多少米
        返回:
            uv: [B, H, W, 2]   每个 BEV 像素对应到地面特征图的 (u, v)
        """
        device = rot.device
        B = rot.shape[0]

        # 这里直接让 BEV 输出分辨率 = 输入特征分辨率
        S_v = self.bev_h  # BEV 高
        S_u = self.bev_w  # BEV 宽

        # 生成网格 (ii: v, jj: u)，范围 [0, S_v-1] / [0, S_u-1]
        ii, jj = torch.meshgrid(
            torch.arange(0, S_v, dtype=torch.float32, device=device),
            torch.arange(0, S_u, dtype=torch.float32, device=device),
            indexing="ij",
        )
        ii = ii.unsqueeze(0).repeat(B, 1, 1)  # [B, H, W]
        jj = jj.unsqueeze(0).repeat(B, 1, 1)  # [B, H, W]

        # 中心 + 平移 (和你原 sat2grd_uv 思路类似)
        center_v = (S_v / 2 - 0.5) + shift_v.view(-1, 1, 1)  # [B,1,1]
        center_u = (S_u / 2 - 0.5) + shift_u.view(-1, 1, 1)  # [B,1,1]

        # 像素半径
        radius = torch.sqrt((ii - center_v) ** 2 + (jj - center_u) ** 2)  # [B, H, W]

        # 平面角度 θ
        theta = torch.atan2(ii - center_v, jj - center_u)  # [-π, π]
        theta = (-np.pi / 2 + (theta % (2 * np.pi))) % (2 * np.pi)  # [0, 2π)
        theta = (
            theta
            + rot.view(-1, 1, 1) * self.rotation_range / 180.0 * np.pi
        ) % (2 * np.pi)

        # θ → 宽度方向像素 u，范围 [0, W)
        u = theta / (2 * np.pi) * W  # [B,H,W]

        # meter_per_pixel 支持标量或 [B]
        if not torch.is_tensor(meter_per_pixel):
            meter_per_pixel = torch.tensor(
                meter_per_pixel, dtype=torch.float32, device=device
            ).view(1)
        meter_per_pixel = meter_per_pixel.view(-1, 1, 1)  # [B,1,1]

        # 半径(像素)→物理距离→俯仰角 φ→高度方向像素 v
        grd_height_tensor = torch.tensor(
            self.grd_height, dtype=torch.float32, device=device
        )
        dist = radius * meter_per_pixel  # 水平距离（米）
        phi = torch.atan2(dist, grd_height_tensor)  # [B,H,W]
        v = phi / np.pi * H  # 映射到 [0, H)

        # 组装 uv（注意这里是 [u,v]，和原 grid_sample 实现一致）
        uv = torch.stack([u, v.float()], dim=-1)  # [B,H,W,2]
        return uv

    @staticmethod
    def bilinear_sample(image, uv):
        """
        手写双线性插值版 (不返回 mask)

        image: [B, C, H, W]
        uv:    [B, H, W, 2] (u,v) 像素坐标，范围可以超过边界，超出部分会被 clamp

        返回:
            out: [B, C, H, W]
        """
        B, C, IH, IW = image.shape
        _, H, W, _ = uv.shape

        ix = uv[..., 0].view(B, 1, H, W)  # u
        iy = uv[..., 1].view(B, 1, H, W)  # v

        with torch.no_grad():
            ix_nw = torch.floor(ix)
            iy_nw = torch.floor(iy)
            ix_ne = ix_nw + 1
            iy_ne = iy_nw
            ix_sw = ix_nw
            iy_sw = iy_nw + 1
            ix_se = ix_nw + 1
            iy_se = iy_nw + 1

            # 边界裁剪
            torch.clamp(ix_nw, 0, IW - 1, out=ix_nw)
            torch.clamp(iy_nw, 0, IH - 1, out=iy_nw)
            torch.clamp(ix_ne, 0, IW - 1, out=ix_ne)
            torch.clamp(iy_ne, 0, IH - 1, out=iy_ne)
            torch.clamp(ix_sw, 0, IW - 1, out=ix_sw)
            torch.clamp(iy_sw, 0, IH - 1, out=iy_sw)
            torch.clamp(ix_se, 0, IW - 1, out=ix_se)
            torch.clamp(iy_se, 0, IH - 1, out=iy_se)

        # 权重
        nw = (ix_se - ix) * (iy_se - iy)
        ne = (ix - ix_sw) * (iy_sw - iy)
        sw = (ix_ne - ix) * (iy - iy_ne)
        se = (ix - ix_nw) * (iy - iy_nw)

        image_flat = image.view(B, C, IH * IW)

        def gather(ix, iy):
            idx = (iy * IW + ix).long().view(B, 1, H * W).repeat(1, C, 1)
            return torch.gather(image_flat, 2, idx).view(B, C, H, W)

        nw_val = gather(ix_nw, iy_nw)
        ne_val = gather(ix_ne, iy_ne)
        sw_val = gather(ix_sw, iy_sw)
        se_val = gather(ix_se, iy_se)

        out = nw_val * nw + ne_val * ne + sw_val * sw + se_val * se
        return out

    def forward(
        self,
        feat_rgb,
        rot=None,
        shift_u=None,
        shift_v=None,
        meter_per_pixel=1.0,
    ):
        """
        feat_rgb: [B, C, H, W]
        rot:      [B] 或 None，默认全 0
        shift_u:  [B] 或 None，默认全 0
        shift_v:  [B] 或 None，默认全 0
        meter_per_pixel: 标量或 [B]，默认 1.0

        返回:
            bev_feat: [B, C, H, W]
        """
        x = tokens_to_feature(x, H, W)  # 转成 (B, C, H, W)
        B, C, H, W = feat_rgb.shape
        device = feat_rgb.device

        # 默认参数 = 0 / 1
        if rot is None:
            rot = torch.zeros(B, device=device, dtype=torch.float32)
        if shift_u is None:
            shift_u = torch.zeros(B, device=device, dtype=torch.float32)
        if shift_v is None:
            shift_v = torch.zeros(B, device=device, dtype=torch.float32)

        uv = self.compute_uv(rot, shift_u, shift_v, H, W, meter_per_pixel)  # [B,H,W,2]
        bev_feat = self.bilinear_sample(feat_rgb, uv)                        # [B,C,H,W]
        return bev_feat
    

def tokens_to_feature(x, H, W):
    """
    x: (B, S, P, C)
    H, W: 还原 token 的空间大小 (P = H * W)

    返回:
        x_5d: (B, S, C, H, W)
        x_bev: (B, C, H, W)   # 取 S=0
    """
    B, S, P, C = x.shape
    assert P == H * W, f"P={P} 必须等于 H*W={H*W}"

    # (B, S, P, C) → (B, S, H, W, C)
    x = x.view(B, S, H, W, C)

    # (B, S, H, W, C) → (B, S, C, H, W)
    x_5d = x.permute(0, 1, 4, 2, 3).contiguous()

    # 取 S=0 的那一帧
    x_rgb = x_5d[:, 0]   # (B, C, H, W)

    return x_rgb
