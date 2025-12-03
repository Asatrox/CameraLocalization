# bev_from_rgb_cross_attn.py
# -*- coding: utf-8 -*-
"""
用 cross-attention 将 RGB 图像特征投影到 BEV 特征
结构参考 BEVFormer / PETR 的 BEV query 跨模态注意力思路

输入:
    imgs: [B, N_cam, 3, H, W]

输出:
    bev_feat: [B, C, H_bev, W_bev]
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from flash_attn.flash_attn_interface import flash_attn_func
    HAS_FLASH_ATTN = True
except Exception:
    HAS_FLASH_ATTN = False

# 2D位置编码
class PositionalEncoding2D(nn.Module):
    """
    标准 2D sin/cos 位置编码，通道维度 C 分成两半分别给 x / y。
    """

    def __init__(self, d_model: int, height: int, width: int):
        super().__init__()
        if d_model % 4 != 0:
            raise ValueError("d_model 必须能被 4 整除，用于 2D sin/cos encoding")
        
        self.d_model = d_model
        self.height = height
        self.width = width

        pe = torch.zeros(d_model, height, width)  # [C, H, W]
        d_model_half = d_model // 2
        d_model_quarter = d_model // 4

        # ========== Y 方向 ==========
        div_term_y = torch.exp(
            torch.arange(0, d_model_quarter, 2, dtype=torch.float32)
            * -(math.log(10000.0) / d_model_quarter)
        )  # [d_model_quarter/2]

        pos_y = torch.arange(height, dtype=torch.float32).unsqueeze(1)  # [H,1]
        pe_y = pos_y * div_term_y  # [H, d_model_quarter/2]

        # broadcast 到 W 维度
        pe_y = pe_y.unsqueeze(2).repeat(1, 1, width)  # [H, d_model_quarter/2, W]

        pe[0:d_model_quarter:2, :, :] = torch.sin(pe_y).permute(1,0,2)
        pe[1:d_model_quarter:2, :, :] = torch.cos(pe_y).permute(1,0,2)

        # ========== X 方向 ==========
        div_term_x = torch.exp(
            torch.arange(0, d_model_quarter, 2, dtype=torch.float32)
            * -(math.log(10000.0) / d_model_quarter)
        )

        pos_x = torch.arange(width, dtype=torch.float32).unsqueeze(1)  # [W,1]
        pe_x = pos_x * div_term_x  # [W, d_model_quarter/2]

        # broadcast 到 H 维度
        pe_x = pe_x.unsqueeze(2).repeat(1, 1, height)  # [W, d_model_quarter/2, H]
        pe_x = pe_x.permute(1,2,0)  # -> [d_model_quarter/2, H, W]

        pe[d_model_quarter:d_model_half:2, :, :] = torch.sin(pe_x)
        pe[d_model_quarter+1:d_model_half:2, :, :] = torch.cos(pe_x)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]，H/W 必须和初始化时一致
        """
        B, C, H, W = x.shape
        if H != self.height or W != self.width:
            raise ValueError(f"输入大小 {H}x{W} 和位置编码大小 {self.height}x{self.width} 不一致")
        return x + self.pe  # 广播到 batch


# Cross-Attention Block (BEV token ↔ image token)
class CrossAttentionBlock(nn.Module):
    """
    单层 BEV cross-attention:
    - Query: BEV tokens [B, L_bev, C]
    - Key/Value: image tokens [B, L_img, C]
    结构: Cross-Attn + FFN + 残差 + LayerNorm

    如果 use_flash_attn=True 且环境满足要求，就用 flash-attn 加速；
    否则回退到 nn.MultiheadAttention。
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        use_flash_attn: bool = False,
    ):
        super().__init__()

        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.use_flash_attn = use_flash_attn

        # 备用：不满足 flash-attn 条件时使用
        self.cross_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True,  # [B, L, C]
        )

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.attn_dropout_p = dropout  # 给 flash-attn 用

    def forward(
        self,
        bev_tokens: torch.Tensor,
        img_tokens: torch.Tensor,
        img_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        bev_tokens: [B, L_bev, C]
        img_tokens: [B, L_img, C]
        img_key_padding_mask: [B, L_img]，True 表示要 mask 的位置（可选）
                             目前 flash-attn 分支里暂时没用 mask，如有需要可以再加 bias。
        """
        B, L_bev, C = bev_tokens.shape
        _, L_img, C_img = img_tokens.shape
        assert C == C_img == self.d_model

        # ------------------ 1) Cross-Attention ------------------
        if (
            self.use_flash_attn
            and HAS_FLASH_ATTN
            and bev_tokens.is_cuda
            and bev_tokens.dtype in (torch.float16, torch.bfloat16)
        ):
            # flash-attn: q、k、v 需要 [B, L, H, D]
            # 我们现在是 [B, L, C]，先拆成多头
            # Q: BEV tokens，K/V: image tokens
            q = bev_tokens.view(B, L_bev, self.nhead, self.head_dim)   # [B, L_bev, H, D]
            k = img_tokens.view(B, L_img, self.nhead, self.head_dim)   # [B, L_img, H, D]
            v = img_tokens.view(B, L_img, self.nhead, self.head_dim)

            # flash_attn_func 支持 Q/K/V 长度不同，即 cross-attention
            # 输出 [B, L_bev, H, D]
            attn_out = flash_attn_func(
                q,
                k,
                v,
                dropout_p=self.attn_dropout_p if self.training else 0.0,
                softmax_scale=None,   # 默认 1/sqrt(D)
                causal=False,
            )
            # 合并头 -> [B, L_bev, C]
            attn_out = attn_out.reshape(B, L_bev, C)
        else:
            # 回退到标准 MultiheadAttention
            attn_out, _ = self.cross_attn(
                query=bev_tokens,
                key=img_tokens,
                value=img_tokens,
                key_padding_mask=img_key_padding_mask,
                need_weights=False,
            )

        bev_tokens = self.norm1(bev_tokens + attn_out)

        # ------------------ 2) FFN ------------------
        ffn_out = self.ffn(bev_tokens)
        bev_tokens = self.norm2(bev_tokens + ffn_out)

        return bev_tokens



# BEV Transformer 编码器 (多层 Cross-Attn)
class BEVTransformerEncoder(nn.Module):
    def __init__(
        self,
        bev_h: int,
        bev_w: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        use_flash_attn: bool = True,
    ):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.d_model = d_model

        # BEV query：可学习，表示 BEV 网格上的“槽”
        self.bev_queries = nn.Parameter(
            torch.zeros(1, bev_h * bev_w, d_model)
        )  # [1, L_bev, C]
        nn.init.xavier_uniform_(self.bev_queries)

        # BEV 的 2D 位置编码
        self.bev_pos_encoder = PositionalEncoding2D(d_model, bev_h, bev_w)

        # 多层 cross-attn
        self.layers = nn.ModuleList(
            [
                CrossAttentionBlock(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    use_flash_attn=use_flash_attn,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        img_tokens: torch.Tensor,
        img_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        img_tokens: [B, L_img, C]
        img_key_padding_mask: [B, L_img] (可选)
        return:
            bev_feat: [B, C, H_bev, W_bev]
        """
        img_tokens = img_tokens[:, 0]
        B, _, C = img_tokens.shape
        device = img_tokens.device

        # 构造 BEV query + 2D BEV 位置编码
        # dummy 用来生成 pos enc
        dummy = torch.zeros(B, C, self.bev_h, self.bev_w, device=device)
        bev_pos = self.bev_pos_encoder(dummy)  # [B, C, H_bev, W_bev]
        bev_pos = bev_pos.flatten(2).transpose(1, 2)  # [B, L_bev, C]

        bev_tokens = self.bev_queries.expand(B, -1, -1) + bev_pos  # [B, L_bev, C]

        # 依次通过多层 cross-attention
        for layer in self.layers:
            bev_tokens = layer(
                bev_tokens=bev_tokens,
                img_tokens=img_tokens,
                img_key_padding_mask=img_key_padding_mask,
            )

        # reshape 回 BEV feature map
        bev_feat = bev_tokens.transpose(1, 2).reshape(
            B, C, self.bev_h, self.bev_w
        )  # [B, C, H_bev, W_bev]

        return bev_feat


class CABEVProjector(nn.Module):
    def __init__(
        self,
        bev_h: int = 40,
        bev_w: int = 128,
        d_model: int = 2048,
        nhead: int = 8,
        num_layers: int = 2,
    ):
        super().__init__()


        # image 特征上也加一个 2D 位置编码（会在 forward 里根据特征 map 尺寸构建）
        self.bev_encoder = BEVTransformerEncoder(
            bev_h=bev_h,
            bev_w=bev_w,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=4 * d_model,
            dropout=0.1,
            use_flash_attn=True,
        )

        # 这个 image pos encoding 会在第一次 forward 时懒加载
        self.img_pos_encoder: Optional[PositionalEncoding2D] = None

    def forward(
        self,
        img_tokens,
        img_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        img_tokens: [B, L_img, C]
        img_masks: [B, N_cam, H_feat, W_feat] 或 [B, N_cam, 1, H_feat, W_feat] (可选)
                   True 表示要 mask 的位置 (比如无效区域 / padding)

        输出:
            bev_feat: [B, C, H_bev, W_bev]
        """

        # 构造 key_padding_mask（如果你有 img_masks 的话）
        img_key_padding_mask = None
        if img_masks is not None:
            # 假设 img_masks 是 [B, N_cam, Hf, Wf]，True=需要mask
            if img_masks.dim() == 5:
                img_masks = img_masks.squeeze(2)  # 去掉通道维
            # [B, N_cam, Hf, Wf] -> [B, L_img]
            img_key_padding_mask = img_masks.view(B, N_cam * Hf * Wf)

        # ------- 3) BEV Transformer 编码器 -------
        bev_feat = self.bev_encoder(
            img_tokens=img_tokens,
            img_key_padding_mask=img_key_padding_mask,
        )  # [B, C, H_bev, W_bev]

        return bev_feat

