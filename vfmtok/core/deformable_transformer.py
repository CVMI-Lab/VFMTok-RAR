# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import torch, pdb
import copy, math
from einops import rearrange
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional, List
from ..modules.attention import Attention3
from ..modules.ops.modules import MSDeformAttn

class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.n_heads = n_heads
        self.rope_base = 10000
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = Attention3(d_model, n_heads, dropout=dropout)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, kv_feature, input_spatial_shape, level_start_index):
        
        bs, n_c, dim = src.shape
        with torch.autocast('cuda', dtype=torch.float32, enabled=True):
            src2 = self.cross_attn(self.with_pos_embed(src, pos), reference_points, kv_feature, input_spatial_shape, level_start_index)

        src = src + self.dropout1(src2)
        src = self.norm1(src)

        q = k = self.with_pos_embed(src, pos)

        mask = ~torch.tril(src.new_ones((n_c, n_c), dtype=torch.bool), diagonal=0)
        with torch.autocast('cuda', dtype=torch.float32, enabled=True):
            tgt2 = self.self_attn(self.with_pos_embed(src, pos), attn_mask=mask)
        src = src + self.dropout4(tgt2)
        src = self.norm3(src)

        # ffn
        src = self.forward_ffn(src)

        return src

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
