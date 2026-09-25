'''
Code from:
    https://github.com/wangxiang1230/OadTR
    https://github.com/ManuBenavent/AAG
'''

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from TensionTRAC.models.fusion.positional_encoding import FixedPositionalEncoding, LearnedPositionalEncoding
import copy


class SelfAttention(nn.Module):
    def __init__(
        self, 
        dim, 
        heads=8, 
        qkv_bias=False, 
        qk_scale=None, 
        dropout_rate=0.0,
        use_ffn=False,
        ffn_hidden_dim=512,
        ffn_dropout=0.1,
        use_pos_encoding=False,
        pos_encoding_type='fixed',  # 'fixed' or 'learned'
        max_seq_length=1024
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = heads
        self.use_ffn = use_ffn
        self.use_pos_encoding = use_pos_encoding
        
        head_dim = dim // heads
        self.scale = qk_scale or head_dim**-0.5

        # Attention components
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(dim)
        
        # FFN components (optional)
        if use_ffn:
            self.norm2 = nn.LayerNorm(dim)
            self.ffn = nn.Sequential(
                nn.Linear(dim, ffn_hidden_dim),
                nn.ReLU(),
                nn.Dropout(ffn_dropout),
                nn.Linear(ffn_hidden_dim, dim),
                nn.Dropout(ffn_dropout)
            )
        else:
            self.norm2 = None
            self.ffn = None
        
        # Positional encoding (optional)
        if use_pos_encoding:
            self.pos_encoding_type = pos_encoding_type
            if pos_encoding_type == 'fixed':
                self.pos_encoding = FixedPositionalEncoding(
                    embedding_dim=dim, 
                    max_length=max_seq_length
                )
            else:
                self.pos_encoding = LearnedPositionalEncoding(
                    max_position_embeddings=max_seq_length,
                    embedding_dim=dim,
                    seq_length=max_seq_length
                )
        else:
            self.pos_encoding_type = None
            self.pos_encoding = None

    def forward(self, x):
        B, N, C = x.shape
        
        # Apply positional encoding before attention
        if self.use_pos_encoding and self.pos_encoding is not None:
            x = self.pos_encoding(x)
        
        # Self-attention with pre-normalization
        x_norm = self.norm1(x)
        
        qkv = (
            self.qkv(x_norm)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        attn_out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        attn_out = self.proj(attn_out)
        attn_out = self.proj_drop(attn_out)
        
        # Residual connection
        x = x + attn_out
        
        # FFN with pre-normalization (optional)
        if self.use_ffn and self.ffn is not None and self.norm2 is not None:
            ffn_out = self.ffn(self.norm2(x))
            x = x + ffn_out
        
        return x


class AxialAttention(nn.Module):
    """
        Axial attention is a way to make self-attention cheaper for data with multiple dimensions, especially images, videos, or grids.
        Breaks down the full attention into separate attention steps along axis like:
            attend across rows, columns, or time.
    """
    def __init__(
        self,
        in_planes,
        out_planes,
        groups=8,
        kernel_size=56,
        stride=1,
        bias=False,
        width=False,
    ):
        assert (in_planes % groups == 0) and (out_planes % groups == 0)
        super(AxialAttention, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups
        self.group_planes = out_planes // groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.width = width

        # Multi-head self attention
        self.qkv_transform = nn.Conv1d(
            in_planes,
            out_planes * 2,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn_qkv = nn.BatchNorm1d(out_planes * 2)
        self.bn_similarity = nn.BatchNorm2d(groups * 3)
        self.bn_output = nn.BatchNorm1d(out_planes * 2)

        # Position embedding
        self.relative = nn.Parameter(
            torch.randn(self.group_planes * 2, kernel_size * 2 - 1),
            requires_grad=True,
        )
        query_index = torch.arange(kernel_size).unsqueeze(0)
        key_index = torch.arange(kernel_size).unsqueeze(1)
        relative_index = key_index - query_index + kernel_size - 1
        self.register_buffer("flatten_index", relative_index.view(-1))
        if stride > 1:
            self.pooling = nn.AvgPool2d(stride, stride=stride)

        self.reset_parameters()

    def forward(self, x):
        if self.width:
            x = x.permute(0, 2, 1, 3)
        else:
            x = x.permute(0, 3, 1, 2)  # N, W, C, H
        N, W, C, H = x.shape
        x = x.contiguous().view(N * W, C, H)

        # Transformations
        qkv = self.bn_qkv(self.qkv_transform(x))
        q, k, v = torch.split(
            qkv.reshape(N * W, self.groups, self.group_planes * 2, H),
            [
                self.group_planes // 2,
                self.group_planes // 2,
                self.group_planes,
            ],
            dim=2,
        )

        # Calculate position embedding
        all_embeddings = torch.index_select(
            self.relative, 1, self.flatten_index
        ).view(self.group_planes * 2, self.kernel_size, self.kernel_size)
        q_embedding, k_embedding, v_embedding = torch.split(
            all_embeddings,
            [
                self.group_planes // 2,
                self.group_planes // 2,
                self.group_planes,
            ],
            dim=0,
        )
        qr = torch.einsum("bgci,cij->bgij", q, q_embedding)
        kr = torch.einsum("bgci,cij->bgij", k, k_embedding).transpose(2, 3)
        qk = torch.einsum("bgci, bgcj->bgij", q, k)
        stacked_similarity = torch.cat([qk, qr, kr], dim=1)
        stacked_similarity = (
            self.bn_similarity(stacked_similarity)
            .view(N * W, 3, self.groups, H, H)
            .sum(dim=1)
        )

        # (N, groups, H, H, W)
        similarity = F.softmax(stacked_similarity, dim=3)
        sv = torch.einsum("bgij,bgcj->bgci", similarity, v)
        sve = torch.einsum("bgij,cij->bgci", similarity, v_embedding)
        stacked_output = torch.cat([sv, sve], dim=-1).view(
            N * W, self.out_planes * 2, H
        )
        output = (
            self.bn_output(stacked_output)
            .view(N, W, self.out_planes, 2, H)
            .sum(dim=-2)
        )

        if self.width:
            output = output.permute(0, 2, 1, 3)
        else:
            output = output.permute(0, 2, 3, 1)

        if self.stride > 1:
            output = self.pooling(output)

        return output

    def reset_parameters(self):
        self.qkv_transform.weight.data.normal_(
            0, math.sqrt(1.0 / self.in_planes)
        )
        nn.init.normal_(self.relative, 0.0, math.sqrt(1.0 / self.group_planes))

class CrossAttentionEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        use_ffn=True,
    ):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.use_ffn = use_ffn

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        if self.use_ffn:
            self.linear1 = nn.Linear(d_model, dim_feedforward)
            self.dropout2 = nn.Dropout(dropout)
            self.linear2 = nn.Linear(dim_feedforward, d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.dropout3 = nn.Dropout(dropout)

            if activation == "relu":
                self.activation = nn.ReLU()
            elif activation == "gelu":
                self.activation = nn.GELU()
            else:
                raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, src, memory, attn_mask=None, memory_key_padding_mask=None):
        """
        src:    [B, N_src, C]     query tokens, e.g. RGB
        memory: [B, N_mem, C]     key/value tokens
        """

        src2, attn_weights = self.cross_attention(
            query=src,
            key=memory,
            value=memory,
            attn_mask=attn_mask,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )

        src = src + self.dropout1(src2)
        src = self.norm1(src)

        if not self.use_ffn:
            return src

        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))

        src = src + self.dropout3(src2)
        src = self.norm2(src)

        return src


class CrossAttentionTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()

        self.layers = nn.ModuleList([
            copy.deepcopy(encoder_layer)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(encoder_layer.d_model)

    def forward(self, src, memory, attn_mask=None, memory_key_padding_mask=None):
        output = src

        for layer in self.layers:
            output = layer(
                output,
                memory,
                attn_mask=attn_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )

        return self.norm(output)
