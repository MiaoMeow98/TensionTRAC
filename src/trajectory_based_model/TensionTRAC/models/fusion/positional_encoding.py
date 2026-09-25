'''
Code from:
    https://github.com/wangxiang1230/OadTR
    https://github.com/ManuBenavent/AAG
'''

import torch
import torch.nn as nn

class FixedPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_length=5000):
        super(FixedPositionalEncoding, self).__init__()

        pe = torch.zeros(max_length, embedding_dim)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / embedding_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return x


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_position_embeddings, embedding_dim, seq_length):
        super(LearnedPositionalEncoding, self).__init__()
        self.pe = nn.Embedding(max_position_embeddings, embedding_dim)
        self.max_position_embeddings = max_position_embeddings

        self.register_buffer(
            "position_ids",
            torch.arange(max_position_embeddings).expand((1, -1)),
        )

    def forward(self, x, position_ids=None):
        # Get actual sequence length from input x
        seq_length = x.size(1)  # x shape: [B, seq_len, D]
        if seq_length > self.max_position_embeddings:
            raise ValueError(
                f"Sequence length {seq_length} exceeds max positional encoding length {self.max_position_embeddings}"
            )
        
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        position_embeddings = self.pe(position_ids)
        return x + position_embeddings
