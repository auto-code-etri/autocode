import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def attention(query, key, value, mask=None, dropout_fn=None):
    """Scaled dot product attention

    Args:
        query (Tensor): Query matrix -> [batch_size, num_head, hidden, d_k]
        key (Tensor): Key matrix -> [batch_size, num_head, hidden, d_k]
        value (Tensor): Value matrix -> [batch_size, num_head, hidden, d_k]
        mask (Tensor): Mask tensor which blocks calculating loss
        dropout_fn (nn.Module): Dropout function

    Returns:
        Calculated value matrix and attention distribution
    """
    d_k = query.shape[-1]
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores.masked_fill_(mask, -np.inf)
    p_attn = F.softmax(scores, dim=-1)

    if dropout_fn is not None:
        p_attn = dropout_fn(p_attn)
    context = torch.matmul(p_attn, value)
    return context, p_attn


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention module

    Args:
        num_head (int): Number of self-attention head
        hidden (int): Hidden dimension of model
        dropout (float): Dropout ratio
    """

    def __init__(self, num_head, hidden, dropout):
        super(MultiHeadAttention, self).__init__()
        assert hidden % num_head == 0
        self.d_k = hidden // num_head
        self.num_head = num_head
        self.dropout = nn.Dropout(dropout)

        self.W_q = nn.Linear(hidden, self.d_k * num_head)
        self.W_k = nn.Linear(hidden, self.d_k * num_head)
        self.W_v = nn.Linear(hidden, self.d_k * num_head)
        self.attn = None
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, hidden)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        b_size = query.shape[0]

        # make q, k, v matrice with multi-head
        # q,k,v: [batch_size, num_head, hidden, d_k]
        q = self.W_q(query).view(b_size, -1, self.num_head, self.d_k).transpose(1, 2)
        k = self.W_k(key).view(b_size, -1, self.num_head, self.d_k).transpose(1, 2)
        v = self.W_v(value).view(b_size, -1, self.num_head, self.d_k).transpose(1, 2)

        # calculate scaled dot product attention
        x, self.attn = attention(q, k, v, mask=mask, dropout_fn=self.dropout)

        # concatenate
        x = x.transpose(1, 2).contiguous().view(b_size, -1, self.num_head * self.d_k)
        return self.fc(x)


class ResidualConnection(nn.Module):
    """Residual connection with layer normalization

    Args:
        f_size (int): Feature size to normalize
        dropout (float): Dropout ratio
    """

    def __init__(self, f_size, dropout):
        super(ResidualConnection, self).__init__()
        self.layernorm = nn.LayerNorm(f_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.layernorm(x)))


class PositionWiseFeedForward(nn.Module):
    """Position-wise feedforward layer

    Args:
        hidden (int): Hidden dimension of model
        fc_hidden (int): Hidden dimension of position-wise feedforward layer
        dropout (float): Dropout ratio
    """

    def __init__(self, hidden, fc_hidden, dropout):
        super(PositionWiseFeedForward, self).__init__()
        self.fc_1 = nn.Linear(hidden, fc_hidden)
        self.fc_2 = nn.Linear(fc_hidden, hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc_2(self.dropout(F.relu(self.fc_1(x))))


class PositionalEncoding(nn.Module):
    """
    Args:
        hidden (int): Hidden dimension of model
        dropout (float): Dropout ratio
    """

    def __init__(self, hidden, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, hidden)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden, 2) * -(math.log(10000.0) / hidden))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.shape[1]].detach()
        return self.dropout(x)
