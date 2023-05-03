import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .net_utils import (
    MultiHeadAttention,
    PositionalEncoding,
    PositionWiseFeedForward,
    ResidualConnection,
)


def clones(module, N: int):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Transformer(nn.Module):
    """Transformer architecture

    Args:
        vocab_size (dict): Vocabulary length
        num_enc_block (int): Number of encoder block
        num_dec_block (int): Number of decoder block
        num_head (int): Number of self-attention head
        hidden (int): Hidden dimension of model
        fc_hidden (int): Hidden dimension of position-wise feedforward layer
        dropout (float): Dropout ratio
    """

    def __init__(
        self,
        vocab_size,
        num_enc_block,
        num_dec_block,
        num_head,
        hidden,
        fc_hidden,
        dropout,
    ):
        super(Transformer, self).__init__()
        # token embedding + positional encoding
        self.src_embed = nn.Sequential(
            nn.Embedding(vocab_size, hidden), PositionalEncoding(hidden, dropout)
        )
        self.tgt_embed = nn.Sequential(
            nn.Embedding(vocab_size, hidden), PositionalEncoding(hidden, dropout)
        )

        # encoder and decoder
        self.encoder = Encoder(
            layer=EncoderLayer(
                num_head=num_head, hidden=hidden, fc_hidden=fc_hidden, dropout=dropout
            ),
            num_block=num_enc_block,
            hidden=hidden,
            dropout=dropout,
        )
        self.decoder = Decoder(
            layer=DecoderLayer(
                num_head=num_head, hidden=hidden, fc_hidden=fc_hidden, dropout=dropout
            ),
            num_block=num_dec_block,
            hidden=hidden,
            dropout=dropout,
        )

        # generator
        self.generator = nn.Linear(hidden, vocab_size)

    def forward(self, src, tgt, src_mask, tgt_mask):
        # encode
        src_embedding = self.src_embed(src)
        memory = self.encoder(src_embedding, src_mask)

        # decode
        tgt_embedding = self.tgt_embed(tgt)
        output = self.decoder(tgt_embedding, memory, src_mask, tgt_mask)
        return self.generator(output)


class Encoder(nn.Module):
    """Transformer encoder blocks

    Args:
        layer (nn.Module): Single encoder block
        num_block (int): Number of encoder block
        hidden (int): Hidden dimension of model
        dropout (float): Dropout ratio
    """

    def __init__(self, layer, num_block, hidden, dropout):
        super(Encoder, self).__init__()
        self.layers = clones(layer, num_block)
        self.layernorm = nn.LayerNorm(hidden)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.layernorm(x)


class EncoderLayer(nn.Module):
    """A single encoder block

    Args:
        num_head (int): Number of self-attention head
        hidden (int): Hidden dimension of model
        fc_hidden (int): Hidden dimension of position-wise feedforward layer
        dropout (float): Dropout ratio
    """

    def __init__(self, num_head, hidden, fc_hidden, dropout):
        super(EncoderLayer, self).__init__()
        self.multihead_attn = MultiHeadAttention(num_head, hidden, dropout)
        self.residual_1 = ResidualConnection(hidden, dropout)

        self.feedforward = PositionWiseFeedForward(hidden, fc_hidden, dropout)
        self.residual_2 = ResidualConnection(hidden, dropout)

    def forward(self, x, mask):
        # multi-head self-attention + residual connection
        # self.residual_k requires 'function' on second parameter, so use lambda
        x = self.residual_1(x, lambda x: self.multihead_attn(x, x, x, mask))

        # position-wise feedforward + residual connection
        x = self.residual_2(x, self.feedforward)
        return x


class Decoder(nn.Module):
    """Transformer decoder blocks

    Args:
        layer (nn.Module): Single decoder block
        num_block (int): Number of decoder block
        hidden (int): Hidden dimension of model
        dropout (float): Dropout ratio
    """

    def __init__(self, layer, num_block, hidden, dropout):
        super(Decoder, self).__init__()
        self.layers = clones(layer, num_block)
        self.layernorm = nn.LayerNorm(hidden)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.layernorm(x)


class DecoderLayer(nn.Module):
    """A single decoder block

    Args:
        num_head (int): Number of self-attention head
        hidden (int): Hidden dimension of model
        fc_hidden (int): Hidden dimension of position-wise feedforward layer
        dropout (float): Dropout ratio
    """

    def __init__(self, num_head, hidden, fc_hidden, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(num_head, hidden, dropout)
        self.residual_1 = ResidualConnection(hidden, dropout)

        self.encdec_attn = MultiHeadAttention(num_head, hidden, dropout)
        self.residual_2 = ResidualConnection(hidden, dropout)

        self.feedforward = PositionWiseFeedForward(hidden, fc_hidden, dropout)
        self.residual_3 = ResidualConnection(hidden, dropout)

    def forward(self, x, memory, src_mask, tgt_mask):
        # x is a target input
        # memory is an encoder output
        m = memory
        x = self.residual_1(x, lambda x: self.self_attn(x, x, x, tgt_mask.squeeze(1)))
        x = self.residual_2(x, lambda x: self.encdec_attn(x, m, m, src_mask))
        return self.residual_3(x, self.feedforward)
