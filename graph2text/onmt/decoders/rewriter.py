"""
Implementation of "Attention is All You Need"
"""

import torch.nn as nn

from onmt.encoders.encoder import EncoderBase
from onmt.modules import MultiHeadedAttention
from onmt.modules.position_ffn import PositionwiseFeedForward
from onmt.utils.misc import sequence_mask
import torch


class RewriterLayer(nn.Module):
    """
    A single layer of the transformer encoder.

    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, d_model, heads, d_ff, dropout, attention_dropout,
                 max_relative_positions=0):
        super(RewriterLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=attention_dropout,
            max_relative_positions=max_relative_positions)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.ffn_layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, text, graph):
        """
        Args:
            inputs (FloatTensor): ``(batch_size, src_len, model_dim)``
            mask (LongTensor): ``(batch_size, 1, src_len)``

        Returns:
            (FloatTensor):

            * outputs ``(batch_size, src_len, model_dim)``
        """
        graph_norm = self.layer_norm_1(graph)
        text_norm = self.layer_norm_2(text)
        context, attns = self.self_attn(graph_norm, graph_norm, text_norm,
                                    mask=None, attn_type="context")
        out = self.dropout(context) + text

        out_norm = self.ffn_layer_norm(out)
        outputs = self.feed_forward(out_norm)
        out = outputs + out

        return out, attns

    def update_dropout(self, dropout, attention_dropout):
        self.self_attn.update_dropout(attention_dropout)
        self.feed_forward.update_dropout(dropout)
        self.dropout.p = dropout


class Rewriter(nn.Module):
    def __init__(self, num_layers=4, d_model=512, heads=8, d_ff=512, dropout=0.3,
                 attention_dropout=0.3, max_relative_positions=0):
        super(Rewriter, self).__init__()

        self.d_model = d_model
        self.rewriter = nn.ModuleList(
            [RewriterLayer(
                self.d_model, heads, d_ff, dropout, attention_dropout,
                max_relative_positions=max_relative_positions)
             for i in range(num_layers)])
        self.layer_norm = nn.LayerNorm(self.d_model, eps=1e-6)

    def forward(self, text, graph):
        for i, layer in enumerate(self.rewriter):
            text, attns = layer(text, graph)

        out = self.layer_norm(text).transpose(0, 1).contiguous()

        return out

    def update_dropout(self, dropout, attention_dropout):
        self.embeddings.update_dropout(dropout)
        for layer in self.transformer:
            layer.update_dropout(dropout, attention_dropout)