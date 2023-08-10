import copy
from turtle import forward

import numpy as np
import torch
import torch.nn as nn

from .positional_encoder import PositionalEncoder
from .ltae import LTAE2d


class mLTAE2d(nn.Module):
    def __init__(
        self,
        in_channels,
        n_head=8,
        d_k=4,
        d_model=128,
        d_ffn=256,
        dropout=0.1,
        positional_encoding=True,
        return_att=False,
    ) -> None:
        super().__init__()
        self.return_att = return_att
        self.seq_ltae = sLTAE2d(
            in_channels=in_channels,
            n_head=n_head,
            d_k=d_k,
            d_model=d_model,
            d_ffn=d_ffn,
            dropout=dropout,
            gelu=True,
            positional_encoding=positional_encoding,
            return_att=return_att,
        )
        self.ltae = LTAE2d(
            in_channels=d_model,
            n_head=n_head,
            d_k=d_k,
            d_model=d_model,
            mlp=[d_model, in_channels],
            dropout=dropout,
            positional_encoding=positional_encoding,
            return_att=return_att,
        )

    def forward(self, x, batch_positions=None, pad_mask=None, return_both_att=False):
        if self.return_att:
            out, att1 = self.seq_ltae(
                x, batch_positions=batch_positions, pad_mask=pad_mask
            )
            out, att2 = self.ltae(
                out, batch_positions=batch_positions, pad_mask=pad_mask
            )
            if return_both_att:
                return out, (att1, att2)
            else:
                return out, att2
        else:
            out = self.seq_ltae(x, batch_positions=batch_positions, pad_mask=pad_mask)
            out = self.ltae(out, batch_positions=batch_positions, pad_mask=pad_mask)
            return out


class sLTAE2d(nn.Module):
    def __init__(
        self,
        in_channels=128,
        n_head=16,
        d_k=4,
        dropout=0.2,
        d_model=256,
        d_ffn=512,
        gelu=False,
        bn_ffn=True,
        T=1000,
        return_att=False,
        positional_encoding=True,
    ):
        super(sLTAE2d, self).__init__()
        self.in_channels = in_channels
        self.return_att = return_att
        self.n_head = n_head

        self.d_model = d_model
        self.inconv = nn.Conv1d(in_channels, d_model, 1)

        if positional_encoding:
            self.positional_encoder = PositionalEncoder(
                self.d_model // n_head, T=T, repeat=n_head
            )
        else:
            self.positional_encoder = None

        self.layernorm1 = nn.LayerNorm(d_model)

        self.attention_heads = MultiHeadAttention(
            n_head=n_head, d_k=d_k, d_in=self.d_model
        )

        activation = nn.GELU() if gelu else nn.ReLU()
        n_neurons = [d_model, d_ffn, d_model]
        layers = []
        for i in range(len(n_neurons) - 1):
            if bn_ffn:
                layers.extend(
                    [
                        nn.Linear(n_neurons[i], n_neurons[i + 1]),
                        nn.BatchNorm1d(n_neurons[i + 1]),
                        activation,
                    ]
                )
            else:
                layers.extend([nn.Linear(n_neurons[i], n_neurons[i + 1]), activation])

        self.ffn = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        self.layernorm2 = nn.LayerNorm(d_model)

    def forward(self, x, batch_positions=None, pad_mask=None):
        sz_b, seq_len, d, h, w = x.shape
        out = x.permute(0, 3, 4, 1, 2).contiguous().view(sz_b * h * w, seq_len, d)

        if pad_mask is not None:
            pad_mask = (
                pad_mask.unsqueeze(-1)
                .repeat((1, 1, h))
                .unsqueeze(-1)
                .repeat((1, 1, 1, w))
            )  # BxTxHxW
            pad_mask = (
                pad_mask.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)
            )

        if self.inconv is not None:
            out = self.inconv(out.permute(0, 2, 1)).permute(0, 2, 1)

        if self.positional_encoder is not None:
            bp = (
                batch_positions.unsqueeze(-1)
                .repeat((1, 1, h))
                .unsqueeze(-1)
                .repeat((1, 1, 1, w))
            )  # BxTxHxW
            bp = bp.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)
            out = out + self.positional_encoder(bp)

        ## residual MHSA
        res = out
        out = self.layernorm1(out)
        outt = self.attention_heads(out, out, out, pad_mask=pad_mask)
        out, attn = outt[0], outt[1]
        out = (
            out.permute(1, 2, 0, 3).contiguous().view(sz_b*h*w, seq_len, -1)
        )  # Concatenate heads
        out = res + self.dropout(out)

        ## residual FFN
        res = out
        out = self.layernorm2(out)
        out = self.dropout(self.ffn(out.view(sz_b *h*w* seq_len, -1))).view(
            sz_b*h*w, seq_len, -1
        )
        out = res + out

        ## Reshape spatial dimensions
        out = out.view(sz_b, h, w,seq_len,  -1).permute(0, 3, 4, 1, 2)
        attn = attn.view(self.n_head, sz_b, h, w, seq_len, seq_len).permute(0, 1, 4, 5, 2, 3)

        if self.return_att:
            return out, attn
        else:
            return out


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module"""

    def __init__(self, n_head, d_k, d_in):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_in = d_in

        self.fc1_q = nn.Linear(d_in, n_head * d_k)
        nn.init.normal_(self.fc1_q.weight, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.fc1_k = nn.Linear(d_in, n_head * d_k)
        nn.init.normal_(self.fc1_k.weight, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))

    def forward(self, q, k, v, pad_mask=None):
        d_k, d_in, n_head = self.d_k, self.d_in, self.n_head
        sz_b, seq_len, _ = q.size()

        q = self.fc1_q(v).view(sz_b, seq_len, n_head, d_k)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_k)  # (n*b) x lk x dk

        k = self.fc1_k(v).view(sz_b, seq_len, n_head, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_k)  # (n*b) x lk x dk

        if pad_mask is not None:
            pad_mask = pad_mask.repeat(
                (n_head, 1)
            )  # replicate pad_mask for each head (nxb) x lk

        v = torch.stack(v.split(v.shape[-1] // n_head, dim=-1)).view(
            n_head * sz_b, seq_len, -1
        )
        out = self.attention(q, k, v, pad_mask=pad_mask)

        output, attn = out[0], out[1]
        attn = attn.view(n_head, sz_b, seq_len, seq_len)

        output = output.view(n_head, sz_b, seq_len, d_in // n_head)

        return output, attn


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention"""

    def __init__(self, temperature, attn_dropout=0.0):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, pad_mask=None, return_comp=False):
        attn = torch.matmul(q, k.transpose(1, 2))
        attn = attn / self.temperature
        if pad_mask is not None:
            attn = attn.masked_fill(pad_mask.unsqueeze(1), -1e3)
        compat = attn
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)
        if return_comp:
            return output, attn, compat
        else:
            return output, attn


