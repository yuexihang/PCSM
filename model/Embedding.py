import math
import torch
import torch.nn as nn
from einops import rearrange


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, min_freq=1 / 2, scale=1.):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.min_freq = min_freq
        self.scale = scale
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, coordinates, device):
        # coordinates [b, n]
        t = coordinates.to(device).type_as(self.inv_freq)
        t = t * (self.scale / self.min_freq)
        freqs = torch.einsum('... i , j -> ... i j', t, self.inv_freq) 
        return torch.cat((freqs, freqs), dim=-1)  


def rotate_half(x):
    x = rearrange(x, '... (j d) -> ... j d', j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(t, freqs):
    return (t * freqs.cos()) + (rotate_half(t) * freqs.sin())


def apply_2d_rotary_pos_emb(t, freqs_x, freqs_y):
    d = t.shape[-1]
    t_x, t_y = t[..., :d // 2], t[..., d // 2:]

    return torch.cat((apply_rotary_pos_emb(t_x, freqs_x),
                      apply_rotary_pos_emb(t_y, freqs_y)), dim=-1)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout, max_len=421 * 421):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)

class RelativePositionalEncoder(nn.Module):
    def __init__(self, emb_dim, max_position=512):
        super(RelativePositionalEncoder, self).__init__()
        self.max_position = max_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_position * 2 + 1, emb_dim))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, seq_len_q, seq_len_k):
        range_vec_q = torch.arange(seq_len_q)
        range_vec_k = torch.arange(seq_len_k)
        relative_matrix = range_vec_k[None, :] - range_vec_q[:, None]
        clipped_relative_matrix = torch.clamp(relative_matrix, -self.max_position, self.max_position)
        relative_position_matrix = clipped_relative_matrix + self.max_position
        embeddings = self.embeddings_table[relative_position_matrix]

        return embeddings

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len, requires_grad=False, sparse = 1):
        super(PositionalEmbedding, self).__init__()

        self.encoding = torch.zeros(max_len, d_model)

        pos = torch.arange(0, max_len)
        pos = pos.float().unsqueeze(dim=1)

        _2i = torch.arange(0, d_model, step=2).float()

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

        self.sparse = sparse

        self.encoding.requires_grad = requires_grad

    def forward(self, seq_len):

        return self.encoding[::self.sparse, :][:seq_len, :]


def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):

    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
