import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from model.Embedding import timestep_embedding
from model.spectral_embedding import robust_spectral
import math
from einops import rearrange

ACTIVATION = {'gelu': nn.GELU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU(0.1),
              'softplus': nn.Softplus, 'ELU': nn.ELU, 'silu': nn.SiLU}

class Calibrated_Spectral_Mixer(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., freq_num=64, H=101, W=31, kernelsize = 3): 
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        self.H = H
        self.W = W

        # Mixer Modules
        self.in_project_x = nn.Conv2d(dim, inner_dim, kernelsize, 1, kernelsize//2 )
        self.in_project_fx = nn.Conv2d(dim, inner_dim, kernelsize, 1, kernelsize//2 )
        self.mlp_trans_weights = nn.Parameter( torch.empty((dim_head, dim_head)) )
        torch.nn.init.kaiming_uniform_(self.mlp_trans_weights, a=math.sqrt(5))
        self.in_project_gates = nn.Linear(dim_head, freq_num)
        for l in [self.in_project_gates]:
            torch.nn.init.orthogonal_(l.weight) 
        self.layernorm2 = nn.LayerNorm( ( freq_num, dim_head ) )
        self.to_out = nn.Sequential( nn.Linear(inner_dim, dim), nn.Dropout(dropout) )

        # Eigenfunctions
        max_freq_num = freq_num + 3
        BASE_MATRIX = robust_spectral.grid_spectral_points( self.H, self.W, k = max_freq_num )
        BASE_MATRIX = BASE_MATRIX.reshape( -1, max_freq_num ).cuda()       # n, max_m
        inver = BASE_MATRIX[ :, :freq_num ]
        inver = F.normalize( inver, p = 2, dim = -1 ) # L2 Norm
        self.inver = nn.Parameter( inver , requires_grad=False)  # h, n, m
        
    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).contiguous().permute(0, 3, 1, 2).contiguous()  # B C H W
        fx_mid = self.in_project_fx(x).permute(0, 2, 3, 1).contiguous().reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()  # B H N C
        
        # Spectral Transform
        x_mid = self.in_project_x(x).permute(0, 2, 3, 1).contiguous().reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()  # B H N C
        eigen_gate = F.softmax(self.in_project_gates(x_mid) / torch.clamp(self.temperature, min=0.1, max=5), dim = -1)  # B H N G
        spectral_embedding = self.inver[ None, None , : , : ] # 1, 1, n, m
        eigens = eigen_gate * spectral_embedding # B H N G
        spectral_feature = torch.einsum("bhnc,bhng->bhgc", fx_mid, eigens)

        # Spectral Domain Processing
        bsize, hsize, gsize, csize = spectral_feature.shape
        spectral_feature = self.layernorm2(spectral_feature.reshape( -1, gsize, csize )).reshape( bsize, hsize, gsize, csize )
        out_spectral_feature = torch.einsum("bhgi,io->bhgo", spectral_feature, self.mlp_trans_weights)

        # Inverse Spectral Transform
        out_x = torch.einsum("bhgc,bhng->bhnc", out_spectral_feature, eigens)
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')

        return self.to_out(out_x)


class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu', res=True):
        super(MLP, self).__init__()

        if act in ACTIVATION.keys():
            act = ACTIVATION[act]
        else:
            raise NotImplementedError
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList([nn.Sequential(nn.Linear(n_hidden, n_hidden), act()) for _ in range(n_layers)])

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x


class Mixer_Block(nn.Module):
    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            dropout: float,
            act='gelu',
            mlp_ratio=4,
            last_layer=False,
            out_dim=1,
            freq_num=32,
            H=85,
            W=85,
            kernelsize=3,
    ):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = Calibrated_Spectral_Mixer(hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,dropout=dropout, freq_num=freq_num, H=H, W=W, kernelsize=kernelsize)

        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)
        
    def forward(self, fx):
        fx = self.Attn(self.ln_1(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        else:
            return fx


class Model(nn.Module):
    def __init__(self,
                 space_dim=1,
                 n_layers=5,
                 n_hidden=256,
                 dropout=0.0,
                 n_head=8,
                 Time_Input=False,
                 act='gelu',
                 mlp_ratio=1,
                 fun_dim=1,
                 out_dim=1,
                 freq_num=32,
                 ref=8,
                 unified_pos=False,
                 H=85,
                 W=85, 
                 kernelsize=3,
                 ):
        super(Model, self).__init__()
        self.__name__ = 'PCSM_Structured'
        self.H = H
        self.W = W
        self.ref = ref
        self.unified_pos = unified_pos
        if self.unified_pos:
            self.pos = self.get_grid()
            self.preprocess = MLP(fun_dim + self.ref * self.ref, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)
        else:
            self.preprocess = MLP(fun_dim + space_dim, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)

        self.Time_Input = Time_Input
        self.n_hidden = n_hidden
        self.space_dim = space_dim
        if Time_Input:
            self.time_fc = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.SiLU(), nn.Linear(n_hidden, n_hidden))

        self.blocks = nn.ModuleList([Mixer_Block(num_heads=n_head, hidden_dim=n_hidden,
                                                      dropout=dropout,
                                                      act=act,
                                                      mlp_ratio=mlp_ratio,
                                                      out_dim=out_dim,
                                                      freq_num=freq_num,
                                                      H=H,
                                                      W=W,
                                                      kernelsize=kernelsize,
                                                      last_layer=(layer_id == n_layers - 1))
                                     for layer_id in range(n_layers)])
        self.initialize_weights()
        self.placeholder = nn.Parameter((1 / (n_hidden)) * torch.rand(n_hidden, dtype=torch.float))

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_grid(self, batchsize=1):
        size_x, size_y = self.H, self.W
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        grid = torch.cat((gridx, gridy), dim=-1).cuda()  # B H W 2

        gridx = torch.tensor(np.linspace(0, 1, self.ref), dtype=torch.float)
        gridx = gridx.reshape(1, self.ref, 1, 1).repeat([batchsize, 1, self.ref, 1])
        gridy = torch.tensor(np.linspace(0, 1, self.ref), dtype=torch.float)
        gridy = gridy.reshape(1, 1, self.ref, 1).repeat([batchsize, self.ref, 1, 1])
        grid_ref = torch.cat((gridx, gridy), dim=-1).cuda()  # B H W 8 8 2

        pos = torch.sqrt(torch.sum((grid[:, :, :, None, None, :] - grid_ref[:, None, None, :, :, :]) ** 2, dim=-1)). \
            reshape(batchsize, size_x, size_y, self.ref * self.ref).contiguous()
        return pos

    def forward(self, x, fx, T=None):
        if self.unified_pos:
            x = self.pos.repeat(x.shape[0], 1, 1, 1).reshape(x.shape[0], self.H * self.W, self.ref * self.ref)
        if fx is not None:
            fx = torch.cat((x, fx), -1)
            fx = self.preprocess(fx)
        else:
            fx = self.preprocess(x)
            fx = fx + self.placeholder[None, None, :]

        if T is not None:
            Time_emb = timestep_embedding(T, self.n_hidden).repeat(1, x.shape[1], 1)
            Time_emb = self.time_fc(Time_emb)
            fx = fx + Time_emb

        for block in self.blocks:
            fx = block(fx)

        return fx
