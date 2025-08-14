from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.parallel import DistributedDataParallel as DDP


class PatchEmbedding(nn.Module):
    def __init__(self, dem_input_dim, rain_num_steps, width):
        super().__init__()
        scale = width ** -0.5
        self.dem_projection = nn.Linear(dem_input_dim, width) 
        self.grid_size_embed = nn.Parameter(scale * torch.randn(5, width))
        self.rain_embed = nn.Sequential(
                nn.Linear(rain_num_steps, 512), # B,N,48 --> B,N,512
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Linear(512, width-2) # 2 for water_level, has_water --> B,N,C-2
            )
        self.pos_embed = nn.Sequential(
                            nn.Linear(2, 512), # N,2 --> N,512
                            nn.GELU(),
                            nn.Linear(512, width), # N,512 --> N,C
                        )

    def forward(self, x, rain, dem_embed, side_lens, square_centers):
        dem_embed = self.dem_projection(dem_embed) # N,1280 --> N,C

        # Reshape for BatchNorm1d
        B, N, _ = rain.shape
        rain = rain.reshape(B * N, -1)  # B*N,48
        rain_embed = self.rain_embed(rain)  # B*N,48 --> B*N,C-2
        rain_embed = rain_embed.reshape(B, N, -1)  # B,N,C-2

        x = torch.cat((x, rain_embed), dim=-1) # B,N,C
        area = self.grid_size_embed[side_lens] # 5,C[N] --> N,C
        pos = self.pos_embed(square_centers) # N,2 --> N,C
        x = x + dem_embed + area + pos # B,N,C
        return x


class ResidualAttentionBlock(nn.Module):
    def __init__(self, width: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(width, n_head)
        self.ln_1 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(width, width * 4)),
            ("gelu", nn.GELU()),
            ("c_proj", nn.Linear(width * 4, width))
        ]))
        self.ln_2 = nn.LayerNorm(width)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask if self.attn_mask is not None else None
        # use flash attention backend
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            output = self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
        return output

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)




class FloodTransformer(nn.Module):
    def __init__(self,
                 context_length: int,
                 dem_input_dim: int,
                 rain_num_steps: int,
                 width: int,
                 heads: int,
                 layers: int,
                 pred_length: int
                 ):
        super().__init__()

        self.context_length = context_length

        self.patch_embedding = PatchEmbedding(dem_input_dim, rain_num_steps, width)

        self.transformer = Transformer(
            width=width,
            layers=layers,
            heads=heads,
            attn_mask=None
        )

        self.regression_head = nn.Sequential(
            nn.Linear(width, width * 4),
            nn.GELU(),
            nn.Linear(width * 4, pred_length),
            )
        
        self.classification_head = nn.Sequential(
            nn.Linear(width, width * 4),
            nn.GELU(),
            nn.Linear(width * 4, pred_length),
            )

        self.ln_final = nn.LayerNorm(width)

        self.initialize_parameters()

    def initialize_parameters(self):
        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)


    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask


    def forward(self, x, rain, dem_embed, side_lens, square_centers):
        x = self.patch_embedding(x, rain, dem_embed, side_lens, square_centers)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        water_level_pred = self.regression_head(x)
        has_water_pred = self.classification_head(x)

        return water_level_pred, has_water_pred

    def setup_ddp(self, rank):
        """Setup DDP training"""
        self.to(rank)  # move model to GPU
        self = DDP(self, device_ids=[rank])
        return self