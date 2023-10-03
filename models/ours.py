import torch
from torch import nn

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange

# helpers

def exists(val):
    return val is not None

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def aggregate_variables(self, x: torch.Tensor):
    """
    x: B, V, L, D
    """
    b, _, l, _ = x.shape
    x = torch.einsum("bvld->blvd", x)
    x = x.flatten(0, 1)  # BxL, V, D

    var_query = self.var_query.repeat_interleave(x.shape[0], dim=0)
    x, _ = self.var_agg(var_query, x, x)  # BxL, D
    x = x.squeeze()

    x = x.unflatten(dim=0, sizes=(b, l))  # B, L, D
    return x
    
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = (dim // self.heads)** -0.5 #dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        weights = self.attend(dots)
        attn = self.dropout(weights)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')


        return self.to_out(out), weights

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    def forward(self, x):
        attn_weights = []
        for attn, ff in self.layers:
            x_att, weight = attn(x)
            x = x_att + x
            x = ff(x) + x

            attn_weights.append(weight)

        return self.norm(x), attn_weights

class aggregate_variables(nn.Module):
    def __init__(self, dims, heads):
        super().__init__()
        """
        x: B, V, L, D
        """
        self.dims = dims
        self.heads = heads

        # variable aggregation: a learnable query and a single-layer cross attention
        self.var_query = nn.Parameter(torch.zeros(1, 1, dims), requires_grad=True)#.to(device)
        self.var_agg   = nn.MultiheadAttention(self.dims, self.heads)#.to(device)

    def forward(self, G1, G2, G3, G4, G5):
        
        b, l, _,  _ = G1.shape

        #G1 = torch.einsum("bvld->blvd", G1)
        G1 = G1.flatten(0, 1)  # BxL, V, D
        var_query1 = self.var_query.repeat_interleave(G1.shape[0], dim=0)
        G1, _ = self.var_agg(var_query1, G1, G1)  # BxL, D
        G1 = G1.squeeze()
        G1 = torch.reshape(G1, (b, 1, l, self.dims))


        #G2 = torch.einsum("bvld->blvd", G2)
        G2 = G2.flatten(0, 1)  # BxL, V, D
        var_query2 = self.var_query.repeat_interleave(G2.shape[0], dim=0)
        G2, _ = self.var_agg(var_query2, G2, G2)  # BxL, D
        G2 = G2.squeeze()
        G2 = torch.reshape(G2, (b, 1, l, self.dims))

        #G3 = torch.einsum("bvld->blvd", G3)
        G3 = G3.flatten(0, 1)  # BxL, V, D
        var_query3 = self.var_query.repeat_interleave(G3.shape[0], dim=0)
        G3, _ = self.var_agg(var_query3, G3, G3)  # BxL, D
        G3 = G3.squeeze()
        G3 = torch.reshape(G3, (b, 1, l, self.dims))

        #G4 = torch.einsum("bvld->blvd", G4)
        G4 = G4.flatten(0, 1)  # BxL, V, D
        var_query4 = self.var_query.repeat_interleave(G4.shape[0], dim=0)
        G4, _ = self.var_agg(var_query4, G4, G4)  # BxL, D
        G4 = G4.squeeze()
        G4 = torch.reshape(G4, (b, 1, l, self.dims))

        #G5 = torch.einsum("bvld->blvd", G5)
        G5 = G5.flatten(0, 1)  # BxL, V, D
        var_query5 = self.var_query.repeat_interleave(G5.shape[0], dim=0)
        G5, _ = self.var_agg(var_query5, G5, G5)  # BxL, D
        G5 = G5.squeeze()
        G5 = torch.reshape(G5, (b, 1, l, self.dims))
        
        agg = torch.cat([G1, G2, G3, G4, G5], dim = 1)

        return agg

class ViT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        image_patch_size,
        frames,
        frame_patch_size,
        num_classes,
        dim,
        spatial_depth,
        temporal_depth,
        heads,
        mlp_dim,
        pool = 'cls',
        channels = 3,
        dim_head = 64,
        dropout = 0.,
        emb_dropout = 0.
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(image_patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        #assert frames % frame_patch_size == 0, 'Frames must be divisible by frame patch size'

        num_image_patches = (image_height // patch_height) * (image_width // patch_width)
        num_frame_patches = 4 #(frames // frame_patch_size)

        

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.global_average_pool = pool == 'mean'

        patch_dim1 = patch_height * patch_width #* frame_patch_size[0] * num_frame_patches
        self.to_patch_embedding_G1 = nn.Sequential(
            Rearrange('b c (f pf) (h p1) (w p2) -> b f (h w pf c) (p1 p2)', p1 = patch_height, p2 = patch_width, pf = frame_patch_size[0]),
            nn.LayerNorm(patch_dim1),
            nn.Linear(patch_dim1, dim),
            nn.LayerNorm(dim)
        )
        patch_dim2 = patch_height * patch_width #* frame_patch_size[1]* num_frame_patches
        self.to_patch_embedding_G2 = nn.Sequential(
            Rearrange('b c (f pf) (h p1) (w p2) -> b f (h w pf c) (p1 p2)', p1 = patch_height, p2 = patch_width, pf = frame_patch_size[1]),
            nn.LayerNorm(patch_dim2),
            nn.Linear(patch_dim2, dim),
            nn.LayerNorm(dim)
        )

        patch_dim3 = patch_height * patch_width #* frame_patch_size[2]* num_frame_patches
        self.to_patch_embedding_G3 = nn.Sequential(
            Rearrange('b c (f pf) (h p1) (w p2) -> b f (h w pf c) (p1 p2)', p1 = patch_height, p2 = patch_width, pf = frame_patch_size[2]),
            nn.LayerNorm(patch_dim3),
            nn.Linear(patch_dim3, dim),
            nn.LayerNorm(dim)
        )
        patch_dim4 =  patch_height * patch_width #* frame_patch_size[3]* num_frame_patches
        self.to_patch_embedding_G4 = nn.Sequential(
            Rearrange('b c (f pf) (h p1) (w p2) -> b f (h w pf c) (p1 p2)', p1 = patch_height, p2 = patch_width, pf = frame_patch_size[3]),
            nn.LayerNorm(patch_dim4),
            nn.Linear(patch_dim4, dim),
            nn.LayerNorm(dim)
        )

        patch_dim5 = patch_height * patch_width #* frame_patch_size[4]* num_frame_patches
        self.to_patch_embedding_G5 = nn.Sequential(
            Rearrange('b c (f pf) (h p1) (w p2) -> b f (h w pf c) (p1 p2)', p1 = patch_height, p2 = patch_width, pf = frame_patch_size[4]),
            nn.LayerNorm(patch_dim5),
            nn.Linear(patch_dim5, dim),
            nn.LayerNorm(dim)
        )

        self.agg = aggregate_variables(dim, heads)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_frame_patches, num_image_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.spatial_cls_token = nn.Parameter(torch.randn(1, 1, dim)) if not self.global_average_pool else None
        self.temporal_cls_token = nn.Parameter(torch.randn(1, 1, dim)) if not self.global_average_pool else None

        self.spatial_transformer = Transformer(dim, spatial_depth, heads, dim_head, mlp_dim, dropout)
        self.temporal_transformer = Transformer(dim, temporal_depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, in_):

        G1 = rearrange(in_[..., :17], 'b t w h d -> b 1 (t d) w h')
        G2 = rearrange(in_[..., 17:40], 'b t w h d -> b 1 (t d) w h')
        G3 = rearrange(in_[..., 40:60], 'b t w h d -> b 1 (t d) w h')
        G4 = rearrange(in_[..., 60:80], 'b t w h d -> b 1 (t d) w h')
        G5 = rearrange(in_[..., 80:97], 'b t w h d -> b 1 (t d) w h')
        

        G1 = self.to_patch_embedding_G1(G1)
        G2 = self.to_patch_embedding_G2(G2)
        G3 = self.to_patch_embedding_G3(G3)
        G4 = self.to_patch_embedding_G4(G4)
        G5 = self.to_patch_embedding_G5(G5)

        
        
        b, f1, n1, _ = G1.shape
        b, f2, n2, _ = G2.shape
        b, f3, n3, _ = G3.shape
        b, f4, n4, _ = G4.shape
        b, f5, n5, _ = G5.shape

        G1 = G1 + self.pos_embedding[:, :, :n1]
        G2 = G2 + self.pos_embedding[:, :, :n2]
        G3 = G3 + self.pos_embedding[:, :, :n3]
        G4 = G4 + self.pos_embedding[:, :, :n4]
        G5 = G5 + self.pos_embedding[:, :, :n5]

        # aggreagation: 
        var_agg = self.agg(G1, G2, G3, G4, G5)
        print(var_agg.shape)

        if exists(self.spatial_cls_token):
            spatial_cls_tokens = repeat(self.spatial_cls_token, '1 1 d -> b f 5 d', b = b, f = f5)
            x = torch.cat((spatial_cls_tokens, x), dim = 2)

        x = self.dropout(x)
        print(f"patch_embedding: {x.shape}")
        x = rearrange(x, 'b f n d -> (b f) n d')
        print(f"patch_embedding: {x.shape}")

        # attend across space

        x, weights_attn_spatial = self.spatial_transformer(x)
        print(f" after self_atten: {x.shape}")

        x = rearrange(x, '(b f) n d -> b f n d', b = b)

        # excise out the spatial cls tokens or average pool for temporal attention

        x = x[:, :, 0] if not self.global_average_pool else reduce(x, 'b f n d -> b f d', 'mean')

        # append temporal CLS tokens

        if exists(self.temporal_cls_token):
            temporal_cls_tokens = repeat(self.temporal_cls_token, '1 1 d-> b 1 d', b = b)

            x = torch.cat((temporal_cls_tokens, x), dim = 1)

        # attend across time

        x, weights_attn_temporal = self.temporal_transformer(x)

        # excise out temporal cls token or average pool

        x = x[:, 0] if not self.global_average_pool else reduce(x, 'b f d -> b d', 'mean')

        x = self.to_latent(x)

        return [weights_attn_spatial, weights_attn_temporal], self.mlp_head(x)