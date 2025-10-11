from torch import nn
import torch, math, pdb
from copy import deepcopy
from ..modules.mlp import build_mlp
from torch.nn import functional as F
from einops import rearrange, repeat
from .transformer import Transformer
from ..engine.ema import requires_grad
from .transformer import build_perceptron
from einops.layers.torch import Rearrange
from ..engine.util import instantiate_from_config
from .transformer import Transformer, DeformableTransformerEncoder

class Encoder(nn.Module):

    def __init__(self, image_size, layer_type, n_carrier,
                 patch_size, dim, depth, num_head, mlp_dim,
                 in_channels=1024, d_model=256, dim_head=64, 
                 enc_n_points = 4, visual_encoder_config = None, dropout=0.):

        super().__init__()

        self.backbone = instantiate_from_config(visual_encoder_config)

        layer = build_mlp(in_channels, dim, dim, 2)
        self.projectors = _get_clones(layer, 4)
        self.norm_layers = _get_clones(nn.LayerNorm(dim), 4)

        self.image_size = image_size
        self.patch_size = patch_size
        
        self.n_carrier = n_carrier
        scale = dim ** -0.5
        self.carrier_tokens = nn.Parameter(torch.randn(1, self.n_carrier, dim) * scale)
        self.slots_pos = nn.Parameter(torch.randn(1, self.n_carrier, dim) * scale)
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        
        self.transformer = DeformableTransformerEncoder(dim, depth, num_feature_levels=4,enc_n_points=enc_n_points)
        self.norm_pre1 = nn.LayerNorm(dim)

        self.initialize_weights()

    def freeze_visual_encoder(self):

        requires_grad(self.backbone, False)
            
    def freeze(self):

        self.eval()
        requires_grad(self, False)

    def vec2tensor(self, x):

        assert x.ndim == 3
        H = W = int(math.sqrt(x.size(1)))
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        return x

    def forward(self, imgs):

        self.backbone.eval()
        with torch.no_grad():
            latent = self.backbone(imgs)

        kv_features = []
        for i, layer in enumerate(self.projectors):
            kv_feature = self.norm_layers[i](layer(latent[i]))
            kv_feature = self.vec2tensor(kv_feature)
            kv_features.append(kv_feature)

        bs = imgs.size(0)
        carrier = self.carrier_tokens.repeat(bs, 1, 1)
        carrier = self.norm_pre1(carrier)

        carrier = self.vec2tensor(carrier)
        output = self.transformer(carrier, self.slots_pos, kv_features)

        return output, latent[-1]

    def interpolate_pos_encoding(self, x, w, h):

        npatch = x.shape[1]
        N = self.pos_embed.shape[1]
        if npatch == N and w == h:
            return self.pos_embed
        
        patch_pos_embed = self.pos_embed
        dim = x.shape[-1]

        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = F.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        return patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

    def prepare_tokens(self, x):
        
        _, _, w, h = x.shape
        # add positional encoding to each token
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = x + self.interpolate_pos_encoding(x, w, h)
        
        return x

    def initialize_weights(self):

        if self.backbone:
            requires_grad(self.backbone, False)

        self.norm_pre1.apply(self._init_weights)
        # self.norm_pre2.apply(self._init_weights)

        for m in self.transformer.parameters():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class Decoder(nn.Module):
    def __init__(self, layer_type, image_size, patch_size, dim, n_carrier, depth,
                 num_head, mlp_dim, dim_head=64, dropout=0., num_register_tokens=4,):
        
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'

        self.dim = dim
        scale = dim ** -0.5
        self.num_patches = (image_size // patch_size) ** 2
        self.num_tokens = 1
        self.num_register_tokens = num_register_tokens

        self.position_embedding = nn.Parameter(torch.randn(1, 576 + 1, dim) * scale)
        self.slot_position_embedding = nn.Parameter(torch.randn(1, n_carrier, dim) * scale)

        self.mask_embedding = nn.Parameter(torch.randn(1, 1, dim) * scale)
        self.mask_dino = nn.Parameter(torch.randn(1, 1, dim) * scale)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * scale)

        self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, dim))

        self.transformer = Transformer(layer_type, dim, depth, num_head, dim_head, mlp_dim, dropout, xformer=False)
        
        self.norm_post1 = nn.LayerNorm(dim)
        self.norm_post2 = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, 1024)

        self.initialize_weights()

    def initialize_weights(self):

        self.apply(self._init_weights)

    def _init_weights(self, m):

        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def interpolate_pos_encoding(self, x, w, h):

        npatch = w * h
        N = self.position_embedding.size(1) - 1
        if npatch == N and w == h:
            return self.position_embedding

        class_pos_embed = self.position_embedding[:, :1]
        patch_pos_embed = self.position_embedding[:, 1:]
        dim = x.shape[2]
        
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w + 0.1, h + 0.1

        m = patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2)
        patch_pos_embed = F.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )

        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        # return patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        pos_embed = torch.cat((class_pos_embed, patch_pos_embed.flatten(2).transpose(1, 2)), dim=1)
        
        return pos_embed

    def prepare_tokens(self, x):

        w = h = int(math.sqrt(x.size(1)))
        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)
        
        return x
    
    def single_forward(self, conds, z, norm, reshape=False):

        bs, n_c, _ = conds.shape
        num_patches = z.size(1)
        H = W = int(math.sqrt(num_patches))
        
        x = torch.cat((self.norm_post1(conds), norm(z)), dim=1)
        mask = ~torch.tril(x.new_ones((n_c + num_patches, n_c + num_patches), dtype=torch.bool), diagonal=0)
        
        x = self.transformer(x, mask=mask)
        if reshape:
            z = rearrange(x[:, n_c + self.num_tokens:n_c + num_patches-self.num_register_tokens], 'b (h w) c -> b c h w', h=H, w=W)
        else:
            z = x[:, n_c + self.num_tokens:n_c + num_patches-self.num_register_tokens]
        return z
    
    def forward(self, slots):

        bs = slots.size(0)
        memory = slots + self.slot_position_embedding

        H = W = int(math.sqrt(self.num_patches))
        z_pos = self.interpolate_pos_encoding(slots, W, H)

        z = repeat(self.mask_embedding, 'f ... -> (b f) ...', b=bs)
        cls_token = repeat(self.cls_token, 'f ... -> (b f) ...',b=bs)

        register_tokens = repeat(self.register_tokens, 'f ... -> (b f) ...', b=bs)
        cc = torch.cat((cls_token + z_pos[:, :1], z + z_pos[:, 1:], register_tokens), dim=1)

        z = self.single_forward(memory, cc, self.norm_post1, True)

        if self.training:
            dinov2 = self.mask_dino + self.position_embedding[:, 1:]
            dinov2 = repeat(dinov2, 'f ... -> (b f) ...', b=bs)
            dinov2 = torch.cat((cls_token + self.position_embedding[:,:1], dinov2, register_tokens), dim=1)
            dinov2 = self.single_forward(memory, dinov2, self.norm_post2, False)
            dinov2 = self.proj(dinov2)
            return z, dinov2
        else:
            return z, None

def _get_clones(module, N):

    return nn.ModuleList([deepcopy(module) for i in range(N)])
