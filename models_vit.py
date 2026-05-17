from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer

from taming.models.vqgan import VQModel
from omegaconf import OmegaConf
import numpy as np
import matplotlib.pyplot as plt
from util.pos_embed import get_2d_sincos_pos_embed
import math

class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, vocab_size, hidden_size, max_position_embeddings, dropout=0.1):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(max_position_embeddings).expand((1, -1)))

        torch.nn.init.normal_(self.word_embeddings.weight, std=.02)
        torch.nn.init.normal_(self.position_embeddings.weight, std=.02)

    def forward(
        self, input_ids
    ):
        input_shape = input_ids.size()

        seq_length = input_shape[1]

        position_ids = self.position_ids[:, :seq_length]

        inputs_embeds = self.word_embeddings(input_ids)

        position_embeddings = self.position_embeddings(position_ids)
        embeddings = inputs_embeds + position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class VisionTransformerMage(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, vqgan_ckpt_path='vqgan_jax.ckpt', **kwargs):
        super(VisionTransformerMage, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

        # --------------------------------------------------------------------------
        # VQGAN specifics
        config = OmegaConf.load('config/vqgan.yaml').model
        self.vqgan = VQModel(ddconfig=config.params.ddconfig,
                             n_embed=config.params.n_embed,
                             embed_dim=config.params.embed_dim,
                             ckpt_path=vqgan_ckpt_path)
        for param in self.vqgan.parameters():
            param.requires_grad = False

        codebook_size = config.params.n_embed
        vocab_size = codebook_size + 1000 + 1  # 1024 codebook size, 1000 classes, 1 for mask token.
        self.fake_class_label = codebook_size + 1100 - 1024
        self.mask_token_label = vocab_size - 1
        self.token_emb = BertEmbeddings(vocab_size=vocab_size,
                                        hidden_size=kwargs['embed_dim'],
                                        max_position_embeddings=256 + 1,
                                        dropout=0.1)


    def save_final_layer_weights(self, save_path_weights, save_path_plot):
        """Save the weights of the final layer and also save a plot of the weights."""
        
        # Get weights from the final layer
        if self.global_pool:
            weights = self.head[1].weight.detach().cpu().numpy()
        else:
            weights = self.norm.weight.detach().cpu().numpy()

        # Save the weights as a .npy file
        np.save(save_path_weights, weights)
        print(f"Final layer weights saved to {save_path_weights}")

        print(weights.shape)
        





    def forward_features(self, x):
        # tokenization
        with torch.no_grad():
            z_q, _, token_tuple = self.vqgan.encode(x)

        _, _, token_indices = token_tuple
        token_indices = token_indices.reshape(z_q.size(0), -1)

        # concate class token
        token_indices = torch.cat(
            [torch.zeros(token_indices.size(0), 1).cuda(device=token_indices.device), token_indices], dim=1)
        token_indices[:, 0] = self.fake_class_label
        token_indices = token_indices.long()
        # bert embedding
        x = self.token_emb(token_indices)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
            # x = self.fc_norm(x)
            # outcome = x[:, 1:, :].mean(dim=1)  # global pool without cls token
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome



class VisionTransformerLease(timm.models.vision_transformer.VisionTransformer):
    """
    Vision Transformer (inference) using VQGAN tokens, no mask/drop.
    - Uses [CLS | tokens], with BertEmbeddings for token ids.
    - Adds 2D positional embeddings: zeros for CLS row, 2D grid positions for tokens.
    - Pools to a single feature vector (CLS or global average of patch tokens).
    """
    def __init__(
        self,
        global_pool: bool = False,
        vqgan_ckpt_path: str = 'vqgan_jax.ckpt',
        keep_tokens: int = 256,  
        **kwargs
    ):
        """
        Args:
            global_pool: if True, mean-pool patch tokens then fc_norm (like your previous class).
            vqgan_ckpt_path: path to VQGAN checkpoint.
            keep_tokens: 256 uses all tokens; 128 keeps a fixed subset in original grid order.
                         Set to 128 if your training used 128 kept tokens to avoid a length shift.
            **kwargs: passed to timm VisionTransformer (must include 'embed_dim' and 'norm_layer').
        """
        super().__init__(**kwargs)

        assert keep_tokens in (128, 256), "keep_tokens must be 128 or 256"
        self.keep_tokens = keep_tokens

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)
            # remove the original norm; we'll apply fc_norm after pooling
            del self.norm  # noqa

        # -----------------------------
        # VQGAN: freeze for inference
        config = OmegaConf.load('config/vqgan.yaml').model
        self.vqgan = VQModel(ddconfig=config.params.ddconfig,
                             n_embed=config.params.n_embed,
                             embed_dim=config.params.embed_dim,
                             ckpt_path=vqgan_ckpt_path)
        for p in self.vqgan.parameters():
            p.requires_grad = False

        # -----------------------------
        # Token embedding setup
        codebook_size = int(config.params.n_embed)
        vocab_size = 17030  
        # These must match how you trained:
        self.fake_class_label = 17030 - 2
        self.mask_token_label = vocab_size - 1

        embed_dim = kwargs['embed_dim']
        self.token_emb = BertEmbeddings(
            vocab_size=vocab_size,
            hidden_size=embed_dim,
            max_position_embeddings=256 + 1,  
            dropout=0.1
        )

        # -----------------------------
        seq_len_expected = 256
        self.pos2d = nn.Parameter(torch.zeros(1, seq_len_expected, embed_dim), requires_grad=False)
        pos2d = get_2d_sincos_pos_embed(embed_dim, int(seq_len_expected ** 0.5), cls_token=False)  # (256, E)
        self.pos2d.data.copy_(torch.from_numpy(pos2d).float().unsqueeze(0))



    # ---- utility ----
    def _gather_keep_indices(self, L: int, device: torch.device, batch: int) -> torch.Tensor:
        """
        Deterministic keep indices in original grid order.
        Strategy: take the last K contiguous indices to mimic a fixed crop; you can switch to [:K] if preferred.
        """
        K = self.keep_tokens
        if K == L:
            return torch.arange(L, device=device).unsqueeze(0).expand(batch, -1)
        # choose a consistent slice; adjust to your training convention if needed
        keep = torch.arange(L - K, L, device=device)
        return keep.unsqueeze(0).expand(batch, -1)

    def save_final_layer_weights(self, save_path_weights, save_path_plot=None):
        """Save the weights of the final layer (same behavior as your original)."""
        if self.global_pool:
            weights = self.head[1].weight.detach().cpu().numpy()
        else:
            weights = self.norm.weight.detach().cpu().numpy()
        np.save(save_path_weights, weights)
        print(f"Final layer weights saved to {save_path_weights}")
        print(weights.shape)

    # ---- main forward ----

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inference path:
          1) VQGAN encode -> (B, 256) token ids
          2) Build [CLS | tokens_kept] ids
          3) Token embeddings + 2D pos enc (zeros for CLS)
          4) Transformer blocks (+ norm)
          5) Pool to a single (B, E) feature vector (CLS or global mean over patches)
        """

        # 1) VQGAN tokenization
        with torch.no_grad():
            z_q, _, token_tuple = self.vqgan.encode(x)

        _, _, token_indices = token_tuple  
        bsz = z_q.size(0)
        dev = token_indices.device

        vq_idx = token_indices.reshape(bsz, -1).long()  # (B, 256)
        L = vq_idx.size(1)
        assert L == 256, f"Expected 256 VQ tokens; got {L}"

        # 2) Choose tokens: either all 256 or a fixed 128 subset in grid order
        keep_idx_sorted = self._gather_keep_indices(L=L, device=dev, batch=bsz)  # (B, K)
        kept_vq = torch.take_along_dim(vq_idx, keep_idx_sorted, dim=1)           # (B, K)

        cls_col = torch.full((bsz, 1), self.fake_class_label, device=dev, dtype=torch.long)
        tokens_in = torch.cat([cls_col, kept_vq], dim=1)  # (B, 1+K)

        # 3) Embeddings + 2D positional encodings
        x_emb = self.token_emb(tokens_in)  # (B, 1+K, E)
        B, Np1, E = x_emb.shape  # Np1 = 1 + K

        # gather 2D positions to match the kept indices (no subsetting if K=256)
        pos2d_full = self.pos2d.expand(B, -1, -1)                                          # (B,256,E)
        pos2d_kept = torch.take_along_dim(pos2d_full, keep_idx_sorted.unsqueeze(-1).expand(-1, -1, E), dim=1)  # (B,K,E)
        pos2d_enc = torch.cat([torch.zeros(B, 1, E, device=dev), pos2d_kept], dim=1)       # (B,1+K,E)

        x = x_emb + pos2d_enc

        # 4) Transformer encoder
        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            # mean-pool patch tokens only, then fc_norm
            x = x[:, 1:, :].mean(dim=1)    # (B, E)
            feat = self.fc_norm(x)
        else:
            # apply ViT's final norm then take CLS
            x = self.norm(x)               # (B, 1+K, E)
            feat = x[:, 0]                 # (B, E)

        return feat








def vit_tiny_patch16(**kwargs):
    model = VisionTransformerMage(
        patch_size=16, embed_dim=192, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model



def vit_base_patch16(**kwargs):
    model = VisionTransformerMage(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base_patch16_lease(**kwargs):
    model = VisionTransformerLease(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformerMage(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16_lease(**kwargs):
    model = VisionTransformerLease(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
