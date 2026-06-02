from functools import partial

import torch
import torch.nn as nn

import torch.nn.functional as F

from timm.models.vision_transformer import PatchEmbed, DropPath, Mlp

from util.pos_embed import get_2d_sincos_pos_embed


from omegaconf import OmegaConf
import numpy as np
import scipy.stats as stats
import torch.distributed as dist

import time
import math

from torchvision.utils import save_image

from typing import Optional

import csv
import os


try:
    from flash_attn import flash_attn_qkvpacked_func
    print("Successfully imported flash_attn_qkvpacked_func")
except ImportError:
    flash_attn_qkvpacked_func = None
    print("flash_attn_qkvpacked_func is not available, proceeding without it.")


try:
    from taming.models.vqgan import VQModel
    print("Successfully imported VQGAN")
except ImportError:
    print("VQGAN is not available, proceeding without it.")


class GatherLayer(torch.autograd.Function):
    """
    Gathers tensors from all process and supports backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        if dist.is_available() and dist.is_initialized():
            output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
            dist.all_gather(output, x)
        else:
            output = [x]
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        if dist.is_available() and dist.is_initialized():
            all_gradients = torch.stack(grads)
            dist.all_reduce(all_gradients)
            grad_out = all_gradients[dist.get_rank()]
        else:
            grad_out = grads[0]
        return grad_out


def gather(X, dim=0):
    """Gathers tensors from all processes, supporting backward propagation."""
    return torch.cat(GatherLayer.apply(X), dim=dim)




class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        with torch.cuda.amp.autocast(enabled=False):
            attn = (q.float() @ k.float().transpose(-2, -1)) * self.scale

        attn = attn - torch.max(attn, dim=-1, keepdim=True)[0]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        if return_attention:
            _, attn = self.attn(self.norm1(x))
            return attn
        else:
            y, _ = self.attn(self.norm1(x))
            x = x + self.drop_path(y)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class LabelSmoothingCrossEntropy(nn.Module):
    """ NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss


class TopKSmoothedLoss(nn.Module):
    def __init__(self, codebook_size, smoothing=0.1, topk_weight=0.3, k=5):
        super().__init__()
        self.ce_loss = LabelSmoothingCrossEntropy(smoothing=smoothing)
        self.codebook_size = codebook_size
        self.topk_weight = topk_weight
        self.k = k

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # x: (B*T, vocab), target: (B*T,)
        base_loss = self.ce_loss(x, target)

        # Get top-k indices
        topk_preds = torch.topk(x, self.k, dim=-1).indices  # (B*T, k)
        target_expanded = target.unsqueeze(1)  # (B*T, 1)
        match_topk = (topk_preds == target_expanded).any(dim=1).float()  # (B*T,)

        # Reward if in top-k (lowers loss)
        # We'll reduce loss for correct top-k predictions
        reward = match_topk * 0.5  # You can tune this scalar
        topk_loss = (1.0 - reward) * base_loss  # Reduce penalty if it's top-k

        # Combine losses
        loss = (1 - self.topk_weight) * base_loss + self.topk_weight * topk_loss
        return loss



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


class MlmLayer(nn.Module):

    def __init__(self, feat_emb_dim, word_emb_dim, vocab_size):
        super().__init__()
        self.fc = nn.Linear(feat_emb_dim, word_emb_dim)
        self.gelu = nn.GELU()
        self.ln = nn.LayerNorm(word_emb_dim)
        self.bias = nn.Parameter(torch.zeros(1, 1, vocab_size))

    def forward(self, x, word_embeddings):
        mlm_hidden = self.fc(x)
        mlm_hidden = self.gelu(mlm_hidden)
        mlm_hidden = self.ln(mlm_hidden)
        word_embeddings = word_embeddings.transpose(0, 1)
        logits = torch.matmul(mlm_hidden, word_embeddings)
        logits = logits + self.bias
        return logits




class Projector(nn.Module):
    def __init__(self, input_dim, proj_hidden_dim, proj_output_dim):
        super(Projector, self).__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim)
        )
    
    def forward(self, x):
        return self.projector(x)



   
 
class FlashAttentionLayer(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.attn_drop_prob = attn_drop
        
        #self.attn_layer = FlashAttention(attention_dropout=self.attn_drop_prob if self.training else 0.0, softmax_scale=self.scale)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, causal=False):
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        if flash_attn_qkvpacked_func is not None:
            attn_output = flash_attn_qkvpacked_func(
                    qkv, 
                    dropout_p=self.attn_drop_prob if self.training else 0.0, 
                    softmax_scale=self.scale, 
                    causal=causal
                )
        else:
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop_prob if self.training else 0.0,
                is_causal=causal,
                scale=self.scale,
            )
            attn_output = attn_output.permute(0, 2, 1, 3)
        
        x = attn_output.reshape(B, N, C) 
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    
    
class FlashBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = FlashAttentionLayer(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        if return_attention:
            raise ValueError("We do not support returning attention maps.")
        else:
            y = self.attn(self.norm1(x))
            x = x + self.drop_path(y)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    








class SorcenViT(nn.Module):
    """ Sorcen Masked Autoencoder with VisionTransformer backbone"""
    def __init__(self, img_size=256, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 mask_ratio_min=0.5, mask_ratio_max=1.0, mask_ratio_mu=0.55, mask_ratio_std=0.25,
                 vqgan_ckpt_path='vqgan_jax_strongaug.ckpt', dino_codebook_path=''):
        super().__init__()

        # --------------------------------------------------------------------------
        # VQGAN specifics from MAGE
        config = OmegaConf.load('config/vqgan.yaml').model

        self.codebook_size = config.params.n_embed
        vocab_size = self.codebook_size + 1000 + 1 
        self.fake_class_label = self.codebook_size + 1100 - 1024
        self.mask_token_label = vocab_size - 1
        self.token_emb = BertEmbeddings(vocab_size=vocab_size,
                                        hidden_size=embed_dim,
                                        max_position_embeddings=256+1,
                                        dropout=0.1)

        # Sorcen Masking Ratio, hyperparams from MAGE 
        self.mask_ratio_min = mask_ratio_min
        self.mask_ratio_generator = stats.truncnorm((mask_ratio_min - mask_ratio_mu) / mask_ratio_std,
                                                    (mask_ratio_max - mask_ratio_mu) / mask_ratio_std,
                                                    loc=mask_ratio_mu, scale=mask_ratio_std)

        self.mask_ratio_generator_nn = stats.truncnorm((mask_ratio_min - 0.85) / mask_ratio_std,
                                                    (mask_ratio_max - 0.85) / mask_ratio_std,
                                                    loc=0.85, scale=mask_ratio_std)                                                    

        # --------------------------------------------------------------------------
        # Sorcen Encoder specifics
        dropout_rate = 0.1
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False) 

        self.blocks = nn.ModuleList([
            FlashBlock(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer,
                  drop=dropout_rate, attn_drop=dropout_rate)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # Sorcen Teacher Encoder
        self.beta = 0.996  # EMA decay factor
        self.blocks_momentum = nn.ModuleList([
            FlashBlock(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer,
                  drop=dropout_rate, attn_drop=dropout_rate)
            for i in range(depth)])


        # --------------------------------------------------------------------------
        # Sorcen Decoder specifics from MAGE
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.pad_with_cls_token = True

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.decoder_pos_embed_learned = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim))  # learnable pos embedding

        self.decoder_blocks = nn.ModuleList([
            FlashBlock(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer,
                  drop=dropout_rate, attn_drop=dropout_rate)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------
        # MlmLayer
        self.mlm_layer = MlmLayer(feat_emb_dim=decoder_embed_dim, word_emb_dim=embed_dim, vocab_size=vocab_size)

        self.norm_pix_loss = norm_pix_loss

        self.criterion = LabelSmoothingCrossEntropy(smoothing=0.1)



        # ---------------------------------------------------------------------------
        # Sorcen Simple Projector Head
        self.projector = Projector(embed_dim, 4096, 512)
        self.momentum_projector = Projector(embed_dim, 4096, 512)

        self.predictor = Projector(512, 2048, 512)

        self.initialize_weights()
  
        # ---------------------------------------------------------------------------
        # Simple online classifier for debugging and experiment tracking (Optional)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1000)  # num_classes
        )        
        

        # -----------------------------------------------------------------------------------


    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.decoder_pos_embed_learned, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)



    def forward_encoder(self, vq_idx, labels):
        
        start_time = time.time()

        token_indices = vq_idx
        
        token_indices = token_indices.reshape(token_indices.size(0), -1)

        gt_indices = token_indices.clone().detach().long()

        # masking
        bsz, seq_len = token_indices.size()
        mask_ratio_min = self.mask_ratio_min
        mask_rate = self.mask_ratio_generator.rvs(1)[0]


        num_dropped_tokens = int(np.ceil(seq_len * mask_ratio_min))
        num_masked_tokens = int(np.ceil(seq_len * mask_rate))




        # it is possible that two elements of the noise is the same, so do a while loop to avoid it
        while True:
            noise = torch.rand(bsz, seq_len, device=vq_idx.device)  # noise in [0, 1]
            sorted_noise, _ = torch.sort(noise, dim=1)  # ascend: small is remove, large is keep
            cutoff_drop = sorted_noise[:, num_dropped_tokens-1:num_dropped_tokens]
            cutoff_mask = sorted_noise[:, num_masked_tokens-1:num_masked_tokens]


            token_drop_mask = (noise <= cutoff_drop).float()
            token_all_mask = (noise <= cutoff_mask).float()

            if token_drop_mask.sum() == bsz*num_dropped_tokens and token_all_mask.sum() == bsz*num_masked_tokens:
                break
            else:
                print("Rerandom the noise!")

    
        token_indices[token_all_mask.nonzero(as_tuple=True)] = self.mask_token_label


        # concate class token
        token_indices = torch.cat([torch.zeros(token_indices.size(0), 1).cuda(device=token_indices.device), token_indices], dim=1)
        token_indices[:, 0] = self.fake_class_label
        token_drop_mask = torch.cat([torch.zeros(token_indices.size(0), 1).cuda(), token_drop_mask], dim=1)
        token_all_mask = torch.cat([torch.zeros(token_indices.size(0), 1).cuda(), token_all_mask], dim=1)
        token_indices = token_indices.long()


        # bert embedding
        input_embeddings = self.token_emb(token_indices)


        bsz, seq_len, emb_dim = input_embeddings.shape

        # dropping
        token_keep_mask = 1 - token_drop_mask
   

        input_embeddings_after_drop = input_embeddings[token_keep_mask.nonzero(as_tuple=True)].reshape(bsz, -1, emb_dim)



        # apply Transformer blocks
        x = input_embeddings_after_drop
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)


        return x, 0, gt_indices, token_drop_mask, token_all_mask




    def nnclr_loss_func(self, nn: torch.Tensor, p: torch.Tensor, temperature: float = 0.3): 

        # From Solo-learn: 

        nn = F.normalize(nn, dim=-1)
        p = F.normalize(p, dim=-1)
        p = gather(p)

        logits = nn @ p.T / temperature

        rank = dist.get_rank()
        n = nn.size(0)
        labels = torch.arange(n * rank, n * (rank + 1), device=p.device)
        loss = F.cross_entropy(logits, labels)
            
        return loss

    def uniformity_loss(self, features):

        # gather across devices
        features = gather(features)

        features = torch.nn.functional.normalize(features)
        sim = features @ features.T 
        loss = sim.pow(2).mean()
        return loss * 0.01

    def forward_decoder(self, x, token_drop_mask, token_all_mask):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        if self.pad_with_cls_token:
            mask_tokens = x[:, 0:1].repeat(1, token_all_mask.shape[1], 1)
        else:
            mask_tokens = self.mask_token.repeat(token_all_mask.shape[0], token_all_mask.shape[1], 1)

        # put undropped tokens into original sequence
        x_after_pad = mask_tokens.clone()
        x_after_pad[(1 - token_drop_mask).nonzero(as_tuple=True)] = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
        # set undropped but masked positions with mask
        x_after_pad = torch.where(token_all_mask.unsqueeze(-1).bool(), mask_tokens, x_after_pad)

        # add pos embed
        x = x_after_pad + self.decoder_pos_embed_learned

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)

        x = self.decoder_norm(x)

        word_embeddings = self.token_emb.word_embeddings.weight.data.detach()
        x = self.mlm_layer(x, word_embeddings)


        return x




    def top_k_categorical_exclude_max(self, logits, k=15):
        # Find the maximum logit along the last dimension and set it to -inf
        max_values, _ = torch.max(logits, dim=-1, keepdim=True)
        logits_without_max = logits.masked_fill(logits == max_values, float('-inf'))

        # Get the top-k logits (excluding the previously masked max logit)
        top_k_values, top_k_indices = torch.topk(logits_without_max, k, dim=-1)

        # Initialize a mask with -inf for all logits
        mask = torch.full_like(logits, float('-inf'))

        # Scatter the top-k values into the mask, retaining their original positions
        mask.scatter_(-1, top_k_indices, top_k_values)

        # Create a new categorical distribution using the masked logits
        distribution = torch.distributions.categorical.Categorical(logits=mask)
        return distribution


    def mask_random_square(self, reconstructed_sequence, mask_token_label, crop_size=8):
        
        # Jittered Spatial Masking computation

        bsz, _, _ = reconstructed_sequence.shape
        
        # Ensure crop_size is smaller than the image dimensions
        if crop_size > reconstructed_sequence.shape[1]:
            raise ValueError(f"crop_size must be smaller than the image dimension size (16).")

        # Randomly pick the top-left corner coordinates for the crop
        top_left_x = torch.randint(0, reconstructed_sequence.shape[1] - crop_size + 1, (bsz,))  
        top_left_y = torch.randint(0, reconstructed_sequence.shape[2] - crop_size + 1, (bsz,))  

        # Create a mask tensor filled with the mask token label
        mask = torch.full_like(reconstructed_sequence, mask_token_label)
        
        # For each image in the batch, mask the selected square region
        for i in range(bsz):
            mask[i, top_left_x[i]:top_left_x[i] + crop_size, top_left_y[i]:top_left_y[i] + crop_size] = -1000

        # Replace the selected region in the image with the mask token label, otherwise keep original
        masked_sequence = torch.where(mask == -1000, reconstructed_sequence, mask_token_label)
        
        return masked_sequence


    def forward_strange(self, gt_indices, logits, mask, epoch):

        # Get the batch size and sequence length
        bsz, seq_len = gt_indices.size()
        
        predicted_indices = logits[:, 1:, :self.codebook_size] 


        sample_dist = self.top_k_categorical_exclude_max(predicted_indices, k=15) 
        sampled_ids = sample_dist.sample()
        mask_expanded = mask[:, 1:].bool()
        reconstructed_sequence = sampled_ids 

        if epoch <= 10: # We warm-up the Decoder, Echos are only used after this warm-up is completed
            reconstructed_sequence = gt_indices.reshape(bsz, 16, 16) 
        else:
            reconstructed_sequence = reconstructed_sequence.reshape(bsz, 16, 16)
        
        reconstructed_sequence = self.mask_random_square(reconstructed_sequence, self.mask_token_label, torch.randint(8, 17, (1,)).item())

        augmented_reconstructed_sequence = reconstructed_sequence.reshape(bsz, 256) 

        # concate class token
        reconstructed_sequence = torch.cat([torch.zeros(augmented_reconstructed_sequence.size(0), 1).cuda(device=augmented_reconstructed_sequence.device), augmented_reconstructed_sequence], dim=1) #
        reconstructed_sequence[:, 0] = self.fake_class_label
        reconstructed_sequence = reconstructed_sequence.long()


        with torch.no_grad(): #No gradients are computed, this branch is updated using EMA
            input_embeddings = self.token_emb(reconstructed_sequence) 

            x = input_embeddings
            for blk in self.blocks_momentum:
                x = blk(x)
            x = self.norm(x)


        return x


    def forward_loss(self, gt_indices, logits, mask):
        bsz, seq_len = gt_indices.size()
        # logits and mask are with seq_len+1 but gt_indices is with seq_len
        loss = self.criterion(logits[:, 1:, :self.codebook_size].reshape(bsz*seq_len, -1), gt_indices.reshape(bsz*seq_len))
        loss = loss.reshape(bsz, seq_len)
        loss = (loss * mask[:, 1:]).sum() / mask[:, 1:].sum()  # mean loss on removed patches
        return loss


    def update_ema(self, epoch):
        """
        Update EMA parameters for all layers.
        """
        beta = self.compute_beta(epoch)
        
        # Update the EMA for each block of Teacher encoder and Proyector Head
        for block, ema_block in zip(self.blocks, self.blocks_momentum):
            for param, ema_param in zip(block.parameters(), ema_block.parameters()):
                ema_param.data = beta * ema_param.data + (1 - beta) * param.data

        with torch.no_grad():
            for param, ema_param in zip(self.projector.parameters(), self.momentum_projector.parameters()):
                ema_param.data = beta * ema_param.data + (1 - beta) * param.data

    def compute_beta(self, epoch):
        """
        Compute the EMA decay rate beta according to the given cosine schedule.
        """
        return 1 - (1 - self.beta) * (0.5 * (math.cos(math.pi * epoch / 1600) + 1)) # For smaller trainings, change this 1600 value
    

        

    def forward(self, vq_idx, labels, epoch):
        
        # Update the Teacher Encoder
        self.update_ema(epoch)
        # Forward pass on the Student Encoder
        latent, _, gt_indices, token_drop_mask, token_all_mask = self.forward_encoder(vq_idx, labels)
        # Forward pass on the Decoder
        logits = self.forward_decoder(latent, token_drop_mask, token_all_mask)

        # Compute Uniformity term of the Contrastive loss
        uni_loss = self.uniformity_loss(latent.mean(1))

        # Compute the Reconstruction loss
        rec_loss = self.forward_loss(gt_indices, logits, token_all_mask) + uni_loss

        # Compute the Echos and Forward pass on the Teacher Encoder
        nn_latent = self.forward_strange(gt_indices, logits, token_all_mask, epoch)

        # Project Echo latents into the Contrastive Space
        proj_nn_latent = self.momentum_projector(nn_latent[:, 1:, :].mean(1))

        # Project Anchor latens into the Contrastive Space and apply the Predictor
        decoded_latent = self.predictor(self.projector(latent[:, 1:, :].mean(1))) 

        # Compute main term of the Contrastive loss using the Anchors and Echos
        nn_loss = self.nnclr_loss_func(proj_nn_latent, decoded_latent, temperature=0.3) * 0.1  

        # On-the-fly classifier training
        if labels is not None:
            classifier_input = latent.detach() #IMPORTANT: One should always detatch the latents
            classifier_logits = self.classifier(classifier_input.mean(1))  # Mean pooling
            classifier_loss = F.cross_entropy(classifier_logits, labels.squeeze())
            accuracy = (classifier_logits.argmax(dim=1) == labels.squeeze()).float().mean()
        else:
            classifier_loss = 0
            accuracy = 0

        total_loss = rec_loss + nn_loss + classifier_loss

        return total_loss, rec_loss, nn_loss, token_all_mask, accuracy, classifier_loss, latent



class LEASE_DecLoss_ViT(nn.Module):
    def __init__(self,
                 img_size=256, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 mask_ratio_min=0.5, mask_ratio_max=1.0, mask_ratio_mu=0.55, mask_ratio_std=0.25,
                 vqgan_ckpt_path='vqgan_jax_strongaug.ckpt',
                 
                 dino_codebook_path="./km_16k.npy",      # path to npy [K,D]
                 info_nce_tau=0.1, #0.1 for IN1k
                 info_nce_weight=0.1,
                 info_nce_full_softmax=True,   # kept for compatibility
                 info_nce_chunk_k=4096):
        super().__init__()

        # --------------------------------------------------------------------------
        # VQ token vocabulary
        config = OmegaConf.load('config/vqgan.yaml').model
        seq_len_expected = 256                     # 16x16 grid
        self.vocab_size = 17030                    
        self.vqgan_vocab_size = 1024               # VQ codes are [0..1023]
        self.num_specials = 2
        self.code_vocab_size = self.vocab_size - self.num_specials
        self.fake_class_label = self.vocab_size - 2
        self.mask_token_label = self.vocab_size - 1

        # Token embedding (shared MLM head weights)
        self.token_emb = BertEmbeddings(
            vocab_size=self.vocab_size,
            hidden_size=embed_dim,
            max_position_embeddings=seq_len_expected + 1,   # +1 for CLS → 257
            dropout=0.1
        )

        # Shared frozen 2D sin-cos pos embed (16x16)
        self.pos2d = nn.Parameter(torch.zeros(1, seq_len_expected, embed_dim), requires_grad=False)
        pos2d = get_2d_sincos_pos_embed(embed_dim, int(seq_len_expected ** 0.5), cls_token=False)  # (256, E)
        self.pos2d.data.copy_(torch.from_numpy(pos2d).float().unsqueeze(0))

        # Masking sampler (same as before)
        self.mask_ratio_min = mask_ratio_min
        self.mask_ratio_generator = stats.truncnorm(
            (mask_ratio_min - mask_ratio_mu) / mask_ratio_std,
            (mask_ratio_max - mask_ratio_mu) / mask_ratio_std,
            loc=mask_ratio_mu, scale=mask_ratio_std
        )

        # --------------------------------------------------------------------------
        # Encoder 
        dropout_rate = 0.1
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = nn.ModuleList([
            FlashBlock(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer,
                       drop=dropout_rate, attn_drop=dropout_rate)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # --------------------------------------------------------------------------
        # VQ decoder 
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.pad_with_cls_token = True
        
        self.decoder_pos_embed_learned = nn.Parameter(
            torch.zeros(1, seq_len_expected + 128 + 1, decoder_embed_dim)
        )
        self.decoder_blocks = nn.ModuleList([
            FlashBlock(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer,
                       drop=dropout_rate, attn_drop=dropout_rate)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = norm_layer(decoder_embed_dim)

        # MLM head + criterion 
        self.mlm_layer = MlmLayer(feat_emb_dim=decoder_embed_dim, word_emb_dim=embed_dim, vocab_size=self.vocab_size)
        self.norm_pix_loss = norm_pix_loss
        self.criterion = LabelSmoothingCrossEntropy(smoothing=0.1)

        # --------------------------------------------------------------------------
        # Disc centroids InfoNCE branch
        assert dino_codebook_path is not None, "Provide codebook_path to use InfoNCE distillation."
        C = np.load(dino_codebook_path).astype(np.float32)  # [K, D]
        C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)      # L2 row-norm
        self.register_buffer("dino_codebook", torch.from_numpy(C))       # [K, D] frozen
        self.dino_dim = C.shape[1]
        # project encoder dim -> DINO centroid dim
        self.proj_to_dino = nn.Sequential(
            nn.Linear(embed_dim, self.dino_dim, bias=False),
        )
        self.info_nce_tau = info_nce_tau
        self.info_nce_weight = info_nce_weight
        self.info_nce_full_softmax = info_nce_full_softmax
        self.info_nce_chunk_k = info_nce_chunk_k

        # Multi-positive target knobs
        self.use_neighbor_targets   = True
        self.neighbor_topk          = 5          # include 5 neighbors in addition to the true centroid
        self.neighbor_teacher_tau   = 0.1        # temperature to compute soft weights over neighbors

        # --------------------------------------------------------------------------
        # Decoder → DINO 
        self.dec_to_dino = nn.Linear(decoder_embed_dim, self.dino_dim, bias=False)
        self.masked_dino_weight = 0.1
        self.masked_dino_tau    = self.info_nce_tau

        self.initialize_weights()

        # Optional online classifier
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1000)  # num_classes
        )

        self.debug = {}

    # =========================
    # Initialization Utilities
    # =========================
    def initialize_weights(self):
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.decoder_pos_embed_learned, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # =========================
    # Helpers
    # =========================
    def _pool_stream_vq(self, x_with_cls, kept_mask=None):
        feats = x_with_cls[:, 1:129, :]  # (B,128,E)
        if kept_mask is None:
            return feats.mean(dim=1)
        unmasked = (~kept_mask).float()
        denom = unmasked.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (feats * unmasked.unsqueeze(-1)).sum(dim=1) / denom

    # =========================
    # Encoder / Decoder
    # =========================


    @torch.compiler.disable
    def _sample_mask_params(self, bsz, L, dev):
        """Non-compilable: scipy truncnorm. Returns mask_rate (float), num_masked (int)."""
        mask_rate  = float(self.mask_ratio_generator.rvs(1)[0])
        num_masked = int(np.ceil(L * mask_rate))
        return mask_rate, num_masked

    @torch.compiler.disable
    def forward_encoder(self, vq_idx, labels):
        bsz = vq_idx.size(0)
        dev = vq_idx.device

        vq_idx = vq_idx.reshape(bsz, 256).long()   # (B,256)
        L = 256
        K_keep = 128

        # ---- sample permutation → drop/keep & mask ----
        perm = torch.rand(bsz, L, device=dev).argsort(dim=1)
        drop_idx = perm[:, :L - K_keep]                 # drop these at the embedding stage
        keep_idx = perm[:, L - K_keep:]                 # keep these

        mask_rate, num_masked = self._sample_mask_params(bsz, L, dev)
        mask_idx   = perm[:, :num_masked]

        masked_256  = torch.zeros(bsz, L, device=dev, dtype=torch.bool)
        dropped_256 = torch.zeros(bsz, L, device=dev, dtype=torch.bool)
        masked_256.scatter_(1, mask_idx, True)
        dropped_256.scatter_(1, drop_idx, True)

        # ---- apply mask token to INPUT IDs only ----
        vq_masked = vq_idx.clone()
        vq_masked[masked_256] = self.mask_token_label

        # ---- EMBED FULL [CLS | 256] first (BertEmbeddings sees all 257 positions) ----
        cls_col     = torch.full((bsz, 1), self.fake_class_label, device=dev, dtype=torch.long)
        tokens_full = torch.cat([cls_col, vq_masked], dim=1)              # (B, 257)
        x_full = self.token_emb(tokens_full)                              # (B, 257, E)

        # ---- NOW drop to 128 on the EMBEDDINGS ----
        keep_idx_sorted = torch.sort(keep_idx, dim=1).values              # (B,128) in original grid order

        # gather kept patch embeddings from the full embedded stream
        patch_keep_emb = torch.take_along_dim(
            x_full[:, 1:, :],                                             # (B,256,E)
            keep_idx_sorted.unsqueeze(-1).expand(-1, -1, x_full.size(-1)),
            dim=1
        )                                                                 # (B,128,E)

        # rebuild [CLS | kept] on EMBEDDINGS
        x = torch.cat([x_full[:, 0:1, :], patch_keep_emb], dim=1)         # (B, 1+128, E)

        # ---- add fixed 2D pos for the kept spatial positions (CLS row zeros) ----
        B, Np1, E = x.shape
        pos2d_kept = torch.take_along_dim(
            self.pos2d.expand(B, -1, -1),                                 # (B,256,E)
            keep_idx_sorted.unsqueeze(-1).expand(-1, -1, E),
            dim=1
        )                                                                 # (B,128,E)
        pos2d_enc = torch.cat([torch.zeros(B, 1, E, device=dev), pos2d_kept], dim=1)
        x = x + pos2d_enc

        # ---- transformer + final norm ----
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)                                                  # (B, 129, E)

        gt_vq256 = vq_idx.detach().long()
        return x, gt_vq256, keep_idx_sorted, masked_256, dropped_256



    def forward_decoder_vq(self, x_with_cls, keep_idx, masked_256, dropped_256, return_feats: bool = False):
        """
        VQ decoder only. Rebuild a 256-slot canvas in original order and decode.
        If return_feats=True, also returns decoder features after decoder_norm: (B,257,Dd).
        """
        B, Np1, E = x_with_cls.shape
        z = self.decoder_embed(x_with_cls)     # (B,129,Dd)
        cls_row = z[:, 0:1, :]                 # (B,1,Dd)
        vq_z    = z[:, 1:129, :]               # (B,128,Dd)

        # Build canvas in original order: fill only visible kept; others are placeholders
        vq_canvas = cls_row.repeat(1, 256, 1)
        kept_mask_vq = torch.take_along_dim(masked_256, keep_idx, dim=1)  # True=masked among kept
        kept_unmasked = ~kept_mask_vq
        place_idx = keep_idx.masked_fill(~kept_unmasked, 0)
        vals = vq_z.masked_fill(~kept_unmasked.unsqueeze(-1), 0)
        vq_canvas.scatter_add_(
            dim=1,
            index=place_idx.unsqueeze(-1).expand(-1, -1, vq_canvas.size(-1)),
            src=vals
        )

        
        dec_in = torch.cat([cls_row, vq_canvas], dim=1)      # (B,257,Dd)
        pos = self.decoder_pos_embed_learned[:, :257, :]
        dec_in = dec_in + pos

        for blk in self.decoder_blocks:
            dec_in = blk(dec_in)
        dec_in = self.decoder_norm(dec_in)                   # (B,257,Dd)

        with torch.no_grad():
            word_embeddings = self.token_emb.word_embeddings.weight.detach()
        logits = self.mlm_layer(dec_in, word_embeddings)     # (B,257,V)

        if return_feats:
            return logits, dec_in
        return logits

    # =========================
    # Losses
    # =========================
    def forward_loss_vq(self, gt_indices, logits, mask):
        assert gt_indices.dtype == torch.long
        bsz, seq_len = gt_indices.size()
        logits_code = logits[:, 1:1+seq_len, :self.vqgan_vocab_size]  # (B,256,1024)
        loss = self.criterion(
            logits_code.reshape(bsz * seq_len, -1),
            gt_indices.reshape(bsz * seq_len)
        ).reshape(bsz, seq_len)
        loss = (loss * mask[:, 1:1+seq_len]).sum() / (mask[:, 1:1+seq_len].sum() + 1e-6)
        return loss

    def info_nce_visible_full(self, z_vis, t_vis):
        # L2-normalize
        z = F.normalize(z_vis, dim=-1)
        C = self.dino_codebook
        if self.info_nce_full_softmax:
            logits = F.linear(z, C) / self.info_nce_tau   # [N,K]
            return F.cross_entropy(logits.float(), t_vis)
        else:
            N, E = z.shape
            K = C.shape[0]
            tau = self.info_nce_tau
            c_pos = C.index_select(0, t_vis)                    # [N,E]
            l_pos = (z * c_pos).sum(-1) / tau                   # [N]
            m = torch.full((N,), -float('inf'), device=z.device, dtype=z.dtype)
            s = torch.zeros((N,), device=z.device, dtype=z.dtype)
            for s_k in range(0, K, self.info_nce_chunk_k):
                e_k = min(K, s_k + self.info_nce_chunk_k)
                Ck = C[s_k:e_k]                                  # [k,E]
                logits_k = (z @ Ck.t()) / tau                    # [N,k]
                m_new = torch.maximum(m, logits_k.max(dim=1).values)
                s = s * torch.exp(m - m_new) + torch.exp(logits_k - m_new.unsqueeze(1)).sum(dim=1)
                m = m_new
            log_denom = m + torch.log(s + 1e-12)
            loss_vec = -(l_pos - log_denom)
            return loss_vec.mean().float()

    
    def info_nce_visible_neighbors(self, z_vis, t_vis, topk=None, teacher_tau=None, loss_tau=None):
        """
        Multi-positive InfoNCE using the correct centroid + its top-k nearest neighbors,
        with similarity-weighted soft targets.

        z_vis: [N, D] (projected to DINO dim; NOT normalized)
        t_vis: [N]    int labels in [0,K-1]
        """
        if topk is None:
            topk = self.neighbor_topk
        if teacher_tau is None:
            teacher_tau = self.neighbor_teacher_tau
        if loss_tau is None:
            loss_tau = self.info_nce_tau

        # normalize
        z = F.normalize(z_vis, dim=-1)                  # [N, D]
        C = F.normalize(self.dino_codebook, dim=-1)     # [K, D]
        N, D = z.shape
        K = C.shape[0]

        # student logits over full codebook
        logits = F.linear(z, C) / loss_tau              # [N, K]
        logp = F.log_softmax(logits, dim=1)             # [N, K]

        # teacher soft labels over (self + neighbors)
        with torch.no_grad():
            c_pos = C.index_select(0, t_vis)            # [N, D]
            sims  = F.linear(c_pos, C)                  # [N, K] cosine
            k_sel = min(topk + 1, K)                    # include "self" centroid
            sims_top, idx_top = sims.topk(k=k_sel, dim=1)         # [N, k_sel]
            weights = F.softmax(sims_top / max(teacher_tau, 1e-6), dim=1)  # [N, k_sel]; sum=1

        logp_sel = logp.gather(1, idx_top)              # [N, k_sel]
        loss = -(weights * logp_sel).sum(dim=1)
        return loss

    def masked_dino_ce(self, dec_feats_257, dino_idx_256, mask_full_257, tau=None):
        if tau is None:
            tau = self.masked_dino_tau

        supervise_mask = (mask_full_257[:, 1:] > 0)  # (B,256) bool

        dec_256 = dec_feats_257[:, 1:, :]                           # (B,256,Dd)
        z = self.dec_to_dino(dec_256)                                # (B,256,D)
        z = F.normalize(z, dim=-1)

        C = F.normalize(self.dino_codebook, dim=-1)                 # [K,D]
        t = dino_idx_256.long()                                      # (B,256)
        logits = F.linear(z, C) / tau                                # (B,256,K)

        B_ce, S_ce, K_ce = logits.shape
        logits_flat = logits.reshape(B_ce * S_ce, K_ce)
        targets_flat = t.reshape(B_ce * S_ce)
        mask_flat = supervise_mask.reshape(B_ce * S_ce)
        ignore_idx = -100
        targets_flat = torch.where(
            mask_flat,
            targets_flat,
            torch.full_like(targets_flat, ignore_idx)
        )
        return F.cross_entropy(logits_flat.float(), targets_flat, ignore_index=ignore_idx)

    # =========================
    # Forward
    # =========================
    def forward(self, vq_idx, dino_idx, labels, epoch):
        """
        vq_idx:   (B,256) VQ token ids in [0..1023]
        dino_idx: (B,256) DINO token ids in [0..K-1] (used for InfoNCE/CE teacher)
        labels:   (B,)    class labels (optional)
        """
        # ---- Encode (VQ-only) ----
        x_with_cls, gt_vq256, keep_idx, masked_256, dropped_256 = self.forward_encoder(vq_idx, labels)

        # ---- Decode once (reuse feats) + VQ MLM loss ----
        logits_vq, dec_feats_257 = self.forward_decoder_vq(
            x_with_cls, keep_idx, masked_256, dropped_256, return_feats=True
        )
        vq_loss_mask256 = (masked_256 | dropped_256).to(torch.float32)
        mask_full_vq = torch.cat(
            [torch.zeros(vq_loss_mask256.size(0), 1, device=vq_loss_mask256.device),
             vq_loss_mask256], dim=1
        )  # (B,257)
        rec_loss_vq = self.forward_loss_vq(gt_vq256, logits_vq, mask_full_vq)

        # ---- Decoder-based DINO CE on masked ∪ dropped tokens ----
        B = vq_idx.size(0)
        dino_idx_256 = dino_idx.reshape(B, 256)
        masked_dino_ce_loss = self.masked_dino_ce(
            dec_feats_257, dino_idx_256, mask_full_vq, tau=self.masked_dino_tau
        )

        # ---- InfoNCE on visible (kept & unmasked) patches ----
        vq_feats_kept = x_with_cls[:, 1:129, :]                         # (B,128,E)
        kept_mask = torch.take_along_dim(masked_256, keep_idx, dim=1)   # (B,128) True=masked among kept
        visible_mask = ~kept_mask                                       # (B,128)

        z_vis = self.proj_to_dino(vq_feats_kept)                         # (B,128,D)
        t_vis = torch.take_along_dim(dino_idx, keep_idx, dim=1).long()   # (B,128)

        B_vis, S_vis = z_vis.shape[:2]
        z_vis_flat = z_vis.reshape(B_vis * S_vis, -1)
        t_vis_flat = t_vis.reshape(B_vis * S_vis)
        mask_flat = visible_mask.reshape(B_vis * S_vis)

        if self.use_neighbor_targets and self.neighbor_topk > 0:
            t_vis_safe = torch.where(mask_flat, t_vis_flat, torch.zeros_like(t_vis_flat))
            raw_loss = self.info_nce_visible_neighbors(
                z_vis_flat, t_vis_safe,
                topk=self.neighbor_topk,
                teacher_tau=self.neighbor_teacher_tau,
                loss_tau=self.info_nce_tau
            )
            w = mask_flat.to(z_vis_flat.dtype)
            info_nce_loss = (raw_loss * w).sum() / (w.sum().clamp(min=1))
        else:
            z_norm = F.normalize(z_vis_flat, dim=-1)
            C_norm = F.normalize(self.dino_codebook, dim=-1)
            logits = F.linear(z_norm, C_norm) / self.info_nce_tau
            ignore_idx = -100
            t_masked = torch.where(mask_flat, t_vis_flat, torch.full_like(t_vis_flat, ignore_idx))
            info_nce_loss = F.cross_entropy(logits.float(), t_masked, ignore_index=ignore_idx)

        # ---- Optional online classifier ----
        if labels is not None:
            y = labels.view(-1)
            classifier_logits = self.classifier(x_with_cls[:, 1:, :].detach().mean(1)) # IMPORTANT: Always detatch latents if used
            classifier_loss = F.cross_entropy(classifier_logits, y)
            accuracy = (classifier_logits.argmax(dim=1) == y).float().mean()
        else:
            classifier_loss = torch.tensor(0.0, device=x_with_cls.device)
            accuracy = torch.tensor(0.0, device=x_with_cls.device)

        # ---- Total loss ----
        total_loss = (
            rec_loss_vq
            + self.info_nce_weight    * info_nce_loss
            + self.masked_dino_weight * masked_dino_ce_loss
            + classifier_loss
        )

        # ---- Debug stash ----
        with torch.no_grad():
            frac_masked   = masked_256.float().mean()
            frac_dropped  = dropped_256.float().mean()
            vq_norm_mean  = x_with_cls[:, 1:129, :].norm(dim=-1).mean()
        self.debug = {
            "rec_loss_vq":         rec_loss_vq.detach(),
            "info_nce_loss":       (self.info_nce_weight * info_nce_loss).detach(),
            "masked_dino_ce":      (self.masked_dino_weight * masked_dino_ce_loss).detach(),
            "frac_masked":         frac_masked.detach(),
            "frac_dropped":        frac_dropped.detach(),
            "vq_feat_norm":        vq_norm_mean.detach(),
        }

        
        return total_loss, rec_loss_vq, mask_full_vq, accuracy, classifier_loss, x_with_cls




class LEASEViT(nn.Module):
    def __init__(self,
                 img_size=256, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 mask_ratio_min=0.5, mask_ratio_max=1.0, mask_ratio_mu=0.55, mask_ratio_std=0.25,
                 vqgan_ckpt_path='vqgan_jax_strongaug.ckpt',
                 
                 dino_codebook_path="./km_16k.npy",      # path to npy [K,D]
                 info_nce_tau=0.1, #0.1 for IN1k
                 info_nce_weight=0.1,
                 info_nce_full_softmax=True,   
                 info_nce_chunk_k=4096):
        super().__init__()

        # --------------------------------------------------------------------------
        # VQ token vocabulary 
        print(dino_codebook_path)
        config = OmegaConf.load('config/vqgan.yaml').model
        seq_len_expected = 256                     # 16x16 grid
        self.vocab_size = 17030                   
        self.vqgan_vocab_size = 1024               # VQ codes are [0..1023]
        self.num_specials = 2
        self.code_vocab_size = self.vocab_size - self.num_specials
        self.fake_class_label = self.vocab_size - 2
        self.mask_token_label = self.vocab_size - 1

        # Token embedding (shared MLM head weights)
        self.token_emb = BertEmbeddings(
            vocab_size=self.vocab_size,
            hidden_size=embed_dim,
            max_position_embeddings=seq_len_expected + 1,   # +1 for CLS → 257
            dropout=0.1
        )

        # Shared frozen 2D sin-cos pos embed (16x16)
        self.pos2d = nn.Parameter(torch.zeros(1, seq_len_expected, embed_dim), requires_grad=False)
        pos2d = get_2d_sincos_pos_embed(embed_dim, int(seq_len_expected ** 0.5), cls_token=False)  # (256, E)
        self.pos2d.data.copy_(torch.from_numpy(pos2d).float().unsqueeze(0))

        # Masking sampler 
        self.mask_ratio_min = mask_ratio_min
        self.mask_ratio_generator = stats.truncnorm(
            (mask_ratio_min - mask_ratio_mu) / mask_ratio_std,
            (mask_ratio_max - mask_ratio_mu) / mask_ratio_std,
            loc=mask_ratio_mu, scale=mask_ratio_std
        )

        # --------------------------------------------------------------------------
        # Encoder 
        dropout_rate = 0.1
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = nn.ModuleList([
            FlashBlock(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer,
                       drop=dropout_rate, attn_drop=dropout_rate)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # --------------------------------------------------------------------------
        # VQ decoder 
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.pad_with_cls_token = True
        # learned decoder pos-embed: keep length 385 to be compatible; we will slice to 257
        self.decoder_pos_embed_learned = nn.Parameter(
            torch.zeros(1, seq_len_expected + 128 + 1, decoder_embed_dim)
        )
        self.decoder_blocks = nn.ModuleList([
            FlashBlock(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer,
                       drop=dropout_rate, attn_drop=dropout_rate)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = norm_layer(decoder_embed_dim)

        # MLM head + criterion 
        self.mlm_layer = MlmLayer(feat_emb_dim=decoder_embed_dim, word_emb_dim=embed_dim, vocab_size=self.vocab_size)
        self.norm_pix_loss = norm_pix_loss
        self.criterion = LabelSmoothingCrossEntropy(smoothing=0.1)

        # --------------------------------------------------------------------------
        # DINO centroids InfoNCE branch (no extra decoder)
        assert dino_codebook_path is not None, "Provide dino_codebook_path to use InfoNCE distillation."
        C = np.load(dino_codebook_path).astype(np.float32)  # [K, D]
        C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)      # L2 row-norm
        self.register_buffer("dino_codebook", torch.from_numpy(C))       # [K, D] frozen
        self.dino_dim = C.shape[1]
        # project encoder dim -> DINO centroid dim
        self.proj_to_dino = nn.Sequential(
            nn.Linear(embed_dim, self.dino_dim, bias=False),
        )
        self.info_nce_tau = info_nce_tau
        self.info_nce_weight = info_nce_weight
        self.info_nce_full_softmax = info_nce_full_softmax
        self.info_nce_chunk_k = info_nce_chunk_k

        # Multi-positive target knobs
        self.use_neighbor_targets   = True
        self.neighbor_topk          = 5          # include 5 neighbors in addition to the true centroid
        self.neighbor_teacher_tau   = 0.1        # temperature to compute soft weights over neighbors

        # --------------------------------------------------------------------------
        # Decoder → DINO (masked-patch CE)
        self.dec_to_dino = nn.Linear(decoder_embed_dim, self.dino_dim, bias=False)
        self.masked_dino_weight = 0.1
        self.masked_dino_tau    = self.info_nce_tau

        self.initialize_weights()

        # Optional online classifier 
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1000)  # num_classes
        )

        self.debug = {}

    # =========================
    # Initialization Utilities
    # =========================
    def initialize_weights(self):
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.decoder_pos_embed_learned, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # =========================
    # Helpers
    # =========================
    def _pool_stream_vq(self, x_with_cls, kept_mask=None):
        feats = x_with_cls[:, 1:129, :]  # (B,128,E)
        if kept_mask is None:
            return feats.mean(dim=1)
        unmasked = (~kept_mask).float()
        denom = unmasked.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (feats * unmasked.unsqueeze(-1)).sum(dim=1) / denom

    # =========================
    # Encoder / Decoder
    # =========================


    @torch.compiler.disable
    def _sample_mask_params(self, bsz, L, dev):
        """Non-compilable: scipy truncnorm. Returns mask_rate (float), num_masked (int)."""
        mask_rate  = float(self.mask_ratio_generator.rvs(1)[0])
        num_masked = int(np.ceil(L * mask_rate))
        return mask_rate, num_masked

    @torch.compiler.disable
    def forward_encoder(self, vq_idx, labels):
        bsz = vq_idx.size(0)
        dev = vq_idx.device

        vq_idx = vq_idx.reshape(bsz, 256).long()   # (B,256)
        L = 256
        K_keep = 128

        # ---- sample permutation → drop/keep & mask ----
        perm = torch.rand(bsz, L, device=dev).argsort(dim=1)
        drop_idx = perm[:, :L - K_keep]                 # drop these at the embedding stage
        keep_idx = perm[:, L - K_keep:]                 # keep these

        mask_rate, num_masked = self._sample_mask_params(bsz, L, dev)
        mask_idx   = perm[:, :num_masked]

        masked_256  = torch.zeros(bsz, L, device=dev, dtype=torch.bool)
        dropped_256 = torch.zeros(bsz, L, device=dev, dtype=torch.bool)
        masked_256.scatter_(1, mask_idx, True)
        dropped_256.scatter_(1, drop_idx, True)

        # ---- apply mask token to INPUT IDs only  ----
        vq_masked = vq_idx.clone()
        vq_masked[masked_256] = self.mask_token_label

        # ---- EMBED FULL [CLS | 256] first (BertEmbeddings sees all 257 positions) ----
        cls_col     = torch.full((bsz, 1), self.fake_class_label, device=dev, dtype=torch.long)
        tokens_full = torch.cat([cls_col, vq_masked], dim=1)              # (B, 257)
        x_full = self.token_emb(tokens_full)                              # (B, 257, E)

        # ---- NOW drop to 128 on the EMBEDDINGS ----
        keep_idx_sorted = torch.sort(keep_idx, dim=1).values              # (B,128) in original grid order

        # gather kept patch embeddings from the full embedded stream
        patch_keep_emb = torch.take_along_dim(
            x_full[:, 1:, :],                                             # (B,256,E)
            keep_idx_sorted.unsqueeze(-1).expand(-1, -1, x_full.size(-1)),
            dim=1
        )                                                                 # (B,128,E)

        # rebuild [CLS | kept] on EMBEDDINGS
        x = torch.cat([x_full[:, 0:1, :], patch_keep_emb], dim=1)         # (B, 1+128, E)

        # ---- add fixed 2D pos for the kept spatial positions (CLS row zeros) ----
        B, Np1, E = x.shape
        pos2d_kept = torch.take_along_dim(
            self.pos2d.expand(B, -1, -1),                                 # (B,256,E)
            keep_idx_sorted.unsqueeze(-1).expand(-1, -1, E),
            dim=1
        )                                                                 # (B,128,E)
        pos2d_enc = torch.cat([torch.zeros(B, 1, E, device=dev), pos2d_kept], dim=1)
        x = x + pos2d_enc

        # ---- transformer + final norm ----
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)                                                  # (B, 129, E)

        gt_vq256 = vq_idx.detach().long()
        return x, gt_vq256, keep_idx_sorted, masked_256, dropped_256



    def forward_decoder_vq(self, x_with_cls, keep_idx, masked_256, dropped_256, return_feats: bool = False):
        """
        VQ decoder only. Rebuild a 256-slot canvas in original order and decode.
        If return_feats=True, also returns decoder features after decoder_norm: (B,257,Dd).
        """
        B, Np1, E = x_with_cls.shape
        z = self.decoder_embed(x_with_cls)     # (B,129,Dd)
        cls_row = z[:, 0:1, :]                 # (B,1,Dd)
        vq_z    = z[:, 1:129, :]               # (B,128,Dd)

        # Build canvas in original order: fill only visible kept; others are placeholders
        vq_canvas = cls_row.repeat(1, 256, 1)
        kept_mask_vq = torch.take_along_dim(masked_256, keep_idx, dim=1)  # True=masked among kept
        kept_unmasked = ~kept_mask_vq
        place_idx = keep_idx.masked_fill(~kept_unmasked, 0)
        vals = vq_z.masked_fill(~kept_unmasked.unsqueeze(-1), 0)
        vq_canvas.scatter_add_(
            dim=1,
            index=place_idx.unsqueeze(-1).expand(-1, -1, vq_canvas.size(-1)),
            src=vals
        )

        # No DINO tail; seq = [CLS | 256]
        dec_in = torch.cat([cls_row, vq_canvas], dim=1)      # (B,257,Dd)
        pos = self.decoder_pos_embed_learned[:, :257, :]
        dec_in = dec_in + pos

        for blk in self.decoder_blocks:
            dec_in = blk(dec_in)
        dec_in = self.decoder_norm(dec_in)                   # (B,257,Dd)

        with torch.no_grad():
            word_embeddings = self.token_emb.word_embeddings.weight.detach()
        logits = self.mlm_layer(dec_in, word_embeddings)     # (B,257,V)

        if return_feats:
            return logits, dec_in
        return logits

    # =========================
    # Losses
    # =========================
    def forward_loss_vq(self, gt_indices, logits, mask):
        assert gt_indices.dtype == torch.long
        bsz, seq_len = gt_indices.size()
        logits_code = logits[:, 1:1+seq_len, :self.vqgan_vocab_size]  # (B,256,1024)
        loss = self.criterion(
            logits_code.reshape(bsz * seq_len, -1),
            gt_indices.reshape(bsz * seq_len)
        ).reshape(bsz, seq_len)
        loss = (loss * mask[:, 1:1+seq_len]).sum() / (mask[:, 1:1+seq_len].sum() + 1e-6)
        return loss

    def info_nce_visible_full(self, z_vis, t_vis):
        # L2-normalize
        z = F.normalize(z_vis, dim=-1)
        C = self.dino_codebook
        if self.info_nce_full_softmax:
            logits = F.linear(z, C) / self.info_nce_tau   # [N,K]
            return F.cross_entropy(logits.float(), t_vis)
        else:
            N, E = z.shape
            K = C.shape[0]
            tau = self.info_nce_tau
            c_pos = C.index_select(0, t_vis)                    # [N,E]
            l_pos = (z * c_pos).sum(-1) / tau                   # [N]
            m = torch.full((N,), -float('inf'), device=z.device, dtype=z.dtype)
            s = torch.zeros((N,), device=z.device, dtype=z.dtype)
            for s_k in range(0, K, self.info_nce_chunk_k):
                e_k = min(K, s_k + self.info_nce_chunk_k)
                Ck = C[s_k:e_k]                                  # [k,E]
                logits_k = (z @ Ck.t()) / tau                    # [N,k]
                m_new = torch.maximum(m, logits_k.max(dim=1).values)
                s = s * torch.exp(m - m_new) + torch.exp(logits_k - m_new.unsqueeze(1)).sum(dim=1)
                m = m_new
            log_denom = m + torch.log(s + 1e-12)
            loss_vec = -(l_pos - log_denom)
            return loss_vec.mean().float()

    
    def info_nce_visible_neighbors(self, z_vis, t_vis, topk=None, teacher_tau=None, loss_tau=None):
        """
        Multi-positive InfoNCE using the correct centroid + its top-k nearest neighbors,
        with similarity-weighted soft targets.

        z_vis: [N, D] (projected to DINO dim; NOT normalized)
        t_vis: [N]    int labels in [0,K-1]
        """
        if topk is None:
            topk = self.neighbor_topk
        if teacher_tau is None:
            teacher_tau = self.neighbor_teacher_tau
        if loss_tau is None:
            loss_tau = self.info_nce_tau

        # normalize
        z = F.normalize(z_vis, dim=-1)                  # [N, D]
        C = F.normalize(self.dino_codebook, dim=-1)     # [K, D]
        N, D = z.shape
        K = C.shape[0]

        # student logits over full codebook
        logits = F.linear(z, C) / loss_tau              # [N, K]
        logp = F.log_softmax(logits, dim=1)             # [N, K]

        # teacher soft labels over (self + neighbors)
        with torch.no_grad():
            c_pos = C.index_select(0, t_vis)            # [N, D]
            sims  = F.linear(c_pos, C)                  # [N, K] cosine
            k_sel = min(topk + 1, K)                    # include "self" centroid
            sims_top, idx_top = sims.topk(k=k_sel, dim=1)         # [N, k_sel]
            weights = F.softmax(sims_top / max(teacher_tau, 1e-6), dim=1)  # [N, k_sel]; sum=1

        logp_sel = logp.gather(1, idx_top)              # [N, k_sel]
        loss = -(weights * logp_sel).sum(dim=1)
        return loss

    def masked_dino_ce(self, dec_feats_257, dino_idx_256, mask_full_257, tau=None):
        if tau is None:
            tau = self.masked_dino_tau

        supervise_mask = (mask_full_257[:, 1:] > 0)  # (B,256) bool

        dec_256 = dec_feats_257[:, 1:, :]                           # (B,256,Dd)
        z = self.dec_to_dino(dec_256)                                # (B,256,D)
        z = F.normalize(z, dim=-1)

        C = F.normalize(self.dino_codebook, dim=-1)                 # [K,D]
        t = dino_idx_256.long()                                      # (B,256)
        logits = F.linear(z, C) / tau                                # (B,256,K)

        B_ce, S_ce, K_ce = logits.shape
        logits_flat = logits.reshape(B_ce * S_ce, K_ce)
        targets_flat = t.reshape(B_ce * S_ce)
        mask_flat = supervise_mask.reshape(B_ce * S_ce)
        ignore_idx = -100
        targets_flat = torch.where(
            mask_flat,
            targets_flat,
            torch.full_like(targets_flat, ignore_idx)
        )
        return F.cross_entropy(logits_flat.float(), targets_flat, ignore_index=ignore_idx)

    # =========================
    # Forward
    # =========================
    def forward(self, vq_idx, dino_idx, labels, epoch):
        """
        vq_idx:   (B,256) VQ token ids in [0..1023]
        dino_idx: (B,256) DINO token ids in [0..K-1] (used for InfoNCE/CE teacher)
        labels:   (B,)    class labels (optional)
        """
        # ---- Encode (VQ-only) ----
        x_with_cls, gt_vq256, keep_idx, masked_256, dropped_256 = self.forward_encoder(vq_idx, labels)

        # ---- Decode once (reuse feats) + VQ MLM loss ----
        logits_vq, dec_feats_257 = self.forward_decoder_vq(
            x_with_cls, keep_idx, masked_256, dropped_256, return_feats=True
        )
        vq_loss_mask256 = (masked_256 | dropped_256).to(torch.float32)
        mask_full_vq = torch.cat(
            [torch.zeros(vq_loss_mask256.size(0), 1, device=vq_loss_mask256.device),
             vq_loss_mask256], dim=1
        )  # (B,257)
        rec_loss_vq = self.forward_loss_vq(gt_vq256, logits_vq, mask_full_vq)

        # ---- Decoder-based DINO CE on masked ∪ dropped tokens (Not used in vanilla LEASE) ----
        B = vq_idx.size(0)
        dino_idx_256 = dino_idx.reshape(B, 256)
        masked_dino_ce_loss = self.masked_dino_ce(
            dec_feats_257, dino_idx_256, mask_full_vq, tau=self.masked_dino_tau
        )

        # ---- InfoNCE on visible (kept & unmasked) patches ----
        vq_feats_kept = x_with_cls[:, 1:129, :]                         # (B,128,E)
        kept_mask = torch.take_along_dim(masked_256, keep_idx, dim=1)   # (B,128) True=masked among kept
        visible_mask = ~kept_mask                                       # (B,128)

        z_vis = self.proj_to_dino(vq_feats_kept)                         # (B,128,D)
        t_vis = torch.take_along_dim(dino_idx, keep_idx, dim=1).long()   # (B,128)

        B_vis, S_vis = z_vis.shape[:2]
        z_vis_flat = z_vis.reshape(B_vis * S_vis, -1)
        t_vis_flat = t_vis.reshape(B_vis * S_vis)
        mask_flat = visible_mask.reshape(B_vis * S_vis)

        if self.use_neighbor_targets and self.neighbor_topk > 0:
            t_vis_safe = torch.where(mask_flat, t_vis_flat, torch.zeros_like(t_vis_flat))
            raw_loss = self.info_nce_visible_neighbors(
                z_vis_flat, t_vis_safe,
                topk=self.neighbor_topk,
                teacher_tau=self.neighbor_teacher_tau,
                loss_tau=self.info_nce_tau
            )
            w = mask_flat.to(z_vis_flat.dtype)
            info_nce_loss = (raw_loss * w).sum() / (w.sum().clamp(min=1))
        else:
            z_norm = F.normalize(z_vis_flat, dim=-1)
            C_norm = F.normalize(self.dino_codebook, dim=-1)
            logits = F.linear(z_norm, C_norm) / self.info_nce_tau
            ignore_idx = -100
            t_masked = torch.where(mask_flat, t_vis_flat, torch.full_like(t_vis_flat, ignore_idx))
            info_nce_loss = F.cross_entropy(logits.float(), t_masked, ignore_index=ignore_idx)

        # ---- Optional online classifier ----
        if labels is not None:
            y = labels.view(-1)
            classifier_logits = self.classifier(x_with_cls[:, 1:, :].detach().mean(1)) # IMPORTANT: Always detatch latents if used
            classifier_loss = F.cross_entropy(classifier_logits, y)
            accuracy = (classifier_logits.argmax(dim=1) == y).float().mean()
        else:
            classifier_loss = torch.tensor(0.0, device=x_with_cls.device)
            accuracy = torch.tensor(0.0, device=x_with_cls.device)

        # ---- Total loss ----
        total_loss = (
            rec_loss_vq
            + self.info_nce_weight    * info_nce_loss
            + 0 * masked_dino_ce_loss #We do not use this loss in vanilla LEASE
            + classifier_loss
        )

        # ---- Debug stash ----
        with torch.no_grad():
            frac_masked   = masked_256.float().mean()
            frac_dropped  = dropped_256.float().mean()
            vq_norm_mean  = x_with_cls[:, 1:129, :].norm(dim=-1).mean()
        self.debug = {
            "rec_loss_vq":         rec_loss_vq.detach(),
            "info_nce_loss":       (self.info_nce_weight * info_nce_loss).detach(),
            "masked_dino_ce":      (self.masked_dino_weight * masked_dino_ce_loss).detach(),
            "frac_masked":         frac_masked.detach(),
            "frac_dropped":        frac_dropped.detach(),
            "vq_feat_norm":        vq_norm_mean.detach(),
        }

        # Keep return signature identical
        return total_loss, rec_loss_vq, mask_full_vq, accuracy, classifier_loss, x_with_cls







class LeaseInference(nn.Module):
    """LEASE inference model using standard attention (no flash_attn)."""
    def __init__(self,
                 img_size=256, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 mask_ratio_min=0.5, mask_ratio_max=1.0, mask_ratio_mu=0.55, mask_ratio_std=0.25,
                 vqgan_ckpt_path='vqgan_jax_strongaug.ckpt',
                 dino_codebook_path="./km_16k.npy",
                 info_nce_tau=0.1, 
                 info_nce_weight=0.1,
                 info_nce_full_softmax=True,
                 info_nce_chunk_k=4096):
        super().__init__()

        config = OmegaConf.load('config/vqgan.yaml').model

        config = OmegaConf.load('config/vqgan.yaml').model
        self.vqgan = VQModel(ddconfig=config.params.ddconfig,
                             n_embed=config.params.n_embed,
                             embed_dim=config.params.embed_dim,
                             ckpt_path=vqgan_ckpt_path)
    
        for param in self.vqgan.parameters():
            param.requires_grad = False

        seq_len_expected = 256
        self.vocab_size = 17030
        self.vqgan_vocab_size = 1024
        self.num_specials = 2
        self.code_vocab_size = self.vocab_size - self.num_specials
        self.fake_class_label = self.vocab_size - 2
        self.mask_token_label = self.vocab_size - 1

        self.token_emb = BertEmbeddings(
            vocab_size=self.vocab_size,
            hidden_size=embed_dim,
            max_position_embeddings=seq_len_expected + 1,
            dropout=0.1
        )

        self.pos2d = nn.Parameter(torch.zeros(1, seq_len_expected, embed_dim), requires_grad=False)
        pos2d = get_2d_sincos_pos_embed(embed_dim, int(seq_len_expected ** 0.5), cls_token=False)
        self.pos2d.data.copy_(torch.from_numpy(pos2d).float().unsqueeze(0))

        self.mask_ratio_min = mask_ratio_min
        self.mask_ratio_generator = stats.truncnorm(
            (mask_ratio_min - mask_ratio_mu) / mask_ratio_std,
            (mask_ratio_max - mask_ratio_mu) / mask_ratio_std,
            loc=mask_ratio_mu, scale=mask_ratio_std
        )

        dropout_rate = 0.1
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer,
                       drop=dropout_rate, attn_drop=dropout_rate)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.pad_with_cls_token = True
        self.decoder_pos_embed_learned = nn.Parameter(
            torch.zeros(1, seq_len_expected + 128 + 1, decoder_embed_dim)
        )
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer,
                       drop=dropout_rate, attn_drop=dropout_rate)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = norm_layer(decoder_embed_dim)

        self.mlm_layer = MlmLayer(feat_emb_dim=decoder_embed_dim, word_emb_dim=embed_dim, vocab_size=self.vocab_size)
        self.norm_pix_loss = norm_pix_loss
        self.criterion = LabelSmoothingCrossEntropy(smoothing=0.1)

        assert dino_codebook_path is not None, "Provide dino_codebook_path to use InfoNCE distillation."
        C = np.load(dino_codebook_path).astype(np.float32)
        C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
        self.register_buffer("dino_codebook", torch.from_numpy(C))
        self.dino_dim = C.shape[1]
        self.proj_to_dino = nn.Sequential(
            nn.Linear(embed_dim, self.dino_dim, bias=False),
        )
        self.info_nce_tau = info_nce_tau
        self.info_nce_weight = info_nce_weight
        self.info_nce_full_softmax = info_nce_full_softmax
        self.info_nce_chunk_k = info_nce_chunk_k

        self.use_neighbor_targets   = True
        self.neighbor_topk          = 5
        self.neighbor_teacher_tau   = 0.1

        self.dec_to_dino = nn.Linear(decoder_embed_dim, self.dino_dim, bias=False)
        self.masked_dino_weight = 0.1
        self.masked_dino_tau    = self.info_nce_tau

        self.initialize_weights()

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1000)
        )

        self.debug = {}

    def initialize_weights(self):
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.decoder_pos_embed_learned, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _pool_stream_vq(self, x_with_cls, kept_mask=None):
        feats = x_with_cls[:, 1:129, :]
        if kept_mask is None:
            return feats.mean(dim=1)
        unmasked = (~kept_mask).float()
        denom = unmasked.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (feats * unmasked.unsqueeze(-1)).sum(dim=1) / denom

    @torch.compiler.disable
    def forward_encoder(self, vq_idx, labels):
        bsz = vq_idx.size(0)
        dev = vq_idx.device

        vq_idx = vq_idx.reshape(bsz, 256).long()
        L = 256
        K_keep = 128

        perm = torch.rand(bsz, L, device=dev).argsort(dim=1)
        drop_idx = perm[:, :L - K_keep]
        keep_idx = perm[:, L - K_keep:]

        mask_rate  = float(self.mask_ratio_generator.rvs(1)[0])
        num_masked = int(np.ceil(L * mask_rate))
        mask_idx   = perm[:, :num_masked]

        masked_256  = torch.zeros(bsz, L, device=dev, dtype=torch.bool)
        dropped_256 = torch.zeros(bsz, L, device=dev, dtype=torch.bool)
        masked_256.scatter_(1, mask_idx, True)
        dropped_256.scatter_(1, drop_idx, True)

        vq_masked = vq_idx.clone()
        vq_masked[masked_256] = self.mask_token_label

        cls_col     = torch.full((bsz, 1), self.fake_class_label, device=dev, dtype=torch.long)
        tokens_full = torch.cat([cls_col, vq_masked], dim=1)
        x_full = self.token_emb(tokens_full)

        keep_idx_sorted = torch.sort(keep_idx, dim=1).values

        patch_keep_emb = torch.take_along_dim(
            x_full[:, 1:, :],
            keep_idx_sorted.unsqueeze(-1).expand(-1, -1, x_full.size(-1)),
            dim=1
        )

        x = torch.cat([x_full[:, 0:1, :], patch_keep_emb], dim=1)

        B, Np1, E = x.shape
        pos2d_kept = torch.take_along_dim(
            self.pos2d.expand(B, -1, -1),
            keep_idx_sorted.unsqueeze(-1).expand(-1, -1, E),
            dim=1
        )
        pos2d_enc = torch.cat([torch.zeros(B, 1, E, device=dev), pos2d_kept], dim=1)
        x = x + pos2d_enc

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        gt_vq256 = vq_idx.detach().long()
        return x, gt_vq256, keep_idx_sorted, masked_256, dropped_256

    def forward_decoder_vq(self, x_with_cls, keep_idx, masked_256, dropped_256, return_feats: bool = False):
        B, Np1, E = x_with_cls.shape
        z = self.decoder_embed(x_with_cls)
        cls_row = z[:, 0:1, :]
        vq_z    = z[:, 1:257, :]

        vq_canvas = cls_row.repeat(1, 256, 1)
        kept_mask_vq = torch.take_along_dim(masked_256, keep_idx, dim=1)
        kept_unmasked = ~kept_mask_vq
        place_idx = keep_idx.masked_fill(~kept_unmasked, 0)
        vals = vq_z.masked_fill(~kept_unmasked.unsqueeze(-1), 0)
        vq_canvas.scatter_add_(
            dim=1,
            index=place_idx.unsqueeze(-1).expand(-1, -1, vq_canvas.size(-1)),
            src=vals
        )

        dec_in = torch.cat([cls_row, vq_canvas], dim=1)
        pos = self.decoder_pos_embed_learned[:, :257, :]
        dec_in = dec_in + pos

        for blk in self.decoder_blocks:
            dec_in = blk(dec_in)
        dec_in = self.decoder_norm(dec_in)

        with torch.no_grad():
            word_embeddings = self.token_emb.word_embeddings.weight.detach()
        logits = self.mlm_layer(dec_in, word_embeddings)

        if return_feats:
            return logits, dec_in
        return logits

    def forward_loss_vq(self, gt_indices, logits, mask):
        assert gt_indices.dtype == torch.long
        bsz, seq_len = gt_indices.size()
        logits_code = logits[:, 1:1+seq_len, :self.vqgan_vocab_size]
        loss = self.criterion(
            logits_code.reshape(bsz * seq_len, -1),
            gt_indices.reshape(bsz * seq_len)
        ).reshape(bsz, seq_len)
        loss = (loss * mask[:, 1:1+seq_len]).sum() / (mask[:, 1:1+seq_len].sum() + 1e-6)
        return loss

    def info_nce_visible_full(self, z_vis, t_vis):
        z = F.normalize(z_vis, dim=-1)
        C = self.dino_codebook
        if self.info_nce_full_softmax:
            logits = F.linear(z, C) / self.info_nce_tau
            return F.cross_entropy(logits.float(), t_vis)
        else:
            N, E = z.shape
            K = C.shape[0]
            tau = self.info_nce_tau
            c_pos = C.index_select(0, t_vis)
            l_pos = (z * c_pos).sum(-1) / tau
            m = torch.full((N,), -float('inf'), device=z.device, dtype=z.dtype)
            s = torch.zeros((N,), device=z.device, dtype=z.dtype)
            for s_k in range(0, K, self.info_nce_chunk_k):
                e_k = min(K, s_k + self.info_nce_chunk_k)
                Ck = C[s_k:e_k]
                logits_k = (z @ Ck.t()) / tau
                m_new = torch.maximum(m, logits_k.max(dim=1).values)
                s = s * torch.exp(m - m_new) + torch.exp(logits_k - m_new.unsqueeze(1)).sum(dim=1)
                m = m_new
            log_denom = m + torch.log(s + 1e-12)
            loss_vec = -(l_pos - log_denom)
            return loss_vec.mean().float()

    def info_nce_visible_neighbors(self, z_vis, t_vis, topk=None, teacher_tau=None, loss_tau=None):
        if topk is None:
            topk = self.neighbor_topk
        if teacher_tau is None:
            teacher_tau = self.neighbor_teacher_tau
        if loss_tau is None:
            loss_tau = self.info_nce_tau

        z = F.normalize(z_vis, dim=-1)
        C = F.normalize(self.dino_codebook, dim=-1)
        N, D = z.shape
        K = C.shape[0]

        logits = F.linear(z, C) / loss_tau
        logp = F.log_softmax(logits, dim=1)

        with torch.no_grad():
            c_pos = C.index_select(0, t_vis)
            sims  = F.linear(c_pos, C)
            k_sel = min(topk + 1, K)                    # include "self" centroid
            sims_top, idx_top = sims.topk(k=k_sel, dim=1)
            weights = F.softmax(sims_top / max(teacher_tau, 1e-6), dim=1)

        logp_sel = logp.gather(1, idx_top)
        loss = -(weights * logp_sel).sum(dim=1)
        return loss

    def masked_dino_ce(self, dec_feats_257, dino_idx_256, mask_full_257, tau=None):
        if tau is None:
            tau = self.masked_dino_tau

        supervise_mask = (mask_full_257[:, 1:] > 0)

        dec_256 = dec_feats_257[:, 1:, :]
        z = self.dec_to_dino(dec_256)
        z = F.normalize(z, dim=-1)

        C = F.normalize(self.dino_codebook, dim=-1)
        t = dino_idx_256.long()
        logits = F.linear(z, C) / tau

        B_ce, S_ce, K_ce = logits.shape
        logits_flat = logits.reshape(B_ce * S_ce, K_ce)
        targets_flat = t.reshape(B_ce * S_ce)
        mask_flat = supervise_mask.reshape(B_ce * S_ce)
        ignore_idx = -100
        targets_flat = torch.where(
            mask_flat,
            targets_flat,
            torch.full_like(targets_flat, ignore_idx)
        )
        return F.cross_entropy(logits_flat.float(), targets_flat, ignore_index=ignore_idx)

    def forward(self, vq_idx, dino_idx, labels, epoch):
        x_with_cls, gt_vq256, keep_idx, masked_256, dropped_256 = self.forward_encoder(vq_idx, labels)

        logits_vq, dec_feats_257 = self.forward_decoder_vq(
            x_with_cls, keep_idx, masked_256, dropped_256, return_feats=True
        )
        vq_loss_mask256 = (masked_256 | dropped_256).to(torch.float32)
        mask_full_vq = torch.cat(
            [torch.zeros(vq_loss_mask256.size(0), 1, device=vq_loss_mask256.device),
             vq_loss_mask256], dim=1
        )
        rec_loss_vq = self.forward_loss_vq(gt_vq256, logits_vq, mask_full_vq)

        B = vq_idx.size(0)
        dino_idx_256 = dino_idx.reshape(B, 256)
        masked_dino_ce_loss = self.masked_dino_ce(
            dec_feats_257, dino_idx_256, mask_full_vq, tau=self.masked_dino_tau
        )

        vq_feats_kept = x_with_cls[:, 1:129, :]
        kept_mask = torch.take_along_dim(masked_256, keep_idx, dim=1)
        visible_mask = ~kept_mask

        z_vis = self.proj_to_dino(vq_feats_kept)
        t_vis = torch.take_along_dim(dino_idx, keep_idx, dim=1).long()

        B_vis, S_vis = z_vis.shape[:2]
        z_vis_flat = z_vis.reshape(B_vis * S_vis, -1)
        t_vis_flat = t_vis.reshape(B_vis * S_vis)
        mask_flat = visible_mask.reshape(B_vis * S_vis)

        if self.use_neighbor_targets and self.neighbor_topk > 0:
            t_vis_safe = torch.where(mask_flat, t_vis_flat, torch.zeros_like(t_vis_flat))
            raw_loss = self.info_nce_visible_neighbors(
                z_vis_flat, t_vis_safe,
                topk=self.neighbor_topk,
                teacher_tau=self.neighbor_teacher_tau,
                loss_tau=self.info_nce_tau
            )
            w = mask_flat.to(z_vis_flat.dtype)
            info_nce_loss = (raw_loss * w).sum() / (w.sum().clamp(min=1))
        else:
            z_norm = F.normalize(z_vis_flat, dim=-1)
            C_norm = F.normalize(self.dino_codebook, dim=-1)
            logits = F.linear(z_norm, C_norm) / self.info_nce_tau
            ignore_idx = -100
            t_masked = torch.where(mask_flat, t_vis_flat, torch.full_like(t_vis_flat, ignore_idx))
            info_nce_loss = F.cross_entropy(logits.float(), t_masked, ignore_index=ignore_idx)

        if labels is not None:
            y = labels.view(-1)
            classifier_logits = self.classifier(x_with_cls[:, 1:, :].detach().mean(1))
            classifier_loss = F.cross_entropy(classifier_logits, y)
            accuracy = (classifier_logits.argmax(dim=1) == y).float().mean()
        else:
            classifier_loss = torch.tensor(0.0, device=x_with_cls.device)
            accuracy = torch.tensor(0.0, device=x_with_cls.device)

        total_loss = (
            rec_loss_vq
            + self.info_nce_weight    * info_nce_loss
            + self.masked_dino_weight * masked_dino_ce_loss
            + classifier_loss
        )

        with torch.no_grad():
            frac_masked   = masked_256.float().mean()
            frac_dropped  = dropped_256.float().mean()
            vq_norm_mean  = x_with_cls[:, 1:129, :].norm(dim=-1).mean()
        self.debug = {
            "rec_loss_vq":         rec_loss_vq.detach(),
            "info_nce_loss":       (self.info_nce_weight * info_nce_loss).detach(),
            "masked_dino_ce":      (self.masked_dino_weight * masked_dino_ce_loss).detach(),
            "frac_masked":         frac_masked.detach(),
            "frac_dropped":        frac_dropped.detach(),
            "vq_feat_norm":        vq_norm_mean.detach(),
        }

        return total_loss, rec_loss_vq, mask_full_vq, accuracy, classifier_loss, x_with_cls


class SorcenInference(nn.Module):
    """Sorcen inference model using standard attention (no flash_attn)."""
    def __init__(self, img_size=256, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 mask_ratio_min=0.5, mask_ratio_max=1.0, mask_ratio_mu=0.55, mask_ratio_std=0.25,
                 vqgan_ckpt_path='vqgan_jax_strongaug.ckpt', dino_codebook_path=''):
        super().__init__()

        config = OmegaConf.load('config/vqgan.yaml').model

        self.vqgan = VQModel(ddconfig=config.params.ddconfig,
                             n_embed=config.params.n_embed,
                             embed_dim=config.params.embed_dim,
                             ckpt_path=vqgan_ckpt_path)
        for param in self.vqgan.parameters():
            param.requires_grad = False

        self.codebook_size = config.params.n_embed
        vocab_size = self.codebook_size + 1000 + 1
        self.fake_class_label = self.codebook_size + 1100 - 1024
        self.mask_token_label = vocab_size - 1
        self.token_emb = BertEmbeddings(vocab_size=vocab_size,
                                        hidden_size=embed_dim,
                                        max_position_embeddings=256+1,
                                        dropout=0.1)

        self.mask_ratio_min = mask_ratio_min
        self.mask_ratio_generator = stats.truncnorm((mask_ratio_min - mask_ratio_mu) / mask_ratio_std,
                                                    (mask_ratio_max - mask_ratio_mu) / mask_ratio_std,
                                                    loc=mask_ratio_mu, scale=mask_ratio_std)

        self.mask_ratio_generator_nn = stats.truncnorm((mask_ratio_min - 0.85) / mask_ratio_std,
                                                    (mask_ratio_max - 0.85) / mask_ratio_std,
                                                    loc=0.85, scale=mask_ratio_std)

        dropout_rate = 0.1
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer,
                  drop=dropout_rate, attn_drop=dropout_rate)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.beta = 0.996
        self.blocks_momentum = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer,
                  drop=dropout_rate, attn_drop=dropout_rate)
            for i in range(depth)])

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.pad_with_cls_token = True

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)
        self.decoder_pos_embed_learned = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim))

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer,
                  drop=dropout_rate, attn_drop=dropout_rate)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True)

        self.mlm_layer = MlmLayer(feat_emb_dim=decoder_embed_dim, word_emb_dim=embed_dim, vocab_size=vocab_size)
        self.norm_pix_loss = norm_pix_loss
        self.criterion = LabelSmoothingCrossEntropy(smoothing=0.1)

        self.projector = Projector(embed_dim, 4096, 512)
        self.momentum_projector = Projector(embed_dim, 4096, 512)
        self.predictor = Projector(512, 2048, 512)

        self.initialize_weights()

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1000)
        )

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.decoder_pos_embed_learned, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder(self, vq_idx, labels):
        token_indices = vq_idx.reshape(vq_idx.size(0), -1)
        gt_indices = token_indices.clone().detach().long()

        bsz, seq_len = token_indices.size()
        mask_rate = self.mask_ratio_generator.rvs(1)[0]
        num_dropped_tokens = int(np.ceil(seq_len * self.mask_ratio_min))
        num_masked_tokens = int(np.ceil(seq_len * mask_rate))

        while True:
            noise = torch.rand(bsz, seq_len, device=vq_idx.device)
            sorted_noise, _ = torch.sort(noise, dim=1)
            cutoff_drop = sorted_noise[:, num_dropped_tokens-1:num_dropped_tokens]
            cutoff_mask = sorted_noise[:, num_masked_tokens-1:num_masked_tokens]
            token_drop_mask = (noise <= cutoff_drop).float()
            token_all_mask = (noise <= cutoff_mask).float()
            if token_drop_mask.sum() == bsz*num_dropped_tokens and token_all_mask.sum() == bsz*num_masked_tokens:
                break

        token_indices[token_all_mask.nonzero(as_tuple=True)] = self.mask_token_label
        token_indices = torch.cat([torch.zeros(token_indices.size(0), 1).cuda(device=token_indices.device), token_indices], dim=1)
        token_indices[:, 0] = self.fake_class_label
        token_drop_mask = torch.cat([torch.zeros(token_indices.size(0), 1).cuda(), token_drop_mask], dim=1)
        token_all_mask = torch.cat([torch.zeros(token_indices.size(0), 1).cuda(), token_all_mask], dim=1)
        token_indices = token_indices.long()

        input_embeddings = self.token_emb(token_indices)
        bsz, seq_len, emb_dim = input_embeddings.shape
        token_keep_mask = 1 - token_drop_mask
        input_embeddings_after_drop = input_embeddings[token_keep_mask.nonzero(as_tuple=True)].reshape(bsz, -1, emb_dim)

        x = input_embeddings_after_drop
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, 0, gt_indices, token_drop_mask, token_all_mask

    def forward_decoder(self, x, token_drop_mask, token_all_mask):
        x = self.decoder_embed(x)

        if self.pad_with_cls_token:
            mask_tokens = x[:, 0:1].repeat(1, token_all_mask.shape[1], 1)
        else:
            mask_tokens = self.mask_token.repeat(token_all_mask.shape[0], token_all_mask.shape[1], 1)

        x_after_pad = mask_tokens.clone()
        x_after_pad[(1 - token_drop_mask).nonzero(as_tuple=True)] = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
        x_after_pad = torch.where(token_all_mask.unsqueeze(-1).bool(), mask_tokens, x_after_pad)

        x = x_after_pad + self.decoder_pos_embed_learned

        for blk in self.decoder_blocks:
            x = blk(x)

        x = self.decoder_norm(x)
        word_embeddings = self.token_emb.word_embeddings.weight.data.detach()
        x = self.mlm_layer(x, word_embeddings)

        return x

    def forward_loss(self, gt_indices, logits, mask):
        bsz, seq_len = gt_indices.size()
        loss = self.criterion(logits[:, 1:, :self.codebook_size].reshape(bsz*seq_len, -1), gt_indices.reshape(bsz*seq_len))
        loss = loss.reshape(bsz, seq_len)
        loss = (loss * mask[:, 1:]).sum() / mask[:, 1:].sum()
        return loss

    def forward(self, vq_idx, labels, epoch):
        latent, _, gt_indices, token_drop_mask, token_all_mask = self.forward_encoder(vq_idx, labels)
        logits = self.forward_decoder(latent, token_drop_mask, token_all_mask)
        rec_loss = self.forward_loss(gt_indices, logits, token_all_mask)
        if labels is not None:
            classifier_logits = self.classifier(latent.detach().mean(1))
            classifier_loss = F.cross_entropy(classifier_logits, labels.squeeze())
            accuracy = (classifier_logits.argmax(dim=1) == labels.squeeze()).float().mean()
        else:
            classifier_loss = torch.tensor(0.0)
            accuracy = torch.tensor(0.0)
        total_loss = rec_loss + classifier_loss
        return total_loss, rec_loss, torch.tensor(0.0), token_all_mask, accuracy, classifier_loss, latent

def inference_sorcen_vit_base_patch16_single(**kwargs):
    model = SorcenInference(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=768, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def inference_sorcen_vit_large_patch16_single(**kwargs):
    model = SorcenInference(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=1024, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def sorcen_vit_base_patch16_single(**kwargs):
    model = SorcenViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=768, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def sorcen_vit_large_patch16_single(**kwargs):
    model = SorcenViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=1024, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def lease_dec_loss_vit_base_patch16_single(**kwargs):
    model = LEASE_DecLoss_ViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=768, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model



def lease_vit_base_patch16_single(**kwargs):
    model = LEASEViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=768, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model



def lease_vit_large_patch16_single(**kwargs):
    model = LEASEViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=1024, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model



def inference_lease_vit_base_patch16_single(**kwargs):
    model = LeaseInference(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=768, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def inference_lease_vit_large_patch16_single(**kwargs):
    model = LeaseInference(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=1024, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

