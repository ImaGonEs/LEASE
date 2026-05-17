import torch
import os
import math
import argparse
import numpy as np
from tqdm import tqdm
import cv2
from PIL import Image


import models_lease_tk_1600  

from omegaconf import OmegaConf
import torch.backends.cudnn as cudnn

# (optional but helps stability/repro)
cudnn.benchmark = False
cudnn.deterministic = True

# ----------------------------
# helpers
# ----------------------------
def mask_by_random_topk(mask_len, probs, temperature=1.0):
    """
    Selects the positions to REMASK next, using Gumbel-TopK on the negative confidence.
    Lower-confidence predictions (after noise) are re-masked.
    """
    mask_len = mask_len.squeeze()
    gumbel = torch.empty_like(probs).exponential_().log().neg()  # ~ Gumbel(0,1)
    confidence = torch.log(probs.clamp_min(1e-12)) + temperature * gumbel
    sorted_confidence, _ = torch.sort(confidence, dim=-1)
    cut_off = sorted_confidence[:, mask_len.long()-1:mask_len.long()]
    masking = (confidence <= cut_off)
    return masking

@torch.no_grad()
def gen_image_lease(model, bsz, seed, num_iter=12, choice_temperature=4.5, device=None):
    """
    Iterative masked-token sampling for Lease (VQ-only path).
    """
    if device is None:
        device = next(model.parameters()).device

    torch.manual_seed(seed)
    np.random.seed(seed)

    # --- model/vocab constants
    L = 256                                   # 16x16 code grid
    codebook_size = int(getattr(model, "vqgan_vocab_size", 1024))
    mask_token_id = int(model.mask_token_label)
    cls_token_id  = int(model.fake_class_label)

    # get the VQ code embedding dim from your config
    config = OmegaConf.load('config/vqgan.yaml').model
    codebook_emb_dim = int(config.params.embed_dim)

    # init unknown tokens
    token_indices = torch.full((bsz, L), mask_token_id, dtype=torch.long, device=device)

    # reusable tensors
    keep_idx = torch.arange(L, device=device).unsqueeze(0).expand(bsz, -1)  # keep all in order
    dropped_256 = torch.zeros(bsz, L, dtype=torch.bool, device=device)      # no drops in generation

    for step in range(num_iter):
        cur_ids = token_indices.clone()

        # ----- build [CLS|tokens]
        cls_col = torch.full((bsz, 1), cls_token_id, dtype=torch.long, device=device)
        tokens_full = torch.cat([cls_col, cur_ids], dim=1)                  # (B, 257)

        # ----- embeddings + fixed 2D pos
        x = model.token_emb(tokens_full)                                    # (B,257,E)
        B, Np1, E = x.shape
        pos_full = torch.cat(
            [torch.zeros(B, 1, E, device=device), model.pos2d.expand(B, -1, -1)],
            dim=1
        )                                                                   # (B,257,E)
        x = x + pos_full

        # ----- encoder
        for blk in model.blocks:
            x = blk(x)
        x = model.norm(x)                                                   # (B,257,E)

        # ----- decoder → logits over global vocab
        masked_256 = (cur_ids == mask_token_id)                             # (B,256) unknowns
        logits = model.forward_decoder_vq(
            x_with_cls=x,
            keep_idx=keep_idx,           # identity placement (B,256)
            masked_256=masked_256,
            dropped_256=dropped_256,
            return_feats=False
        )                                                                     
        logits = logits[:, 1:, :codebook_size]                                # (B,256,1024)

        # ----- categorical sample predictions
        sample_dist = torch.distributions.categorical.Categorical(logits=logits)
        sampled_ids = sample_dist.sample()                                    # (B,256)

        # keep known ids; fill unknown with samples
        unknown_map = masked_256
        sampled_ids = torch.where(unknown_map, sampled_ids, cur_ids)

        # ----- cosine annealed remask ratio
        ratio = float(step + 1) / float(num_iter)
        mask_ratio = math.cos(math.pi / 2.0 * ratio)                          # 1→0 over time

        # ----- confidence and randomized top-k masking
        probs = torch.nn.functional.softmax(logits, dim=-1)
        selected_probs = torch.gather(probs, dim=-1, index=sampled_ids.unsqueeze(-1)).squeeze(-1)  # (B,256)
        
        selected_probs = torch.where(unknown_map, selected_probs, torch.full_like(selected_probs, float("+inf")))

        
        mask_len = torch.full((bsz, 1), float(np.floor(L * mask_ratio)), device=device)
        
        remaining_unknown = unknown_map.sum(dim=-1, keepdim=True).float()
        mask_len = torch.maximum(
            torch.ones_like(mask_len),
            torch.minimum(remaining_unknown - 1.0, mask_len)
        )

        
        temperature = choice_temperature * (1.0 - ratio)
        masking = mask_by_random_topk(mask_len[0], selected_probs, temperature)  # (B,256) bool

        
        token_indices = torch.where(masking, torch.full_like(sampled_ids, mask_token_id), sampled_ids)

    # ----- FINAL FILL: eliminate any leftover mask ids (greedy pass with full context)
    leftover = (token_indices == mask_token_id)  # (B,256)
    if leftover.any():
        cls_col = torch.full((bsz, 1), cls_token_id, dtype=torch.long, device=device)
        tokens_full = torch.cat([cls_col, token_indices], dim=1)              # (B,257)
        x = model.token_emb(tokens_full)
        B, Np1, E = x.shape
        pos_full = torch.cat(
            [torch.zeros(B, 1, E, device=device), model.pos2d.expand(B, -1, -1)],
            dim=1
        )
        x = x + pos_full
        for blk in model.blocks:
            x = blk(x)
        x = model.norm(x)

        logits = model.forward_decoder_vq(
            x_with_cls=x,
            keep_idx=keep_idx,
            masked_256=leftover,           
            dropped_256=torch.zeros_like(leftover),
            return_feats=False
        )[:, 1:, :codebook_size]          
        greedy = logits.argmax(dim=-1)  
        token_indices = torch.where(leftover, greedy, token_indices)

    # ----- sanity checks before decoding
    assert token_indices.min().item() >= 0
    assert token_indices.max().item() < codebook_size
    assert token_indices.shape == (bsz, 256)
    assert token_indices.dtype == torch.long


    # ----- VQGAN decode (prefer decode_code; else embed → NCHW → decode)
    z_q = model.vqgan.quantize.get_codebook_entry(
        token_indices, shape=(bsz, 16, 16, codebook_emb_dim)
    ).contiguous().float()      
                             # (B,D,16,16)

    gen_images = model.vqgan.decode(z_q)                                     # (B,3,H,W)

    return gen_images



parser = argparse.ArgumentParser('LEASE generation', add_help=False)
parser.add_argument('--temp', default=4.5, type=float, help='sampling temperature')
parser.add_argument('--num_iter', default=12, type=int, help='iterations')
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--num_images', default=50000, type=int, help='images to generate')
parser.add_argument('--ckpt', type=str, required=True, help='checkpoint path')
parser.add_argument('--model', default='inference_lease_vit_base_patch16_single', type=str, help='model ctor name in module')
parser.add_argument('--output_dir', default='output_dir/fid/gen/lease', type=str, help='output dir')

args = parser.parse_args()


vqgan_ckpt_path = 'vqgan_jax_strongaug.ckpt'
ModelCtor = models_lease_tk_1600.__dict__[args.model]
model = ModelCtor(vqgan_ckpt_path=vqgan_ckpt_path)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

# load checkpoint
checkpoint = torch.load(args.ckpt, map_location='cpu')
state_dict = checkpoint.get('model', checkpoint)


filtered_state_dict = {
    k: v for k, v in state_dict.items()
    if not (k.startswith('classifier.2.weight') or k.startswith('classifier.2.bias'))
}
missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)

if missing_keys:
    print(f"Missing keys:\n {missing_keys}")
if unexpected_keys:
    print(f"Unexpected keys:\n {unexpected_keys}")

model.eval()

# --------- generation loop
num_steps = args.num_images // args.batch_size + 1
save_folder = os.path.join(args.output_dir, f"temp{args.temp}-iter{args.num_iter}")
os.makedirs(save_folder, exist_ok=True)
print(save_folder)

with torch.no_grad():
    img_counter = 0
    for i in tqdm(range(num_steps)):
        gen_images_batch = gen_image_lease(
            model, bsz=args.batch_size, seed=i,
            choice_temperature=args.temp, num_iter=args.num_iter, device=device
        ).detach().cpu()  # (B,3,H,W)

        # save
        B = gen_images_batch.size(0)
        for b in range(B):
            if img_counter >= args.num_images:
                break
            img = np.clip(gen_images_batch[b].numpy().transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8)
            save_path = os.path.join(save_folder, f"{str(img_counter).zfill(5)}.png")
            Image.fromarray(img).save(save_path)
            print(f"Image saved in {save_path}")
            img_counter += 1
