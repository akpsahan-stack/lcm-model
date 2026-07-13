"""
LCM — Linear Consciousness Model  training script.

Adapted from nanoGPT (Karpathy) with the following changes:
  1. LCMConfig / LCM replace GPTConfig / GPT
  2. n_head removed (no attention — linear recurrence instead)
  3. forward() returns 3 values: (logits, loss, states)
  4. Optional FSDP for large hidden dimensions
  5. Gradient checkpointing support
  6. Multimodal data loading (text + image tokens)
  7. Prime-aware gradient modulation (optional)
  8. from_pretrained GPT-2 path removed (architecturally incompatible)

Usage:
  Single GPU:    python train.py --batch_size=32 --compile=False
  Dual T4 DDP:   torchrun --standalone --nproc_per_node=2 train.py
  With config:   python train.py config/train_xe.py
  FSDP (large):  torchrun --standalone --nproc_per_node=2 train.py --use_fsdp=True
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import LCMConfig, LCM

# ═══════════════════════════════════════════════════════════
# CONFIG — defaults tuned for LCM on Dual T4
# (overridden by config/train_xe.py or CLI args)
# ═══════════════════════════════════════════════════════════

# I/O
out_dir                  = 'out-lcm'
eval_interval            = 2000
log_interval             = 1
eval_iters               = 200
eval_only                = False
always_save_checkpoint   = True
init_from                = 'scratch'   # 'scratch' or 'resume'

# wandb logging
wandb_log                = False
wandb_project            = 'lcm'
wandb_run_name           = 'lcm-run'

# data
dataset                  = 'openwebtext'
gradient_accumulation_steps = 5 * 8
batch_size               = 12
block_size               = 1024

# LCM model (no n_head — recurrence replaces attention)
n_layer                  = 12
n_embd                   = 768
dropout                  = 0.0
bias                     = False
ffn_ratio                = 4.0
use_multimodal           = True
image_vocab_size         = 1024

# adamw optimizer
learning_rate            = 6e-4
max_iters                = 600000
weight_decay             = 1e-1
beta1                    = 0.9
beta2                    = 0.95
grad_clip                = 1.0

# learning rate decay (cosine with warmup)
decay_lr                 = True
warmup_iters             = 2000
lr_decay_iters           = 600000
min_lr                   = 6e-5

# DDP / FSDP
backend                  = 'nccl'
use_fsdp                 = False
use_gradient_checkpointing = True

# system
device                   = 'cuda'
dtype                    = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile                  = True

# multimodal (optional)
use_multimodal_data      = False
image_codec_type         = 'vqgan'

# prime gradient modulation (experimental, 0.0 = off)
prime_grad_modulation    = 0.0

# ═══════════════════════════════════════════════════════════
# CLI / CONFIG FILE OVERRIDES
# ═══════════════════════════════════════════════════════════

config_keys = [
    k for k, v in globals().items()
    if not k.startswith('_') and isinstance(v, (int, float, bool, str))
]
exec(open('configurator.py').read())
config = {k: globals()[k] for k in config_keys}

# ═══════════════════════════════════════════════════════════
# DDP INIT
# ═══════════════════════════════════════════════════════════

ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend=backend)
    ddp_rank       = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset    = ddp_rank
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset    = 0
    ddp_world_size = 1

tokens_per_iter = (
    gradient_accumulation_steps * ddp_world_size * batch_size * block_size
)
print(f"tokens per iteration: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True

device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {
    'float32':  torch.float32,
    'bfloat16': torch.bfloat16,
    'float16':  torch.float16,
}[dtype]
ctx = (
    nullcontext()
    if device_type == 'cpu'
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)

# ═══════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════

data_dir = os.path.join('data', dataset)

# multimodal loader (lazy import — only loaded when needed)
_multimodal_loader = None


def _get_multimodal_loader():
    """Lazy-load multimodal DataLoader singleton."""
    global _multimodal_loader
    if _multimodal_loader is None:
        from multimodal_loader import create_dataloader
        _multimodal_loader = {
            'train': create_dataloader(
                'train', data_dir, block_size, batch_size, device_type
            ),
            'val': create_dataloader(
                'val', data_dir, block_size, batch_size, device_type
            ),
        }
    return _multimodal_loader


_multimodal_iter = None


def get_batch(split: str):
    """
    Text-only batch loader (memmap, same as nanoGPT).
    Returns (x, y) each [batch_size, block_size].
    """
    fname = 'train.bin' if split == 'train' else 'val.bin'
    data  = np.memmap(os.path.join(data_dir, fname), dtype=np.uint16, mode='r')
    ix    = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([
        torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix
    ])
    y = torch.stack([
        torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64))
        for i in ix
    ])
    if device_type == 'cuda':
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def get_multimodal_batch(split: str):
    """
    Multimodal batch loader.
    Returns (x, y, modality_mask) if multimodal data available,
    otherwise falls back to text-only (x, y) with all-zero mask.
    """
    global _multimodal_iter

    if not use_multimodal_data:
        x, y = get_batch(split)
        mod_mask = torch.zeros_like(x, dtype=torch.int8)
        return x, y, mod_mask

    # use multimodal DataLoader
    loaders = _get_multimodal_loader()
    loader  = loaders[split]

    if _multimodal_iter is None or split == 'val':
        _multimodal_iter = iter(loader)

    try:
        batch = next(_multimodal_iter)
    except StopIteration:
        _multimodal_iter = iter(loader)
        batch = next(_multimodal_iter)

    from multimodal_loader import move_batch_to_device
    return move_batch_to_device(batch, device)


# ═══════════════════════════════════════════════════════════
# MODEL INIT
# ═══════════════════════════════════════════════════════════

iter_num      = 0
best_val_loss = 1e9

# read vocab_size from dataset metadata if available
meta_path       = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta.get('vocab_size', None)
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# LCM model args (no n_head)
model_args = dict(
    n_layer          = n_layer,
    n_embd           = n_embd,
    block_size       = block_size,
    bias             = bias,
    dropout          = dropout,
    ffn_ratio        = ffn_ratio,
    use_multimodal   = use_multimodal,
    image_vocab_size = image_vocab_size,
    vocab_size       = None,
)

if init_from == 'scratch':
    print("Initializing a new LCM from scratch")
    if meta_vocab_size is None:
        print("defaulting to vocab_size 50304 (50257 padded to multiple of 64)")
    model_args['vocab_size'] = (
        meta_vocab_size if meta_vocab_size is not None else 50304
    )
    lcmconf = LCMConfig(**model_args)
    model   = LCM(lcmconf)

elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    ckpt_path  = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    for k in [
        'n_layer', 'n_embd', 'block_size', 'bias',
        'vocab_size', 'ffn_ratio', 'use_multimodal', 'image_vocab_size',
    ]:
        model_args[k] = checkpoint_model_args[k]
    lcmconf    = LCMConfig(**model_args)
    model      = LCM(lcmconf)
    state_dict = checkpoint['model']
    # strip compile prefix if present
    unwanted = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted):
            state_dict[k[len(unwanted):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num      = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

else:
    raise ValueError(
        f"init_from='{init_from}' not supported. Use 'scratch' or 'resume'."
    )

# optionally shrink block_size
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size

model.to(device)

# ═══════════════════════════════════════════════════════════
# GRADIENT CHECKPOINTING
# ═══════════════════════════════════════════════════════════

if use_gradient_checkpointing and hasattr(model, 'transformer'):
    from torch.utils.checkpoint import checkpoint as torch_ckpt

    for blk in model.transformer.h:
        orig_fwd = blk.forward

        def _make_ckpt_fwd(f):
            def _wrapped(x, state=None):
                return torch_ckpt(f, x, state, use_reentrant=False)
            return _wrapped

        blk.forward = _make_ckpt_fwd(orig_fwd)

    print("gradient checkpointing enabled on all RecurrenceBlocks")

# ═══════════════════════════════════════════════════════════
# OPTIMIZER
# ═══════════════════════════════════════════════════════════

scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

optimizer = model.configure_optimizers(
    weight_decay, learning_rate, (beta1, beta2), device_type
)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None  # free memory

# ═══════════════════════════════════════════════════════════
# TORCH.COMPILE
# ═══════════════════════════════════════════════════════════

if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)

# ═══════════════════════════════════════════════════════════
# DDP / FSDP WRAP
# ═══════════════════════════════════════════════════════════

if ddp:
    if use_fsdp:
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            MixedPrecision,
            BackwardPrefetch,
        )
        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
        import functools

        auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy, min_num_params=1_000_000
        )
        mp_policy = MixedPrecision(
            param_dtype  = ptdtype,
            reduce_dtype = ptdtype,
            buffer_dtype = ptdtype,
        )
        model = FSDP(
            model,
            auto_wrap_policy  = auto_wrap_policy,
            mixed_precision   = mp_policy,
            backward_prefetch = BackwardPrefetch.BACKWARD_PRE,
            device_id         = ddp_local_rank,
            limit_all_gathers = True,
        )
        if master_process:
            print("FSDP wrapping complete")
    else:
        model = DDP(model, device_ids=[ddp_local_rank])

# ═══════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════

@torch.no_grad()
def estimate_loss():
    """Average loss over eval_iters batches for each split."""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            if use_multimodal_data:
                X, Y, _ = get_multimodal_batch(split)
            else:
                X, Y = get_batch(split)
            with ctx:
                # LCM forward → (logits, loss, states)
                _, loss, _ = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# ═══════════════════════════════════════════════════════════
# LEARNING RATE SCHEDULE
# ═══════════════════════════════════════════════════════════

def get_lr(it):
    """Cosine decay with linear warmup."""
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# ═══════════════════════════════════════════════════════════
# PRIME GRADIENT MODULATION (experimental)
# ═══════════════════════════════════════════════════════════

def _is_prime(n):
    """Fast primality test."""
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def apply_prime_grad_modulation(model, step, strength):
    """
    Scale gradient norms by (1 + strength) at prime-numbered steps.
    Research experiment. Disable by setting prime_grad_modulation = 0.0
    """
    if strength <= 0.0:
        return
    if _is_prime(step):
        raw = model.module if hasattr(model, 'module') else model
        for p in raw.parameters():
            if p.grad is not None:
                p.grad.data *= (1.0 + strength)

# ═══════════════════════════════════════════════════════════
# WANDB
# ═══════════════════════════════════════════════════════════

if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# ═══════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════

# fetch first batch
if use_multimodal_data:
    X, Y, mod_mask = get_multimodal_batch('train')
else:
    X, Y = get_batch('train')

t0             = time.time()
local_iter_num = 0
raw_model      = model.module if ddp else model
running_mfu    = -1.0

while True:

    # ── learning rate ──
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for pg in optimizer.param_groups:
        pg['lr'] = lr

    # ── evaluate & checkpoint ──
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        train_loss = losses['train']
        val_loss   = losses['val']
        train_ppl  = math.exp(train_loss) if train_loss < 20 else float('inf')
        val_ppl    = math.exp(val_loss)   if val_loss < 20   else float('inf')

        print(
            f"step {iter_num}: "
            f"train loss {train_loss:.4f} (ppl {train_ppl:.2f}), "
            f"val loss {val_loss:.4f} (ppl {val_ppl:.2f})"
        )

        if wandb_log:
            wandb.log({
                'iter':       iter_num,
                'train/loss': train_loss,
                'val/loss':   val_loss,
                'train/ppl':  train_ppl,
                'val/ppl':    val_ppl,
                'lr':         lr,
                'mfu':        running_mfu * 100,
            })

        if val_loss < best_val_loss or always_save_checkpoint:
            best_val_loss = val_loss
            if iter_num > 0:
                ckpt = {
                    'model':         raw_model.state_dict(),
                    'optimizer':     optimizer.state_dict(),
                    'model_args':    model_args,
                    'iter_num':      iter_num,
                    'best_val_loss': best_val_loss,
                    'config':        config,
                }
                ckpt_path = os.path.join(out_dir, 'ckpt.pt')
                print(f"saving checkpoint to {ckpt_path}")
                torch.save(ckpt, ckpt_path)

    if iter_num == 0 and eval_only:
        break

    # ── forward / backward with gradient accumulation ──
    for micro_step in range(gradient_accumulation_steps):

        # DDP: sync gradients only at last micro-step
        if ddp and not use_fsdp:
            model.require_backward_grad_sync = (
                micro_step == gradient_accumulation_steps - 1
            )

        with ctx:
            # LCM forward → (logits, loss, states)
            # states discarded during training (no caching)
            logits, loss, _ = model(X, Y)
            loss = loss / gradient_accumulation_steps

        # prefetch next batch while GPU computes backward
        if use_multimodal_data:
            X, Y, mod_mask = get_multimodal_batch('train')
        else:
            X, Y = get_batch('train')

        scaler.scale(loss).backward()

    # ── prime gradient modulation (before clip) ──
    if prime_grad_modulation > 0.0:
        apply_prime_grad_modulation(raw_model, iter_num, prime_grad_modulation)

    # ── gradient clipping ──
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    # ── optimizer step ──
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # ── timing & logging ──
    t1 = time.time()
    dt = t1 - t0
    t0 = t1

    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        ppl   = math.exp(lossf) if lossf < 20 else float('inf')

        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(
                batch_size * gradient_accumulation_steps, dt
            )
            running_mfu = (
                mfu if running_mfu == -1.0
                else 0.9 * running_mfu + 0.1 * mfu
            )

        prime_mark = " P" if (prime_grad_modulation > 0 and _is_prime(iter_num)) else ""

        print(
            f"iter {iter_num:>6d} | "
            f"loss {lossf:.4f} | "
            f"ppl {ppl:>8.2f} | "
            f"dt {dt * 1000:>7.1f}ms | "
            f"mfu {running_mfu * 100:.2f}%"
            f"{prime_mark}"
        )

    iter_num       += 1
    local_iter_num += 1
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()