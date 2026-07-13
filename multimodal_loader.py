"""
LCM — Multimodal Data Loader

Handles text tokenization (tiktoken BPE), image tokenization (VQGAN codebook),
interleaved sequence construction, and batch delivery.

Data flow:
  Raw (text + images)
       │
       ▼  prepare_multimodal_data()
  Pre-tokenized numpy arrays  (input_ids, targets, modality_mask)
       │
       ▼  MultimodalDataset + DataLoader
  Batches for train.py

Sequence format:
  <text tokens> <IMG_START> <image tokens> <IMG_END> <text tokens> ...
  modality:  0  0  0  0  -1   1  1  1  1  -1   0  0  0  0

Usage:
  # One-time data preparation:
  python multimodal_loader.py --prepare --data_dir data/multimodal

  # In train.py:
  from multimodal_loader import create_dataloader
  loader = create_dataloader('train', config)
  for batch in loader:
      input_ids, targets, modality_mask = batch
"""

import os
import json
import math
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ═══════════════════════════════════════════════════════════
# SPECIAL TOKENS
# ═══════════════════════════════════════════════════════════

# These ids are appended AFTER the base text vocabulary.
# If base vocab_size = 50257, then:
#   IMG_START = 50257
#   IMG_END   = 50258
# The model's text_wte must have (base_vocab + num_special) entries.

NUM_SPECIAL_TOKENS = 2
IMG_START_OFFSET   = 0   # base_vocab_size + 0
IMG_END_OFFSET     = 1   # base_vocab_size + 1

# Modality mask values
MOD_TEXT    =  0   # use text_wte
MOD_IMAGE   =  1   # use image_wte
MOD_SPECIAL = -1   # use text_wte (for <IMG_START>, <IMG_END>)


# ═══════════════════════════════════════════════════════════
# TEXT TOKENIZER  (tiktoken BPE — same as GPT-2/GPT-4)
# ═══════════════════════════════════════════════════════════

class TextTokenizer:
    """Wraps tiktoken for GPT-2 BPE encoding."""

    def __init__(self, encoding_name: str = "gpt2"):
        import tiktoken
        self.enc = tiktoken.get_encoding(encoding_name)
        self._vocab_size = self.enc.n_vocab  # 50257 for gpt2

    @property
    def base_vocab_size(self) -> int:
        """Vocab size WITHOUT special tokens."""
        return self._vocab_size

    @property
    def vocab_size(self) -> int:
        """Vocab size INCLUDING special tokens."""
        return self._vocab_size + NUM_SPECIAL_TOKENS

    @property
    def img_start_id(self) -> int:
        return self._vocab_size + IMG_START_OFFSET

    @property
    def img_end_id(self) -> int:
        return self._vocab_size + IMG_END_OFFSET

    def encode(self, text: str) -> List[int]:
        return self.enc.encode(text, allowed_special="all")

    def decode(self, ids: List[int]) -> str:
        # filter out special tokens that tiktoken doesn't know
        base_ids = [i for i in ids if i < self._vocab_size]
        return self.enc.decode(base_ids)


# ═══════════════════════════════════════════════════════════
# IMAGE TOKENIZER  (delegates to image_codec.py)
# ═══════════════════════════════════════════════════════════

class ImageTokenizer:
    """
    Wraps image_codec.ImageCodec to convert images ↔ discrete tokens.
    Tokens are in range [0, codebook_size).
    """

    def __init__(self, codec_type: str = "vqgan", **kwargs):
        from image_codec import ImageCodec
        self.codec = ImageCodec(codec_type=codec_type, **kwargs)
        self._vocab_size = self.codec.codebook_size

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def encode(self, image_path: str) -> List[int]:
        """Image file → flat list of discrete token ids."""
        return self.codec.encode(image_path)

    def decode(self, token_ids: List[int], grid_shape: Tuple[int, int]):
        """Token ids → PIL Image."""
        return self.codec.decode(token_ids, grid_shape)


# ═══════════════════════════════════════════════════════════
# INTERLEAVED SEQUENCE BUILDER
# ═══════════════════════════════════════════════════════════

def build_interleaved_sequence(
    text_ids: List[int],
    image_ids: Optional[List[int]],
    text_tok: TextTokenizer,
    block_size: int,
) -> Tuple[List[int], List[int]]:
    """
    Build one interleaved (text + image) sequence with markers.

    Returns
    -------
    token_ids   : List[int]  — mixed token ids  (length ≤ block_size)
    modality    : List[int]  — per-token modality (same length)
    """
    tokens: List[int]    = []
    modalities: List[int] = []

    # add text portion
    for tid in text_ids:
        if len(tokens) >= block_size:
            break
        tokens.append(tid)
        modalities.append(MOD_TEXT)

    # add image portion (if present)
    if image_ids is not None and len(tokens) < block_size:
        # <IMG_START>
        tokens.append(text_tok.img_start_id)
        modalities.append(MOD_SPECIAL)

        for iid in image_ids:
            if len(tokens) >= block_size - 1:  # reserve 1 for IMG_END
                break
            tokens.append(iid)
            modalities.append(MOD_IMAGE)

        # <IMG_END>
        if len(tokens) < block_size:
            tokens.append(text_tok.img_end_id)
            modalities.append(MOD_SPECIAL)

    return tokens, modalities


# ═══════════════════════════════════════════════════════════
# DATA PREPARATION  (run once before training)
# ═══════════════════════════════════════════════════════════

def prepare_multimodal_data(
    data_dir: str,
    block_size: int = 1024,
    val_split: float = 0.01,
    codec_type: str = "vqgan",
):
    """
    One-time preparation: raw data → pre-tokenized numpy arrays.

    Expected input structure:
        data_dir/
        ├── metadata.jsonl    # {"text": "...", "image": "img/001.jpg"} per line
        └── img/              # image files (referenced by metadata)

    Output (created in data_dir):
        ├── train.bin         # memmap uint16 — interleaved token ids
        ├── train_mod.npy     # npy int8     — modality mask
        ├── val.bin           # memmap uint16 — val tokens
        ├── val_mod.npy       # npy int8     — val modality
        └── meta.pkl          # vocab sizes, special token ids
    """
    data_dir = Path(data_dir)
    meta_path = data_dir / "metadata.jsonl"

    text_tok  = TextTokenizer("gpt2")
    img_tok   = None
    has_images = (data_dir / "img").exists()

    if has_images:
        try:
            img_tok = ImageTokenizer(codec_type=codec_type)
            print(f"Image tokenizer loaded: codebook_size={img_tok.vocab_size}")
        except Exception as e:
            print(f"WARNING: Could not load image codec ({e}). Text-only mode.")
            has_images = False

    # ── read metadata ──
    entries = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            entries.append(json.loads(line.strip()))
    print(f"Read {len(entries)} entries from {meta_path}")

    # ── tokenize all entries ──
    all_token_ids: List[int]    = []
    all_modalities: List[int]   = []

    for entry in entries:
        text = entry.get("text", "")
        img_path = entry.get("image", None)

        text_ids = text_tok.encode(text) if text else []

        image_ids = None
        if has_images and img_path and img_tok is not None:
            full_path = str(data_dir / img_path)
            if os.path.exists(full_path):
                try:
                    image_ids = img_tok.encode(full_path)
                except Exception:
                    image_ids = None

        toks, mods = build_interleaved_sequence(
            text_ids, image_ids, text_tok, block_size
        )
        all_token_ids.extend(toks)
        all_modalities.extend(mods)

        # insert separator (newline token = 628 for GPT-2)
        all_token_ids.append(628)
        all_modalities.append(MOD_TEXT)

    total_tokens = len(all_token_ids)
    print(f"Total tokens: {total_tokens:,}")

    # ── split into train / val ──
    arr_ids  = np.array(all_token_ids,  dtype=np.uint16)
    arr_mod  = np.array(all_modalities, dtype=np.int8)
    split    = int(total_tokens * (1.0 - val_split))

    # ── save ──
    for name, ids_arr, mod_arr, length in [
        ("train", arr_ids[:split], arr_mod[:split], split),
        ("val",   arr_ids[split:], arr_mod[split:], total_tokens - split),
    ]:
        bin_path  = data_dir / f"{name}.bin"
        mod_path  = data_dir / f"{name}_mod.npy"

        # memmap for token ids (same as nanoGPT)
        fp = np.memmap(str(bin_path), dtype=np.uint16, mode='w+', shape=(length,))
        fp[:] = ids_arr[:]
        fp.flush()

        np.save(str(mod_path), mod_arr)
        print(f"Saved {name}: {length:,} tokens → {bin_path}")

    # ── save metadata ──
    meta = {
        "text_vocab_size":    text_tok.base_vocab_size,
        "image_vocab_size":   img_tok.vocab_size if img_tok else 0,
        "total_vocab_size":   text_tok.vocab_size,
        "img_start_id":       text_tok.img_start_id,
        "img_end_id":         text_tok.img_end_id,
        "block_size":         block_size,
        "num_special_tokens": NUM_SPECIAL_TOKENS,
        "has_images":         has_images,
    }
    with open(data_dir / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)
    print(f"Saved meta.pkl: {meta}")


# ═══════════════════════════════════════════════════════════
# PYTORCH DATASET
# ═══════════════════════════════════════════════════════════

class MultimodalDataset(Dataset):
    """
    Loads pre-tokenized interleaved data from disk.
    Returns (input_ids, targets, modality_mask) per sample.

    If modality mask file doesn't exist, returns text-only
    (all zeros mask).
    """

    def __init__(
        self,
        split: str,
        data_dir: str,
        block_size: int,
    ):
        self.block_size = block_size

        bin_path = os.path.join(data_dir, f"{split}.bin")
        mod_path = os.path.join(data_dir, f"{split}_mod.npy")

        # load token ids as memmap (memory efficient)
        self.data = np.memmap(bin_path, dtype=np.uint16, mode='r')
        self.n_tokens = len(self.data)

        # load modality mask if available
        if os.path.exists(mod_path):
            self.modality = np.load(mod_path)
            assert len(self.modality) == self.n_tokens, (
                f"Token/modality length mismatch: "
                f"{self.n_tokens} vs {len(self.modality)}"
            )
        else:
            self.modality = None  # text-only fallback

        # number of valid starting positions
        self.n_samples = self.n_tokens - block_size
        if self.n_samples <= 0:
            raise ValueError(
                f"Dataset too small: {self.n_tokens} tokens, "
                f"need > {block_size}"
            )

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        input_ids     : [block_size] int64
        targets       : [block_size] int64   (shifted right by 1)
        modality_mask : [block_size] int8    (0=text, 1=image, -1=special)
        """
        chunk = self.data[idx : idx + self.block_size + 1]

        input_ids = torch.from_numpy(chunk[:-1].astype(np.int64))   # [block_size]
        targets   = torch.from_numpy(chunk[1:].astype(np.int64))    # [block_size]

        if self.modality is not None:
            mod = self.modality[idx : idx + self.block_size]
            modality_mask = torch.from_numpy(mod.astype(np.int8))
        else:
            modality_mask = torch.zeros(self.block_size, dtype=torch.int8)

        return input_ids, targets, modality_mask


# ═══════════════════════════════════════════════════════════
# DATALOADER FACTORY
# ═══════════════════════════════════════════════════════════

def create_dataloader(
    split: str,
    data_dir: str,
    block_size: int,
    batch_size: int,
    device_type: str = 'cuda',
    num_workers: int = 0,
) -> DataLoader:
    """
    Create a DataLoader for LCM training.

    Returns batches of (input_ids, targets, modality_mask).
    """
    dataset = MultimodalDataset(split, data_dir, block_size)

    def _collate(batch):
        """Stack and optionally pin memory."""
        input_ids     = torch.stack([b[0] for b in batch])
        targets       = torch.stack([b[1] for b in batch])
        modality_mask = torch.stack([b[2] for b in batch])

        if device_type == 'cuda':
            input_ids     = input_ids.pin_memory()
            targets       = targets.pin_memory()
            modality_mask = modality_mask.pin_memory()

        return input_ids, targets, modality_mask

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=_collate,
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )

    print(
        f"DataLoader [{split}]: "
        f"{len(dataset):,} samples, "
        f"{len(loader):,} batches, "
        f"block_size={block_size}"
    )
    return loader


def move_batch_to_device(batch, device):
    """Move (input_ids, targets, modality_mask) to device."""
    input_ids, targets, modality_mask = batch
    return (
        input_ids.to(device, non_blocking=True),
        targets.to(device, non_blocking=True),
        modality_mask.to(device, non_blocking=True),
    )


# ═══════════════════════════════════════════════════════════
# VISUALIZATION / DEBUG
# ═══════════════════════════════════════════════════════════

def visualize_sequence(
    input_ids: torch.Tensor,
    modality_mask: torch.Tensor,
    text_tok: TextTokenizer,
    max_tokens: int = 80,
):
    """Print a human-readable view of one interleaved sequence."""
    ids  = input_ids[:max_tokens].tolist()
    mods = modality_mask[:max_tokens].tolist()

    lines = []
    for tid, mod in zip(ids, mods):
        if mod == MOD_IMAGE:
            lines.append(f"[IMG:{tid}]")
        elif mod == MOD_SPECIAL:
            if tid == text_tok.img_start_id:
                lines.append("<IMG>")
            elif tid == text_tok.img_end_id:
                lines.append("</IMG>")
            else:
                lines.append(f"<S:{tid}>")
        else:
            decoded = text_tok.decode([tid])
            lines.append(decoded.replace("\n", "\\n"))

    print(" ".join(lines))


# ═══════════════════════════════════════════════════════════
# TEXT-ONLY CONVENIENCE  (for initial training without images)
# ═══════════════════════════════════════════════════════════

def prepare_text_only_data(data_dir: str, block_size: int = 1024):
    """
    Fallback: prepare text-only data (no images).
    Reads all .txt files in data_dir/texts/ and tokenizes.

    Output:
        data_dir/train.bin, data_dir/val.bin, data_dir/meta.pkl
    """
    text_dir = Path(data_dir) / "texts"
    if not text_dir.exists():
        raise FileNotFoundError(f"Text directory not found: {text_dir}")

    text_tok = TextTokenizer("gpt2")
    all_ids: List[int] = []

    for txt_file in sorted(text_dir.glob("*.txt")):
        with open(txt_file, "r", encoding="utf-8") as f:
            text = f.read()
        all_ids.extend(text_tok.encode(text))
        all_ids.append(628)  # newline separator

    total = len(all_ids)
    print(f"Text-only: {total:,} tokens from {len(list(text_dir.glob('*.txt')))} files")

    arr = np.array(all_ids, dtype=np.uint16)
    split = int(total * 0.99)

    for name, sub_arr, length in [
        ("train", arr[:split], split),
        ("val",   arr[split:], total - split),
    ]:
        fp = np.memmap(
            os.path.join(data_dir, f"{name}.bin"),
            dtype=np.uint16, mode='w+', shape=(length,),
        )
        fp[:] = sub_arr[:]
        fp.flush()

    meta = {
        "vocab_size": text_tok.vocab_size,
        "has_images": False,
    }
    with open(os.path.join(data_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)
    print(f"Saved meta.pkl")


# ═══════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LCM Multimodal Data Loader")
    parser.add_argument("--prepare", action="store_true", help="Run data preparation")
    parser.add_argument("--text_only", action="store_true", help="Text-only mode")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--val_split", type=float, default=0.01)
    parser.add_argument("--codec", type=str, default="vqgan")
    parser.add_argument("--debug", action="store_true", help="Load & visualize a batch")
    args = parser.parse_args()

    if args.prepare:
        if args.text_only:
            prepare_text_only_data(args.data_dir, args.block_size)
        else:
            prepare_multimodal_data(
                args.data_dir, args.block_size, args.val_split, args.codec
            )

    elif args.debug:
        # load one batch and visualize
        loader = create_dataloader(
            "train", args.data_dir, args.block_size, batch_size=1
        )
        batch = next(iter(loader))
        ids, targets, mods = batch
        text_tok = TextTokenizer("gpt2")
        print(f"input_ids shape:     {ids.shape}")
        print(f"targets shape:       {targets.shape}")
        print(f"modality_mask shape: {mods.shape}")
        print(f"text tokens:   {(mods == 0).sum().item()}")
        print(f"image tokens:  {(mods == 1).sum().item()}")
        print(f"special tokens:{(mods == -1).sum().item()}")
        print("---")
        visualize_sequence(ids[0], mods[0], text_tok)
    else:
        parser.print_help()