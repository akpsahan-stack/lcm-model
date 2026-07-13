"""
LCM — Linear Consciousness Model  generation script.

Loads a trained LCM checkpoint and generates:
  1. Text only (standard autoregressive LM)
  2. Image only (prompt with description → image tokens → decode)
  3. Interleaved (text → image → text continuation)

Two generation modes:
  generate()      — recompute full sequence each step (simple, O(T) per step)
  generate_fast()  — use cached hidden states (O(1) per step, LCM advantage)

Usage:
  # Text generation
  python sample.py --out_dir=out-lcm --start="The meaning of life is"

  # Text generation (O(1) per step)
  python sample.py --out_dir=out-lcm --start="Once upon a time" --fast

  # Image generation from text prompt
  python sample.py --out_dir=out-lcm --start="a beautiful sunset" --mode=image --image_out=sunset.png

  # Interleaved generation
  python sample.py --out_dir=out-lcm --start="Look at this:" --mode=interleaved

  # Interactive chat
  python sample.py --out_dir=out-lcm --interactive
"""

import os
import sys
import pickle
import argparse
from pathlib import Path
from contextlib import nullcontext

import torch
import torch.nn.functional as F
import numpy as np

from model import LCMConfig, LCM

# ═══════════════════════════════════════════════════════════
# CLI ARGUMENTS
# ═══════════════════════════════════════════════════════════

def get_args():
    parser = argparse.ArgumentParser(
        description="LCM — Text & Image Generation"
    )

    # checkpoint
    parser.add_argument(
        '--out_dir', type=str, default='out-lcm',
        help='directory containing ckpt.pt'
    )
    parser.add_argument(
        '--device', type=str, default='cuda',
        help='cuda | cpu | cuda:0 | cuda:1'
    )
    parser.add_argument(
        '--dtype', type=str, default='bfloat16',
        help='float32 | bfloat16 | float16'
    )

    # generation mode
    parser.add_argument(
        '--mode', type=str, default='text',
        choices=['text', 'image', 'interleaved'],
        help='what to generate'
    )
    parser.add_argument(
        '--fast', action='store_true',
        help='use O(1)-per-step generation with state caching'
    )

    # text generation params
    parser.add_argument(
        '--start', type=str, default='\n',
        help='prompt text (or image description for image mode)'
    )
    parser.add_argument(
        '--num_samples', type=int, default=1,
        help='number of independent samples to generate'
    )
    parser.add_argument(
        '--max_new_tokens', type=int, default=500,
        help='max tokens to generate'
    )
    parser.add_argument(
        '--temperature', type=float, default=0.8,
        help='sampling temperature (0 = greedy, 1 = uniform)'
    )
    parser.add_argument(
        '--top_k', type=int, default=200,
        help='top-k filtering (0 = disabled)'
    )

    # image generation params
    parser.add_argument(
        '--image_out', type=str, default='generated.png',
        help='output path for generated image'
    )
    parser.add_argument(
        '--image_codec', type=str, default='simple_vq',
        help='image codec type: simple_vq | vqgan | random_codebook'
    )
    parser.add_argument(
        '--image_ckpt', type=str, default='checkpoints/vq.pt',
        help='path to VQ-VAE checkpoint'
    )
    parser.add_argument(
        '--image_size', type=int, default=256,
        help='generated image resolution'
    )
    parser.add_argument(
        '--image_codebook', type=int, default=1024,
        help='VQ codebook size'
    )

    # interactive mode
    parser.add_argument(
        '--interactive', action='store_true',
        help='interactive chat loop'
    )

    # compile
    parser.add_argument(
        '--compile', action='store_true',
        help='torch.compile for faster generation'
    )

    return parser.parse_args()

# ═══════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════

def load_model(args):
    """Load LCM from checkpoint."""
    ckpt_path = os.path.join(args.out_dir, 'ckpt.pt')
    if not os.path.exists(ckpt_path):
        print(f"ERROR: checkpoint not found at {ckpt_path}")
        print(f"Available checkpoints:")
        for p in Path(args.out_dir).glob('*.pt'):
            print(f"  {p}")
        sys.exit(1)

    print(f"Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=args.device)

    # reconstruct model
    model_args = checkpoint['model_args']
    lcmconf = LCMConfig(**model_args)
    model = LCM(lcmconf)

    state_dict = checkpoint['model']
    # strip compile prefix if present
    unwanted = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted):
            state_dict[k[len(unwanted):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.to(args.device)
    model.eval()

    print(
        f"LCM loaded: {model.get_num_params() / 1e6:.2f}M params, "
        f"n_layer={lcmconf.n_layer}, n_embd={lcmconf.n_embd}, "
        f"block_size={lcmconf.block_size}"
    )
    return model, lcmconf

# ═══════════════════════════════════════════════════════════
# TOKENIZER LOADING
# ═══════════════════════════════════════════════════════════

def load_text_tokenizer():
    """Load tiktoken GPT-2 BPE tokenizer."""
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    return enc

# ═══════════════════════════════════════════════════════════
# IMAGE CODEC LOADING
# ═══════════════════════════════════════════════════════════

def load_image_codec(args, config):
    """Load image codec for image generation."""
    try:
        from image_codec import ImageCodec
        codec = ImageCodec(
            codec_type     = args.image_codec,
            resolution     = args.image_size,
            codebook_size  = args.image_codebook,
            checkpoint_path = args.image_ckpt if os.path.exists(args.image_ckpt) else None,
            device         = args.device,
        )
        return codec
    except Exception as e:
        print(f"WARNING: Could not load image codec: {e}")
        print(f"Image generation will not be available.")
        return None

# ═══════════════════════════════════════════════════════════
# TEXT GENERATION
# ═══════════════════════════════════════════════════════════

@torch.no_grad()
def generate_text(model, enc, args, config):
    """
    Generate text continuations from a prompt.

    Uses generate() for simplicity or generate_fast() for O(1) per step.
    """
    prompt = args.start
    print(f"Prompt: {repr(prompt)}")
    print(f"Mode: {'fast (O(1))' if args.fast else 'standard'}")
    print(f"Temperature: {args.temperature}, Top-k: {args.top_k}")
    print("=" * 60)

    # encode prompt
    prompt_ids = enc.encode(prompt)
    if len(prompt_ids) == 0:
        prompt_ids = [enc.encode('\n')[0]]
    x = torch.tensor(prompt_ids, dtype=torch.long, device=args.device)
    x = x.unsqueeze(0).repeat(args.num_samples, 1)  # [num_samples, T]

    # generate
    if args.fast:
        y = model.generate_fast(
            x,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k if args.top_k > 0 else None,
            modality='text',
        )
    else:
        y = model.generate(
            x,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k if args.top_k > 0 else None,
            modality='text',
        )

    # decode and print
    for i in range(args.num_samples):
        tokens = y[i].tolist()
        text = enc.decode(tokens)
        print(f"\n--- Sample {i + 1} ---")
        print(text)

    print("\n" + "=" * 60)
    print(f"Generated {args.num_samples} sample(s), "
          f"{args.max_new_tokens} tokens each")

# ═══════════════════════════════════════════════════════════
# IMAGE GENERATION
# ═══════════════════════════════════════════════════════════

@torch.no_grad()
def generate_image(model, enc, codec, args, config):
    """
    Generate an image from a text description.

    Flow:
      1. Encode text prompt
      2. Run LCM forward to get hidden states
      3. Switch to image output head
      4. Autoregressively generate image tokens
      5. Decode tokens → image via VQ-VAE decoder

    NOTE: This requires the model to have been trained on
    interleaved text+image data for meaningful results.
    Without multimodal training, image tokens will be random.
    """
    prompt = args.start
    print(f"Image prompt: {repr(prompt)}")

    if codec is None:
        print("ERROR: Image codec not available. Cannot generate images.")
        return

    h, w = codec.grid_shape
    num_image_tokens = h * w
    print(f"Image grid: {h}×{w} = {num_image_tokens} tokens")
    print(f"Codec: {args.image_codec}, codebook: {args.image_codebook}")

    # encode text prompt
    prompt_ids = enc.encode(prompt)
    if len(prompt_ids) == 0:
        prompt_ids = enc.encode("an image of")

    # add <IMG_START> token (from multimodal_loader)
    from multimodal_loader import TextTokenizer, IMG_START_OFFSET
    text_tok = TextTokenizer()
    img_start = text_tok.img_start_id
    img_end   = text_tok.img_end_id

    # build input: text prompt + <IMG_START>
    x = torch.tensor(
        prompt_ids + [img_start],
        dtype=torch.long,
        device=args.device,
    ).unsqueeze(0)  # [1, T]

    # get hidden states from text processing
    _, _, states = model(x, modality='text')

    # now generate image tokens one by one using image head
    print(f"Generating {num_image_tokens} image tokens...")

    img_token_ids = []
    current = torch.tensor([[img_start]], dtype=torch.long, device=args.device)

    for t in range(num_image_tokens):
        logits, _, states = model(
            current, modality='image', hidden_states=states
        )
        logits = logits[:, -1, :] / args.temperature

        if args.top_k > 0:
            v, _ = torch.topk(logits, min(args.top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        token_id = next_token.item()
        img_token_ids.append(token_id)
        current = next_token

        if (t + 1) % 50 == 0:
            print(f"  {t + 1}/{num_image_tokens} tokens generated...")

    # decode image tokens → image
    print(f"Decoding {len(img_token_ids)} tokens to image...")
    image = codec.decode(img_token_ids, (h, w))
    image.save(args.image_out)
    print(f"Image saved to {args.image_out}")

    # report token statistics
    unique_tokens = len(set(img_token_ids))
    print(f"Unique tokens used: {unique_tokens}/{args.image_codebook}")

# ═══════════════════════════════════════════════════════════
# INTERLEAVED GENERATION
# ═══════════════════════════════════════════════════════════

@torch.no_grad()
def generate_interleaved(model, enc, codec, args, config):
    """
    Generate interleaved text and image content.

    Flow:
      1. Start with text prompt
      2. Generate text until <IMG_START> is produced (or force it)
      3. Generate image tokens
      4. Generate text continuation after <IMG_END>
      5. Repeat

    This demonstrates the any-to-any capability of LCM.
    """
    prompt = args.start
    print(f"Interleaved prompt: {repr(prompt)}")

    if codec is None:
        print("WARNING: No image codec. Falling back to text-only generation.")
        generate_text(model, enc, args, config)
        return

    from multimodal_loader import TextTokenizer
    text_tok = TextTokenizer()
    img_start = text_tok.img_start_id
    img_end   = text_tok.img_end_id

    h, w = codec.grid_shape
    tokens_per_image = h * w

    # encode prompt
    prompt_ids = enc.encode(prompt)
    x = torch.tensor(prompt_ids, dtype=torch.long, device=args.device).unsqueeze(0)

    # Phase 1: generate text (up to ~100 tokens or until we decide to insert image)
    text_phase_tokens = 100
    print(f"Phase 1: Generating {text_phase_tokens} text tokens...")

    if args.fast:
        text_out = model.generate_fast(
            x,
            max_new_tokens=text_phase_tokens,
            temperature=args.temperature,
            top_k=args.top_k if args.top_k > 0 else None,
            modality='text',
        )
    else:
        text_out = model.generate(
            x,
            max_new_tokens=text_phase_tokens,
            temperature=args.temperature,
            top_k=args.top_k if args.top_k > 0 else None,
            modality='text',
        )

    generated_text_ print(f"Text: {generated_text_1}")
    print("-" * 40)

    # Phase 2: generate image
    print(f"Phase 2: Generating image ({h}×{w} = {tokens_per_image} tokens)...")

    # feed the generated text + <IMG_START> to get states
    img_prompt = torch.cat([
        text_out,
        torch.tensor([[img_start]], dtype=torch.long, device=args.device),
    ], dim=1)

    _, _, states = model(img_prompt, modality='text')

    img_tokens = []
    current = torch.tensor([[img_start]], dtype=torch.long, device=args.device)
    for t in range(tokens_per_image):
        logits, _, states = model(
            current, modality='image', hidden_states=states
        )
        logits = logits[:, -1, :] / args.temperature
        if args.top_k > 0:
            v, _ = torch.topk(logits, min(args.top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        probs = F.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)
        img_tokens.append(next_tok.item())
        current = next_tok

    # decode image
    image = codec.decode(img_tokens, (h, w))
    image_path = args.image_out
    image.save(image_path)
    print(f"Image saved to {image_path}")

    # Phase 3: text continuation after image
    print(f"Phase 3: Generating text continuation...")

    # feed <IMG_END> and continue
    img_end_tok = torch.tensor([[img_end]], dtype=torch.long, device=args.device)
    _, _, states = model(img_end_tok, modality='text', hidden_states=states)

    # generate continuation
    cont_start = torch.tensor(
        enc.encode("\n"), dtype=torch.long, device=args.device
    ).unsqueeze(0)

    if args.fast:
        text_out_2 = model.generate_fast(
            cont_start,
            max_new_tokens=text_phase_tokens,
            temperature=args.temperature,
            top_k=args.top_k if args.top_k > 0 else None,
            modality='text',
            hidden_states=states if hasattr(model, 'generate_fast') else None,
        )
    else:
        # for standard generate, concatenate and re-generate
        full_so_far = torch.cat([img_prompt, img_end_tok, cont_start], dim=1)
        # crop to block_size if needed
        if full_so_far.size1 = enc.decode(text_out[0].tolist())
   (1) > config.block_size:
            full_so_far = full_so_far[:, -config.block_size:]
        text_out_2 = model.generate(
            full_so_far,
            max_new_tokens=text_phase_tokens,
            temperature=args.temperature,
            top_k=args.top_k if args.top_k > 0 else None,
            modality='text',
        )

    generated_text_2 = enc.decode(text_out_2[0].tolist())
    print(f"Contin════════uation: {generated_text_2}")

    print("\n" + "=" * 60)
    print(f"Interleaved generation complete.")
    print(f"  Text: {text_phase_tokens * 2} tokens")
    print(f"  Image: {image_path}")

# ═══════════════════════════════════════════════════════════
# INTERACTIVE MODE
# ═══════════════════════════════════════════════════════════

@torch.no_grad()
def interactive_loop(model, enc, args, config):
    """
    Interactive text generation loop.
    Type a prompt, get a continuation. Ctrl+C to exit.
    """
    print("=" * 60)
    print("LCM Interactive Generation")
    print(f"Model: {model.get_num_params() / 1e6:.1f}M params")
    print(f"Mode: {'fast (O(1))' if        # clean up: stop at first double args.fast else 'standard'}")
    print(f"Temperature: {args.temperature}, Top-k: {args.top_k}")
    print("Type your prompt and press Enter. Ctrl+C to exit.")
    print("=" * 60)

    hidden_states = None  # cached states for fast mode
    conversation = ""     # running conversation context

    while True:
        try:
            user_input = input("\nYou: ")
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input.strip():
            continue

        # special commands
        if user_input.strip().lower() == '/reset':
            conversation = ""
            hidden_states = None
            print("[Context reset]")
            continue

        if user_input.strip().lower().startswith('/temp '):
            try:
                args.temperature = float(user_input.strip().split()[1])
                print(f"[Temperature set to {args.temperature}]")
            except (ValueError, IndexError):
                print("[Usage: /temp 0.8]")
            continue

        if user_input.strip().lower().startswith('/tokens '):
            try:
                args.max_new_tokens = int(user_input.strip().split()[1])
                print(f"[Max tokens set to {args.max_new_tokens}]")
            except (ValueError, IndexError):
                print("[Usage: /tokens 500]")
            continue

        # append to conversation
        conversation += user_input + "\n"

        # encode
        input_ids = enc.encode(conversation)
        if len(input_ids) > config.block_size:
            # crop to fit block_size
            input_ids = input_ids[-(config.block_size - args.max_new_tokens):]
            conversation = enc.decode(input_ids)

        x = torch.tensor(input_ids, dtype=torch.long, device=args.device).unsqueeze(0)

        # generate
        if args.fast:
            # use cached states if available
            if hidden_states is not None:
                # only process the new tokens
                _, _, hidden_states = model(x, modality='text')
           .py                                      y = model.generate_fast(
                x,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k if args.top_k > 0 else None,
                modality='text',
            )
        else:
            y = model.generate(
                x,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k if args.top_k > 0 else None,
                modality='text',
            )

        # decode only the new tokens
        all_tokens = y[0].tolist()
        full_text = enc.decode(all_tokens)
        # extract just the generated part (after the prompt)
        generated = full_text[len(conversation):]
 newline
        if '\n\n' in generated:
            generated = generated.split('\n\n')[0]

        conversation += generated + "\n"

        print(f"\nLCM: {generated}")

# ═══════════════════════════════════════════════════════════
# MAIN
# ═════════════════════ codec, args, config)

        else:
            print(f"Unknown mode: {args.mode}")


if __name__ == '__main__':
    main()