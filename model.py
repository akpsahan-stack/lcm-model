"""
LCM — Linear Consciousness Model
Full architecture definition, single file.

Core Innovation
───────────────
Replaces O(n²) self-attention with O(1) recurrent linear scan:

    h_t = gate_t ⊙ (A ⊙ h_{t-1}) + (1 − gate_t) ⊙ (B · x_t)
    y_t = C · h_t

Three mathematical pillars:
  1. A (diagonal) — Ramanujan ellipse perimeter initialization
  2. gate_t       — prime-number-aware selective gate (Sati / mindfulness)
  3. Per-dim scale — ellipse eccentricity → diverse memory lifetimes

Buddhist Viññāṇa analogy:
  h_{t-1} = departing consciousness  (atīta citta)
  x_t     = sense-door contact       (phassa)
  gate_t  = mindfulness factor       (sati)
  h_t     = arising consciousness    (uppāda citta)

References:
  - Ramanujan ellipse perimeter approximation
  - Mamba (S6) selective state-space model
  - RWKV linear attention replacement
"""

import math
import inspect
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
from torch.nn import functional as F


# ─────────────────────────────────────────────────────────
# LayerNorm
# ─────────────────────────────────────────────────────────

class LayerNorm(nn.Module):
    """LayerNorm with optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


# ─────────────────────────────────────────────────────────
# Mathematical Foundations
# ─────────────────────────────────────────────────────────

def sieve_of_eratosthenes(n: int) -> torch.Tensor:
    """Boolean tensor [n]: True at prime indices."""
    p = torch.ones(n + 1, dtype=torch.bool)
    p[0] = p[1] = False
    for i in range(2, int(n**0.5) + 1):
        if p[i]:
            p[i * i :: i] = False
    return p[:n]


def ramanujan_perimeter(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Ramanujan's first approximation for ellipse perimeter.

        h = ((a − b) / (a + b))²
        P ≈ π(a + b)(1 + 3h / (10 + √(4 − 3h)))
    """
    aa, bb = a.abs(), b.abs()
    h = ((aa - bb) / (aa + bb + 1e-8)) ** 2
    P = math.pi * (aa + bb) * (
        1.0 + 3.0 * h / (10.0 + torch.sqrt((4.0 - 3.0 * h).clamp(min=0.01)))
    )
    return P


def build_ellipse_dynamics(
    n_embd: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Per-dimension memory dynamics from ellipse geometry.

    Each dimension i is assigned an ellipse (a_i, b_i).
    Ramanujan perimeter P_i controls how fast that dimension's
    hidden-state evolves:

        large P_i  →  fast update  (short memory, high responsiveness)
        small P_i  →  slow update  (long  memory, low  responsiveness)

    The (a, b) values sweep sinusoidally across dimensions so every
    dimension gets a unique "eccentricity" — giving the model a rich
    spectrum of memory lifetimes without any extra parameters.

    Returns (decay_scale, input_scale, perimeter) — each [n_embd].
    """
    t = torch.arange(n_embd, dtype=torch.float32) / n_embd
    # parametric sweep: different phase per axis for diversity
    a = (0.5 + 0.5 * torch.sin(2 * math.pi * t)).clamp(0.1, 1.0)
    b = (0.5 + 0.5 * torch.cos(2 * math.pi * t + math.pi / 4)).clamp(0.1, 1.0)
    P = ramanujan_perimeter(a, b)
    lo, hi = P.min(), P.max()
    # normalise into [0.5, 0.99] → guarantees |A| < 1 (stable recurrence)
    decay = 0.5 + 0.49 * (P - lo) / (hi - lo + 1e-8)
    return decay, 1.0 - decay, P


# ─────────────────────────────────────────────────────────
# RecurrentLinearScan  (replaces CausalSelfAttention)
# ─────────────────────────────────────────────────────────

class RecurrentLinearScan(nn.Module):
    """
    O(1)-memory recurrent scan replacing O(n²) Q×Kᵀ attention.

    Forward equation:

        h_t = gate_t ⊙ (A ⊙ h_{t-1}) + (1 − gate_t) ⊙ (B · x_t)
        y_t = C · h_t

    A  = diag(exp(A_log))              diagonal, Ramanujan-initialised
    B  = self.B_proj                   learned input  projection
    C  = self.C_proj                   learned output projection
    gate_t = σ(W_gate · x_t + boost · 𝟙[is_prime(t)])

    Only h_t ∈ ℝ^D is carried between steps → O(1) memory per token.
    """

    def __init__(self, config):
        super().__init__()
        D = config.n_embd
        self.n_embd = D

        # ── projections ──
        self.B_proj    = nn.Linear(D, D, bias=config.bias)
        self.C_proj    = nn.Linear(D, D, bias=config.bias)
        self.gate_proj = nn.Linear(D, D, bias=config.bias)
        self.gate_norm = nn.LayerNorm(D)

        # ── diagonal A in log-space (numerical stability) ──
        self.A_log = nn.Parameter(torch.zeros(D))

        # ── fixed ellipse dynamics (not learnable) ──
        decay, inp, P = build_ellipse_dynamics(D)
        self.register_buffer("ellipse_decay", decay)
        self.register_buffer("ellipse_input", inp)
        self.register_buffer("ellipse_perimeter", P)

        # ── prime-number gate modulation ──
        self.register_buffer(
            "prime_mask", sieve_of_eratosthenes(config.block_size)
        )
        self.prime_boost = nn.Parameter(torch.tensor(0.5))

        self.drop = nn.Dropout(config.dropout)

        # ── initialise A from Ramanujan; bias gate toward input ──
        with torch.no_grad():
            self.A_log.copy_(torch.log(decay + 1e-8))
        nn.init.constant_(self.gate_proj.bias, -1.0)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x     : [B, T, D]
        state : [B, D] cached hidden state or None

        Returns
        -------
        y           : [B, T, D]
        final_state : [B, D]  (detached — for generation caching)
        """
        B, T, D = x.shape

        # ── parallel gate computation ──
        g = torch.sigmoid(self.gate_proj(self.gate_norm(x)))      # [B,T,D]
        pf = self.prime_mask[:T].float().view(1, T, 1)            # [1,T,1]
        g = (g + self.prime_boost * pf).clamp(0.0, 1.0)

        # ── input contribution, ellipse-scaled ──
        Bx = self.B_proj(x) * self.ellipse_input.view(1, 1, D)   # [B,T,D]

        # ── diagonal state transition ──
        A = self.A_log.exp().clamp(max=0.999)                     # [D]

        # ── sequential scan ──
        # NOTE: parallel associative scan (O(log T)) is possible via
        #       CUDA kernel — see Mamba selective_scan_cuda for reference.
        h = state if state is not None else x.new_zeros(B, D)
        outs: List[torch.Tensor] = []
        for t in range(T):
            h = g[:, t] * (A * h) + (1.0 - g[:, t]) * Bx[:, t]
            outs.append(self.C_proj(h))

        y = torch.stack(outs, dim=1)                              # [B,T,D]
        return y, h.detach()


# ─────────────────────────────────────────────────────────
# MLP  (SwiGLU — LLaMA-style gated FFN)
# ─────────────────────────────────────────────────────────

class MLP(nn.Module):
    """Feed-forward with SwiGLU: Silu(gate) ⊙ up → down."""

    def __init__(self, config):
        super().__init__()
        h = int(config.n_embd * config.ffn_ratio)
        self.gate_proj = nn.Linear(config.n_embd, h, bias=config.bias)
        self.up_proj   = nn.Linear(config.n_embd, h, bias=config.bias)
        self.down_proj = nn.Linear(h, config.n_embd, bias=config.bias)
        self.drop      = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.drop(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


# ─────────────────────────────────────────────────────────
# Block  (Pre-norm Recurrence Block)
# ─────────────────────────────────────────────────────────

class Block(nn.Module):
    """
    LayerNorm → RecurrentLinearScan → residual
    LayerNorm → MLP                  → residual
    """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, config.bias)
        self.scan = RecurrentLinearScan(config)
        self.ln_2 = LayerNorm(config.n_embd, config.bias)
        self.mlp  = MLP(config)

    def forward(self, x, state=None):
        scan_out, new_state = self.scan(self.ln_1(x), state)
        x = x + scan_out
        x = x + self.mlp(self.ln_2(x))
        return x, new_state


# ─────────────────────────────────────────────────────────
# LCMConfig
# ─────────────────────────────────────────────────────────

@dataclass
class LCMConfig:
    block_size:       int   = 1024
    vocab_size:       int   = 50304       # padded to nearest 64
    image_vocab_size: int   = 1024        # VQGAN codebook size
    n_layer:          int   = 12
    n_embd:           int   = 768         # hidden dimension
    dropout:          float = 0.0
    bias:             bool  = True
    ffn_ratio:        float = 4.0         # FFN hidden = n_embd × ffn_ratio
    use_multimodal:   bool  = True        # enable dual input/output heads


# ─────────────────────────────────────────────────────────
# LCM  —  Linear Consciousness Model
# ─────────────────────────────────────────────────────────

class LCM(nn.Module):
    """
    Any-to-Any Linear Consciousness Model.

    Input  : text tokens  OR  image tokens   (shared embedding space)
    Output : text logits   AND/OR image logits (dual output heads)
    Memory : O(1) per token                   (only h_t carried forward)
    """

    # ────────────── __init__ ──────────────

    def __init__(self, config: LCMConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # ── Dual input embeddings (shared vector space) ──
        embd_dict = dict(
            text_wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe      = nn.Embedding(config.block_size, config.n_embd),
            drop     = nn.Dropout(config.dropout),
            h        = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f     = LayerNorm(config.n_embd, config.bias),
        )
        if config.use_multimodal:
            embd_dict["image_wte"] = nn.Embedding(
                config.image_vocab_size, config.n_embd
            )
        self.transformer = nn.ModuleDict(embd_dict)

        # ── Dual output heads ──
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.image_head: Optional[nn.Linear] = (
            nn.Linear(config.n_embd, config.image_vocab_size, bias=False)
            if config.use_multimodal
            else None
        )

        # ── Weight tying ──
        #   text embedding  ↔  text output head
        #   image embedding ↔  image output head
        self.transformer.text_wte.weight = self.lm_head.weight
        if self.image_head is not None:
            self.transformer.image_wte.weight = self.image_head.weight

        # ── Initialise all weights ──
        self.apply(self._init_weights)

        # special scaled init for residual projections (per GPT-2 paper)
        for pn, p in self.named_parameters():
            if pn.endswith("down_proj.weight") or pn.endswith("C_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        n = self.get_num_params()
        print("LCM parameters: %.2fM" % (n / 1e6))

    # ────────────── helpers ──────────────

    def get_num_params(self, non_embedding=True):
        """Return parameter count; subtract position embeddings by default."""
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.transformer.wpe.weight.numel()
        return n

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def crop_block_size(self, block_size):
        """Model surgery: decrease block_size if needed."""
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(
            self.transformer.wpe.weight[:block_size]
        )
        for blk in self.transformer.h:
            blk.scan.prime_mask = blk.scan.prime_mask[:block_size]

    # ────────────── forward ──────────────

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        modality: str = "text",
        hidden_states: Optional[List[torch.Tensor]] = None,
    ):
        """
        Parameters
        ----------
        idx           : [B, T]  token ids
        targets       : [B, T]  target ids (training) or None (inference)
        modality      : 'text'  |  'image'
        hidden_states : list[n_layer] × [B, D]  cached states, or None

        Returns
        -------
        logits : [B, T, V]   V = vocab_size or image_vocab_size
        loss   : scalar | None
        states : list[n_layer] × [B, D]   (for generation caching)
        """
        device = idx.device
        B, T = idx.size()
        assert T <= self.config.block_size, (
            f"Sequence length {T} exceeds block_size {self.config.block_size}"
        )
        pos = torch.arange(0, T, dtype=torch.long, device=device)

        # ── 1. modality-aware embedding ──
        if modality == "image" and "image_wte" in self.transformer:
            tok_emb = self.transformer.image_wte(idx)
        else:
            tok_emb = self.transformer.text_wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        # ── 2. recurrent processing through all layers ──
        states_in = hidden_states or [None] * self.config.n_layer
        states_out: List[torch.Tensor] = []
        for i, blk in enumerate(self.transformer.h):
            x, s = blk(x, states_in[i])
            states_out.append(s)

        x = self.transformer.ln_f(x)

        # ── 3. output head (text or image) ──
        if modality == "image" and self.image_head is not None:
            head = self.image_head
        else:
            head = self.lm_head

        if targets is not None:
            logits = head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            # inference: only project the last position
            logits = head(x[:, [-1], :])
            loss = None

        return logits, loss, states_out

    # ────────────── optimizer ──────────────

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        Create AdamW with two param groups:
          • 2-D tensors (weights + embeddings) → weight decay
          • 1-D tensors (biases + layernorms)  → no decay
        """
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params   = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params,   "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay   = sum(p.numel() for p in decay_params)
        num_nodecay = sum(p.numel() for p in nodecay_params)
        print(f"decayed   : {len(decay_params):>4} tensors, {num_decay:>12,} params")
        print(f"non-decay : {len(nodecay_params):>4} tensors, {num_nodecay:>12,} params")

        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra
        )
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    # ────────────── MFU estimation ──────────────

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """
        Estimate model FLOPs utilization vs A100 bf16 peak (312 TFLOPS).
        For recurrence, per-token FLOPS ≈ 6N (no attention O(n²) term).
        See PaLM paper Appendix B: https://arxiv.org/abs/2204.02311
        """
        N = self.get_num_params()
        T = self.config.block_size
        flops_per_token  = 6 * N
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter   = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved   = flops_per_iter / dt
        return flops_achieved / 312e12

    # ────────────── generation (standard) ──────────────

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        modality: str = "text",
    ):
        """
        Standard autoregressive generation.
        Recomputes full sequence each step — simple but O(T) per step.
        For O(1)-per-step, use generate_fast().
        """
        for _ in range(max_new_tokens):
            # crop to block_size if needed
            idx_cond = (
                idx if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )
            logits, _, _ = self(idx_cond, modality=modality)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    # ────────────── generation (O(1) per step) ──────────────

    @torch.no_grad()
    def generate_fast(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        modality: str = "text",
    ):
        """
        O(1)-per-step generation with hidden-state caching.
        Only the LAST token is processed each step; all prior
        context lives in the cached h_t states (constant memory).
        This is the key advantage of the LCM recurrence.
        """
        # 1. encode the full prompt → get initial hidden states
        _, _, states = self(idx, modality=modality)

        for _ in range(max_new_tokens):
            # 2. only feed the newest token with cached states
            logits, _, states = self(
                idx[:, -1:], modality=modality, hidden_states=states
            )
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx