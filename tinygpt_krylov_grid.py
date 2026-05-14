#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch",
#   "pandas",
# ]
# ///
"""
TinyGPT Krylov synthetic experiments.

This is a deliberately small, hackable experiment runner for comparing:
  - causal softmax attention
  - causal positive Krylov / polynomial recurrent attention
  - simple hybrid stacks such as krylov -> softmax

It supports both one-off training runs and grid runs that write CSV/JSONL summaries.
"""

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Vocabulary
# -----------------------------

SPECIAL = ["<pad>", "<bos>", "<eos>", "?", ":", ";", "=", ">"]
KEYS = [chr(ord("A") + i) for i in range(16)]
VALS = [str(i) for i in range(16)]
HOPS = [str(i) for i in range(1, 8)]

VOCAB = SPECIAL + KEYS + VALS
STOI = {s: i for i, s in enumerate(VOCAB)}
ITOS = {i: s for s, i in STOI.items()}

PAD_ID = STOI["<pad>"]
BOS_ID = STOI["<bos>"]
EOS_ID = STOI["<eos>"]


def encode(tokens: List[str]) -> List[int]:
    return [STOI[t] for t in tokens]


def decode(ids: List[int]) -> List[str]:
    return [ITOS[int(i)] for i in ids]


# -----------------------------
# Synthetic grammars
# -----------------------------

@dataclass
class Example:
    ids: List[int]
    answer_pos: int
    answer_id: int
    text: str


def make_kv_example(
    n_pairs: int = 6,
    seq_total_len: int = 32,
) -> Example:
    """
    Example:
        <bos> C : 4 ; F : 3 ; D : 0 ; ? F = 3 <eos>

    Values are sampled without replacement to keep the retrieval target clean.
    """
    keys = random.sample(KEYS, n_pairs)
    vals = random.sample(VALS, n_pairs)
    mapping = dict(zip(keys, vals))

    query_key = random.choice(keys)
    answer = mapping[query_key]

    tokens = ["<bos>"]
    for k, v in zip(keys, vals):
        tokens += [k, ":", v, ";"]

    tokens += ["?", query_key, "=", answer, "<eos>"]

    ids = encode(tokens)
    if len(ids) > seq_total_len:
        raise ValueError(f"Example too long: {len(ids)} > {seq_total_len}: {' '.join(tokens)}")

    eq_pos = tokens.index("=")
    answer_pos = eq_pos
    ids = ids + [PAD_ID] * (seq_total_len - len(ids))

    return Example(ids=ids, answer_pos=answer_pos, answer_id=STOI[answer], text=" ".join(tokens))


def make_path_example(
    n_edges: int = 6,
    hop: int = 3,
    seq_total_len: int = 40,
    hide_hop_token: bool = False,
) -> Example:
    """
    Example with hop token:
        <bos> A > B ; B > C ; C > D ; ? A 3 = D <eos>

    Example without hop token:
        <bos> A > B ; B > C ; C > D ; ? A = D <eos>

    The no-hop-token form is often cleaner for fixed-hop diagnostics.
    """
    chain_len = n_edges + 1
    chain = random.sample(KEYS, chain_len)

    hop = min(hop, n_edges)
    start_idx = random.randint(0, n_edges - hop)
    start = chain[start_idx]
    answer = chain[start_idx + hop]

    tokens = ["<bos>"]
    for i in range(n_edges):
        tokens += [chain[i], ">", chain[i + 1], ";"]

    if hide_hop_token:
        tokens += ["?", start, "=", answer, "<eos>"]
    else:
        tokens += ["?", start, str(hop), "=", answer, "<eos>"]

    ids = encode(tokens)
    if len(ids) > seq_total_len:
        raise ValueError(f"Example too long: {len(ids)} > {seq_total_len}: {' '.join(tokens)}")

    eq_pos = tokens.index("=")
    answer_pos = eq_pos
    ids = ids + [PAD_ID] * (seq_total_len - len(ids))

    return Example(ids=ids, answer_pos=answer_pos, answer_id=STOI[answer], text=" ".join(tokens))


def make_batch(
    batch_size: int,
    grammar: str,
    seq_total_len: int,
    n_pairs: int,
    n_edges: int,
    hop: int,
    hide_hop_token: bool,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    examples: List[Example] = []

    for _ in range(batch_size):
        if grammar == "kv":
            ex = make_kv_example(n_pairs=n_pairs, seq_total_len=seq_total_len)
        elif grammar == "path":
            ex = make_path_example(
                n_edges=n_edges,
                hop=hop,
                seq_total_len=seq_total_len,
                hide_hop_token=hide_hop_token,
            )
        else:
            raise ValueError(f"Unknown grammar: {grammar}")
        examples.append(ex)

    ids = torch.tensor([ex.ids for ex in examples], dtype=torch.long, device=device)
    x = ids[:, :-1]
    y = ids[:, 1:]

    answer_pos = torch.tensor([ex.answer_pos for ex in examples], dtype=torch.long, device=device)
    answer_id = torch.tensor([ex.answer_id for ex in examples], dtype=torch.long, device=device)
    texts = [ex.text for ex in examples]

    return x, y, answer_pos, answer_id, texts


# -----------------------------
# Attention layers
# -----------------------------

def positive_l1_features(x: torch.Tensor, beta: float = 0.5, eps: float = 1e-4) -> torch.Tensor:
    """x: [B, H, T, D]. Returns positive L1-normalized features over D."""
    phi = F.softplus(beta * x) + eps
    return phi / (phi.sum(dim=-1, keepdim=True) + eps)


class CausalKrylovAttention(nn.Module):
    """
    Causal positive Krylov / polynomial mixer.

    c_0 = 0 by default, so raw current value is not included inside the
    normalized mixer; the transformer residual carries current-token content.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        m: int = 3,
        beta: float = 0.5,
        rho: float = 1.0,
        state_decay: float = 0.95,
        eps: float = 1e-6,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.m = m
        self.beta = beta
        self.rho = rho
        self.state_decay = state_decay
        self.eps = eps

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def coeffs(self, device, dtype) -> torch.Tensor:
        c = torch.zeros(self.m + 1, device=device, dtype=dtype)
        for r in range(1, self.m + 1):
            c[r] = self.rho ** (r - 1)
        return c

    def forward(self, x: torch.Tensor, return_diagnostics: bool = False):
        B, T, C = x.shape
        H = self.n_heads
        D = self.d_head

        q = self.wq(x).view(B, T, H, D).transpose(1, 2)  # [B,H,T,D]
        k = self.wk(x).view(B, T, H, D).transpose(1, 2)
        v = self.wv(x).view(B, T, H, D).transpose(1, 2)

        u_all = positive_l1_features(q, beta=self.beta)
        w_all = positive_l1_features(k, beta=self.beta)
        coeffs = self.coeffs(x.device, x.dtype)

        S = [x.new_zeros(B, H, D, D) for _ in range(self.m)]
        g = [x.new_zeros(B, H, D) for _ in range(self.m)]

        ys = []
        z_stats = []
        order_norm_stats = []

        for t in range(T):
            u = u_all[:, :, t, :]
            w = w_all[:, :, t, :]
            vt = v[:, :, t, :]

            Y_orders = [vt]
            Z_orders = [torch.ones(B, H, device=x.device, dtype=x.dtype)]
            new_S = list(S)
            new_g = list(g)

            for r in range(self.m):
                Y_prev = Y_orders[r]
                Z_prev = Z_orders[r]

                outer = w.unsqueeze(-1) * Y_prev.unsqueeze(-2)
                S_r = self.state_decay * S[r] + outer
                g_r = self.state_decay * g[r] + w * Z_prev.unsqueeze(-1)

                Y_next = torch.einsum("bhij,bhi->bhj", S_r, u)
                Z_next = (u * g_r).sum(dim=-1)

                new_S[r] = S_r
                new_g[r] = g_r
                Y_orders.append(Y_next)
                Z_orders.append(Z_next)

            S = new_S
            g = new_g

            numerator = 0.0
            denominator = 0.0
            for r in range(self.m + 1):
                numerator = numerator + coeffs[r] * Y_orders[r]
                denominator = denominator + coeffs[r] * Z_orders[r].unsqueeze(-1)

            yt = numerator / denominator.clamp_min(self.eps)
            ys.append(yt)

            if return_diagnostics:
                z_stats.append(denominator.detach())
                order_norm_stats.append(torch.stack([yo.detach().norm(dim=-1).mean() for yo in Y_orders]))

        y = torch.stack(ys, dim=2)  # [B,H,T,D]
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.dropout(self.wo(y))

        if not return_diagnostics:
            return y

        z_cat = torch.cat([z.reshape(-1) for z in z_stats])
        order_norms = torch.stack(order_norm_stats).mean(dim=0)
        diag = {
            "minZ": float(z_cat.min().cpu()),
            "maxZ": float(z_cat.max().cpu()),
            "meanZ": float(z_cat.mean().cpu()),
            "mean_order_norms": [float(v.cpu()) for v in order_norms],
        }
        return y, diag


class CausalSoftmaxAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, return_diagnostics: bool = False):
        B, T, C = x.shape
        H = self.n_heads
        D = self.d_head

        q = self.wq(x).view(B, T, H, D).transpose(1, 2)
        k = self.wk(x).view(B, T, H, D).transpose(1, 2)
        v = self.wv(x).view(B, T, H, D).transpose(1, 2)

        scores = torch.einsum("bhtd,bhsd->bhts", q, k) / math.sqrt(D)
        causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(causal_mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        y = torch.einsum("bhts,bhsd->bhtd", weights, v)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.dropout(self.wo(y))

        if return_diagnostics:
            w = weights.clamp_min(1e-9)
            return y, {"mean_entropy": float((-(w * w.log()).sum(dim=-1)).mean().detach().cpu())}
        return y


# -----------------------------
# Tiny GPT model
# -----------------------------

class Block(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        attn_kind: str,
        dropout: float,
        krylov_m: int,
        krylov_beta: float,
        krylov_rho: float,
        krylov_state_decay: float,
    ):
        super().__init__()
        self.attn_kind = attn_kind
        self.ln1 = nn.LayerNorm(d_model)

        if attn_kind == "softmax":
            self.attn = CausalSoftmaxAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        elif attn_kind == "krylov":
            self.attn = CausalKrylovAttention(
                d_model=d_model,
                n_heads=n_heads,
                m=krylov_m,
                beta=krylov_beta,
                rho=krylov_rho,
                state_decay=krylov_state_decay,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown attn_kind: {attn_kind}")

        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        d_model: int = 64,
        n_heads: int = 4,
        d_ff: int = 256,
        n_layers: int = 2,
        dropout: float = 0.0,
        attn_kind: str = "krylov",
        attn_schedule: Optional[List[str]] = None,
        krylov_m: int = 3,
        krylov_beta: float = 0.5,
        krylov_rho: float = 1.0,
        krylov_state_decay: float = 0.95,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(seq_len, d_model)

        if attn_schedule is None:
            attn_schedule = [attn_kind] * n_layers
        else:
            n_layers = len(attn_schedule)

        self.attn_schedule = attn_schedule
        self.blocks = nn.ModuleList([
            Block(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                attn_kind=kind,
                dropout=dropout,
                krylov_m=krylov_m,
                krylov_beta=krylov_beta,
                krylov_rho=krylov_rho,
                krylov_state_decay=krylov_state_decay,
            )
            for kind in attn_schedule
        ])

        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, idx: torch.Tensor):
        B, T = idx.shape
        assert T <= self.seq_len
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.tok(idx) + self.pos(pos)
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln(x))


# -----------------------------
# Training / evaluation
# -----------------------------

@dataclass
class ExperimentConfig:
    name: str
    grammar: str
    attn: str
    attn_schedule: Optional[str]
    steps: int
    batch_size: int
    eval_batches: int
    log_every: int
    eval_every: int
    seq_len: int
    n_pairs: int
    n_edges: int
    hop: int
    hide_hop_token: bool
    d_model: int
    n_heads: int
    d_ff: int
    n_layers: int
    dropout: float
    krylov_m: int
    krylov_beta: float
    krylov_rho: float
    krylov_state_decay: float
    lr: float
    weight_decay: float
    grad_clip: float
    answer_loss_weight: float
    target_acc: float
    early_stop: bool
    device: str
    seed: int


def parse_attn_schedule(attn: str, n_layers: int, schedule: Optional[str]) -> List[str]:
    if schedule is None or schedule.strip() == "":
        return [attn] * n_layers

    raw = schedule.replace(" ", "").lower()
    aliases = {
        "s": "softmax",
        "soft": "softmax",
        "softmax": "softmax",
        "k": "krylov",
        "krylov": "krylov",
    }
    parts = [p for p in raw.replace(">", ",").replace("-", ",").split(",") if p]
    parsed = []
    for p in parts:
        if p not in aliases:
            raise ValueError(f"Unknown schedule token {p!r} in {schedule!r}")
        parsed.append(aliases[p])
    return parsed


def compute_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    answer_pos: torch.Tensor,
    answer_loss_weight: float,
) -> torch.Tensor:
    per_token = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        y.reshape(-1),
        ignore_index=PAD_ID,
        reduction="none",
    ).view_as(y)

    mask = (y != PAD_ID).float()
    weights = mask.clone()
    if answer_loss_weight != 1.0:
        bidx = torch.arange(y.size(0), device=y.device)
        weights[bidx, answer_pos] = answer_loss_weight
        weights = weights * mask

    return (per_token * weights).sum() / weights.sum().clamp_min(1.0)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    cfg: ExperimentConfig,
    seq_total_len: int,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_answer_correct = 0
    total_answer = 0

    for _ in range(cfg.eval_batches):
        x, y, answer_pos, answer_id, _ = make_batch(
            batch_size=cfg.batch_size,
            grammar=cfg.grammar,
            seq_total_len=seq_total_len,
            n_pairs=cfg.n_pairs,
            n_edges=cfg.n_edges,
            hop=cfg.hop,
            hide_hop_token=cfg.hide_hop_token,
            device=device,
        )
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            ignore_index=PAD_ID,
            reduction="sum",
        )
        nonpad = (y != PAD_ID).sum().item()
        total_loss += float(loss.item())
        total_tokens += nonpad

        bidx = torch.arange(cfg.batch_size, device=device)
        pred_answer = logits[bidx, answer_pos, :].argmax(dim=-1)
        total_answer_correct += int((pred_answer == answer_id).sum().item())
        total_answer += cfg.batch_size

    avg_loss = total_loss / max(total_tokens, 1)
    return {
        "loss": avg_loss,
        "ppl": math.exp(min(avg_loss, 20.0)),
        "answer_acc": total_answer_correct / max(total_answer, 1),
    }


def build_model(cfg: ExperimentConfig) -> TinyGPT:
    schedule = parse_attn_schedule(cfg.attn, cfg.n_layers, cfg.attn_schedule)
    return TinyGPT(
        vocab_size=len(VOCAB),
        seq_len=cfg.seq_len,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        d_ff=cfg.d_ff,
        n_layers=len(schedule),
        dropout=cfg.dropout,
        attn_kind=cfg.attn,
        attn_schedule=schedule,
        krylov_m=cfg.krylov_m,
        krylov_beta=cfg.krylov_beta,
        krylov_rho=cfg.krylov_rho,
        krylov_state_decay=cfg.krylov_state_decay,
    )


def train_one(cfg: ExperimentConfig, verbose: bool = True, sample_prints: bool = True) -> Dict[str, object]:
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device(cfg.device)
    seq_total_len = cfg.seq_len + 1
    schedule = parse_attn_schedule(cfg.attn, cfg.n_layers, cfg.attn_schedule)

    model = build_model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    if verbose:
        print("=" * 100)
        print(f"experiment={cfg.name}")
        print(
            f"grammar={cfg.grammar} hop={cfg.hop} n_edges={cfg.n_edges} n_pairs={cfg.n_pairs} "
            f"seq_len={cfg.seq_len} hide_hop={cfg.hide_hop_token}"
        )
        print(
            f"schedule={'>'.join(schedule)} m={cfg.krylov_m} rho={cfg.krylov_rho} "
            f"beta={cfg.krylov_beta} decay={cfg.krylov_state_decay}"
        )
        print(
            f"layers={len(schedule)} d_model={cfg.d_model} heads={cfg.n_heads} d_ff={cfg.d_ff} "
            f"steps={cfg.steps} seed={cfg.seed} device={device}"
        )
        print()

    t0 = time.time()
    solved_step: Optional[int] = None
    best_acc = 0.0
    best_eval_loss = float("inf")
    last_metrics: Dict[str, float] = {"loss": float("nan"), "ppl": float("nan"), "answer_acc": 0.0}

    for step in range(1, cfg.steps + 1):
        model.train()
        x, y, answer_pos, answer_id, _ = make_batch(
            batch_size=cfg.batch_size,
            grammar=cfg.grammar,
            seq_total_len=seq_total_len,
            n_pairs=cfg.n_pairs,
            n_edges=cfg.n_edges,
            hop=cfg.hop,
            hide_hop_token=cfg.hide_hop_token,
            device=device,
        )
        logits = model(x)
        loss = compute_loss(logits, y, answer_pos, cfg.answer_loss_weight)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        if verbose and (step % cfg.log_every == 0 or step == 1):
            with torch.no_grad():
                bidx = torch.arange(cfg.batch_size, device=device)
                pred_answer = logits[bidx, answer_pos, :].argmax(dim=-1)
                train_answer_acc = (pred_answer == answer_id).float().mean().item()
            print(
                f"step={step:6d} loss={loss.item():.4f} "
                f"train_answer_acc={train_answer_acc:.3f} elapsed={time.time() - t0:.1f}s"
            )

        if step % cfg.eval_every == 0 or step == cfg.steps:
            metrics = evaluate(model, cfg, seq_total_len, device)
            last_metrics = metrics
            best_acc = max(best_acc, metrics["answer_acc"])
            best_eval_loss = min(best_eval_loss, metrics["loss"])

            if verbose:
                print(
                    f"[eval] step={step:6d} loss={metrics['loss']:.4f} "
                    f"ppl={metrics['ppl']:.2f} answer_acc={metrics['answer_acc']:.3f}"
                )

            if sample_prints and verbose:
                with torch.no_grad():
                    x_ex, _, ap_ex, aid_ex, texts_ex = make_batch(
                        batch_size=min(4, cfg.batch_size),
                        grammar=cfg.grammar,
                        seq_total_len=seq_total_len,
                        n_pairs=cfg.n_pairs,
                        n_edges=cfg.n_edges,
                        hop=cfg.hop,
                        hide_hop_token=cfg.hide_hop_token,
                        device=device,
                    )
                    logits_ex = model(x_ex)
                    pred = logits_ex[torch.arange(x_ex.size(0), device=device), ap_ex, :].argmax(dim=-1)
                    for i in range(x_ex.size(0)):
                        print("   ", texts_ex[i], f" | pred={ITOS[int(pred[i])]} target={ITOS[int(aid_ex[i])]}")
                print()

            if metrics["answer_acc"] >= cfg.target_acc and solved_step is None:
                solved_step = step
                if verbose:
                    print(f"[solved] step={step} answer_acc={metrics['answer_acc']:.3f}")
                if cfg.early_stop:
                    break

    elapsed = time.time() - t0
    result = {
        "name": cfg.name,
        "grammar": cfg.grammar,
        "hop": cfg.hop,
        "n_edges": cfg.n_edges,
        "n_pairs": cfg.n_pairs,
        "hide_hop_token": cfg.hide_hop_token,
        "schedule": ">".join(schedule),
        "attn": cfg.attn,
        "krylov_m": cfg.krylov_m,
        "krylov_rho": cfg.krylov_rho,
        "krylov_beta": cfg.krylov_beta,
        "krylov_state_decay": cfg.krylov_state_decay,
        "n_layers": len(schedule),
        "d_model": cfg.d_model,
        "n_heads": cfg.n_heads,
        "d_ff": cfg.d_ff,
        "seed": cfg.seed,
        "steps_budget": cfg.steps,
        "solved_step": solved_step if solved_step is not None else "",
        "best_acc": best_acc,
        "best_eval_loss": best_eval_loss,
        "final_acc": last_metrics["answer_acc"],
        "final_eval_loss": last_metrics["loss"],
        "final_ppl": last_metrics["ppl"],
        "elapsed_s": elapsed,
    }

    if verbose:
        print(f"[done] {cfg.name} solved_step={result['solved_step']} best_acc={best_acc:.3f} elapsed={elapsed:.1f}s")
        print()
    return result


# -----------------------------
# Grid construction
# -----------------------------

def cfg_from_args(args, **overrides) -> ExperimentConfig:
    base = dict(
        name="single",
        grammar=args.grammar,
        attn=args.attn,
        attn_schedule=args.attn_schedule,
        steps=args.steps,
        batch_size=args.batch_size,
        eval_batches=args.eval_batches,
        log_every=args.log_every,
        eval_every=args.eval_every,
        seq_len=args.seq_len,
        n_pairs=args.n_pairs,
        n_edges=args.n_edges,
        hop=args.hop,
        hide_hop_token=args.hide_hop_token,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        n_layers=args.n_layers,
        dropout=args.dropout,
        krylov_m=args.krylov_m,
        krylov_beta=args.krylov_beta,
        krylov_rho=args.krylov_rho,
        krylov_state_decay=args.krylov_state_decay,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        answer_loss_weight=args.answer_loss_weight,
        target_acc=args.target_acc,
        early_stop=args.early_stop,
        device=args.device,
        seed=args.seed,
    )
    base.update(overrides)
    return ExperimentConfig(**base)


def make_schedule(kind: str, depth: int, m: int) -> Tuple[str, str]:
    """Return (attn, schedule_string) for a named architecture."""
    if kind == "softmax":
        return "softmax", ",".join(["softmax"] * depth)
    if kind == "krylov":
        return "krylov", ",".join(["krylov"] * depth)
    if kind == "krylov_then_softmax":
        if depth == 1:
            return "softmax", "softmax"
        return "krylov", ",".join(["krylov"] * (depth - 1) + ["softmax"])
    if kind == "half_krylov_then_softmax":
        n_k = max(1, depth // 2)
        n_s = depth - n_k
        return "krylov", ",".join(["krylov"] * n_k + ["softmax"] * n_s)
    if kind == "softmax_then_krylov":
        if depth == 1:
            return "krylov", "krylov"
        return "krylov", ",".join(["softmax"] + ["krylov"] * (depth - 1))
    raise ValueError(f"Unknown architecture kind: {kind}")


def build_grid(args) -> List[ExperimentConfig]:
    seeds = [int(s) for s in str(args.grid_seeds).split(",") if s.strip()]
    grid: List[ExperimentConfig] = []

    if args.grid == "path_depth":
        # Main grid: does Krylov order / hybrid readout reduce required depth or steps?
        depths = [int(x) for x in args.grid_depths.split(",") if x]
        hops = [int(x) for x in args.grid_hops.split(",") if x]
        archs = [x for x in args.grid_archs.split(",") if x]
        ms = [int(x) for x in args.grid_m_values.split(",") if x]

        for hop in hops:
            for depth in depths:
                for arch in archs:
                    if arch == "softmax":
                        m_values = [1]
                    else:
                        m_values = ms
                    for m in m_values:
                        attn, schedule = make_schedule(arch, depth, m)
                        for seed in seeds:
                            name = f"path_h{hop}_d{depth}_{arch}_m{m}_seed{seed}"
                            grid.append(cfg_from_args(
                                args,
                                name=name,
                                grammar="path",
                                hop=hop,
                                n_edges=max(args.n_edges, hop),
                                seq_len=args.seq_len,
                                hide_hop_token=args.hide_hop_token,
                                attn=attn,
                                attn_schedule=schedule,
                                n_layers=depth,
                                krylov_m=m,
                                seed=seed,
                            ))

    elif args.grid == "kv_ladder":
        # Sanity ladder for sharp retrieval difficulty.
        pairs_values = [int(x) for x in args.grid_pairs.split(",") if x]
        archs = [x for x in args.grid_archs.split(",") if x]
        ms = [int(x) for x in args.grid_m_values.split(",") if x]
        for n_pairs in pairs_values:
            seq_len = max(args.seq_len, 4 * n_pairs + 7)
            for arch in archs:
                m_values = [1] if arch == "softmax" else ms
                for m in m_values:
                    attn, schedule = make_schedule(arch, args.n_layers, m)
                    for seed in seeds:
                        name = f"kv_p{n_pairs}_{arch}_m{m}_seed{seed}"
                        grid.append(cfg_from_args(
                            args,
                            name=name,
                            grammar="kv",
                            n_pairs=n_pairs,
                            seq_len=seq_len,
                            attn=attn,
                            attn_schedule=schedule,
                            krylov_m=m,
                            seed=seed,
                        ))

    elif args.grid == "quick":
        # Small sanity grid suitable for checking script mechanics.
        for arch, depth, m, hop in [
            ("softmax", 2, 1, 1),
            ("krylov", 2, 1, 1),
            ("krylov", 2, 2, 2),
            ("krylov_then_softmax", 2, 2, 2),
        ]:
            attn, schedule = make_schedule(arch, depth, m)
            grid.append(cfg_from_args(
                args,
                name=f"quick_{arch}_d{depth}_m{m}_h{hop}",
                grammar="path",
                hop=hop,
                n_edges=max(3, hop),
                attn=attn,
                attn_schedule=schedule,
                n_layers=depth,
                krylov_m=m,
            ))
    else:
        raise ValueError(f"Unknown grid preset: {args.grid}")

    if args.max_experiments is not None:
        grid = grid[: args.max_experiments]
    return grid


def write_result(output_dir: str, result: Dict[str, object], cfg: ExperimentConfig) -> None:
    os.makedirs(output_dir, exist_ok=True)
    jsonl_path = os.path.join(output_dir, "results.jsonl")
    csv_path = os.path.join(output_dir, "results.csv")

    row = dict(result)
    row["config"] = json.dumps(asdict(cfg), sort_keys=True)

    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")

    file_exists = os.path.exists(csv_path)
    fieldnames = [
        "name", "grammar", "hop", "n_edges", "n_pairs", "hide_hop_token",
        "schedule", "attn", "krylov_m", "krylov_rho", "krylov_beta", "krylov_state_decay",
        "n_layers", "d_model", "n_heads", "d_ff", "seed", "steps_budget", "solved_step",
        "best_acc", "best_eval_loss", "final_acc", "final_eval_loss", "final_ppl", "elapsed_s",
    ]
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: result.get(k, "") for k in fieldnames})


def run_grid(args) -> None:
    grid = build_grid(args)
    os.makedirs(args.output_dir, exist_ok=True)

    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in grid], f, indent=2, sort_keys=True)

    print(f"Prepared {len(grid)} experiments. output_dir={args.output_dir}")
    for i, cfg in enumerate(grid, start=1):
        print(f"\n### GRID {i}/{len(grid)}: {cfg.name}")
        result = train_one(cfg, verbose=not args.quiet_grid, sample_prints=args.sample_prints)
        write_result(args.output_dir, result, cfg)
        print(f"### RESULT {i}/{len(grid)}: {result}")

    print(f"\nGrid complete. CSV: {os.path.join(args.output_dir, 'results.csv')}")
    print(f"JSONL: {os.path.join(args.output_dir, 'results.jsonl')}")


# -----------------------------
# CLI
# -----------------------------

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--grid", type=str, default="none", choices=["none", "quick", "path_depth", "kv_ladder"])
    p.add_argument("--output-dir", type=str, default="runs/krylov_grid")
    p.add_argument("--quiet-grid", action="store_true")
    p.add_argument("--sample-prints", action="store_true")
    p.add_argument("--max-experiments", type=int, default=None)

    p.add_argument("--grid-seeds", type=str, default="1")
    p.add_argument("--grid-depths", type=str, default="2,3,4")
    p.add_argument("--grid-hops", type=str, default="2,3")
    p.add_argument("--grid-m-values", type=str, default="1,2,3")
    p.add_argument(
        "--grid-archs",
        type=str,
        default="softmax,krylov,krylov_then_softmax",
        help="Comma list from: softmax,krylov,krylov_then_softmax,half_krylov_then_softmax,softmax_then_krylov",
    )
    p.add_argument("--grid-pairs", type=str, default="2,3,4,6")

    p.add_argument("--grammar", type=str, default="kv", choices=["kv", "path"])
    p.add_argument("--attn", type=str, default="krylov", choices=["krylov", "softmax"])
    p.add_argument("--attn-schedule", type=str, default=None, help="Comma/arrow schedule, e.g. krylov,krylov,softmax or k,k,s")

    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--target-acc", type=float, default=0.99)
    p.add_argument("--early-stop", action="store_true")
    p.add_argument("--answer-loss-weight", type=float, default=1.0)

    p.add_argument("--seq-len", type=int, default=39)
    p.add_argument("--n-pairs", type=int, default=6)
    p.add_argument("--n-edges", type=int, default=6)
    p.add_argument("--hop", type=int, default=3)
    p.add_argument("--hide-hop-token", action="store_true")

    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--d-ff", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.0)

    p.add_argument("--krylov-m", type=int, default=3)
    p.add_argument("--krylov-beta", type=float, default=0.5)
    p.add_argument("--krylov-rho", type=float, default=1.0)
    p.add_argument("--krylov-state-decay", type=float, default=0.95)

    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)

    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=1)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.grid == "none":
        cfg = cfg_from_args(args)
        train_one(cfg, verbose=True, sample_prints=True)
    else:
        run_grid(args)
