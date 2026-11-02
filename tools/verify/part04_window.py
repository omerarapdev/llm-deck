#!/usr/bin/env python3
"""Part 4: a fixed-window neural language model (Bengio et al. 2003) on our SQL grammar.

For a window of k previous tokens the model is
    x = [C(w_{t-k+1}); ...; C(w_t)]        (k embeddings of 16 numbers, concatenated)
    h = tanh(d + H x)                      (64 hidden units)
    y = b + U h                            (113 logits, then softmax)
with no direct input-to-output connections. Positions before the start of the
query read a padding row of C. It is trained with the same recipe as the
transformer (Adam 0.9/0.95, lr 0.01, 100 warmup steps, cosine to 0.001, 6,000
steps of 128 freshly sampled queries, seed 7) and scored with the exact expected
loss over all 270 queries, so the numbers can be compared with the n-gram
limits, the floor (0.6937) and the transformer (0.6942).

Run:  <venv>/bin/python tools/verify/part04_window.py
Prints one JSON line per window size (exact loss, per-position loss, and the
probabilities after our query's prefix SELECT name FROM users WHERE);
tools/verify/part04.mjs checks the sheet text against these numbers.
"""
import json
import math
import random
import re
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
raw = (ROOT / "data" / "model-data.js").read_text()
D = json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
SCHEMA, VALUES, VOCAB = D["schema"], D["values"], D["vocab"]
STOI = {s: i for i, s in enumerate(VOCAB)}
V = len(VOCAB)
PRETOK = re.compile(r" ?[A-Za-z_]+| ?[0-9]+| ?[^\sA-Za-z_0-9]+")


def ids_of(q):
    # after all 74 merges every word of the grammar is one token
    return [STOI[w] for w in PRETOK.findall(q)]


def sample_query(rng):
    t = rng.choice(list(SCHEMA))
    c1 = rng.choice(SCHEMA[t])
    c2 = rng.choice([c for c in SCHEMA[t] if c != c1])
    v = rng.choice(VALUES)
    return f"SELECT {c1} FROM {t} WHERE {c2} = {v};"


def all_queries():
    out = []
    for t, cols in SCHEMA.items():
        for c1 in cols:
            rest = [c for c in cols if c != c1]
            for c2 in rest:
                for v in VALUES:
                    out.append((f"SELECT {c1} FROM {t} WHERE {c2} = {v};", 1 / len(SCHEMA) / len(cols) / len(rest) / len(VALUES)))
    return out


def windows(ids, k):
    """ids: [B, 9] -> contexts [B, 8, k] (padding id V before the start) and targets [B, 8]."""
    B = ids.shape[0]
    pad = torch.full((B, k - 1), V, dtype=torch.long)
    seq = torch.cat([pad, ids[:, :8]], 1)
    ctx = torch.stack([seq[:, i: i + k] for i in range(8)], 1)
    return ctx, ids[:, 1:]


def train(k, m=16, h=64, steps=6000, batch=128, lr=1e-2, warmup=100, seed=7):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    P = {
        "C": (torch.randn(V + 1, m, generator=g) * 0.5).requires_grad_(),
        "H": (torch.randn(k * m, h, generator=g) * (k * m) ** -0.5).requires_grad_(),
        "d": torch.zeros(h, requires_grad=True),
        "U": (torch.randn(h, V, generator=g) * 0.02).requires_grad_(),
        "b": torch.zeros(V, requires_grad=True),
    }
    opt = torch.optim.Adam(P.values(), lr=lr, betas=(0.9, 0.95))

    def logits(ctx):
        x = P["C"][ctx].reshape(*ctx.shape[:-1], k * m)
        return torch.tanh(x @ P["H"] + P["d"]) @ P["U"] + P["b"]

    def lr_at(s):
        if s < warmup:
            return lr * (s + 1) / warmup
        t = (s - warmup) / max(1, steps - warmup)
        return lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))

    rng = random.Random(seed + 99)
    for s in range(steps):
        ids = torch.tensor([ids_of(sample_query(rng)) for _ in range(batch)])
        ctx, tgt = windows(ids, k)
        loss = F.cross_entropy(logits(ctx).reshape(-1, V), tgt.reshape(-1))
        for gr in opt.param_groups:
            gr["lr"] = lr_at(s)
        opt.zero_grad()
        loss.backward()
        opt.step()
    qs = all_queries()
    ids = torch.tensor([ids_of(q) for q, _ in qs])
    p = torch.tensor([pq for _, pq in qs], dtype=torch.float64)
    with torch.no_grad():
        ctx, tgt = windows(ids, k)
        nll = F.cross_entropy(logits(ctx).reshape(-1, V), tgt.reshape(-1), reduction="none").view(len(qs), 8).double()
        per = (nll * p[:, None]).sum(0)
        # next-token probabilities after our query's prefix SELECT name FROM users WHERE
        run = torch.tensor([ids_of("SELECT name FROM users WHERE id = 7;")])
        rctx, _ = windows(run, k)
        pw = logits(rctx)[0, 4].softmax(-1)
        where = {c: round(pw[STOI[" " + c]].item(), 4) for c in ("id", "name", "email", "age")}
    n_params = sum(t.numel() for t in P.values())
    return {"k": k, "params": n_params, "mean": round(per.mean().item(), 6), "perPos": [round(x, 6) for x in per.tolist()], "afterWhere": where}


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    for k in (1, 2, 3, 4, 8):
        print(json.dumps(train(k)), flush=True)
