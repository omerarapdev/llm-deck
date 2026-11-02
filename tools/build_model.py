#!/usr/bin/env python3
"""Builds the tiny SQL language model used throughout the deck.

Every number the slides show (tokenizer merges, token ids, weights,
probabilities, attention patterns, loss curves) comes from this script, so the
running example stays consistent from the first slide to the last.

Usage:
    python tools/build_model.py            # train + write data/model-data.js
    python tools/build_model.py --explore  # train only, print diagnostics
"""
import argparse
import base64
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------- the world
SCHEMA = {
    "users": ["id", "name", "email", "age"],
    "orders": ["id", "user_id", "total", "status"],
    "products": ["id", "title", "price"],
}
VALUES = [str(v) for v in range(1, 10)]
RUNNING = "SELECT name FROM users WHERE id = 7;"


def sample_query(rng):
    """Grammar: SELECT c1 FROM t WHERE c2 = v;  with c1, c2 columns of t and c2 != c1."""
    t = rng.choice(list(SCHEMA))
    c1 = rng.choice(SCHEMA[t])
    c2 = rng.choice([c for c in SCHEMA[t] if c != c1])
    v = rng.choice(VALUES)
    return f"SELECT {c1} FROM {t} WHERE {c2} = {v};"


def all_queries():
    """Every query the grammar can produce, with its exact probability."""
    out = []
    for t, cols in SCHEMA.items():
        for c1 in cols:
            rest = [c for c in cols if c != c1]
            for c2 in rest:
                for v in VALUES:
                    p = 1 / len(SCHEMA) / len(cols) / len(rest) / len(VALUES)
                    out.append((f"SELECT {c1} FROM {t} WHERE {c2} = {v};", p))
    return out


# ---------------------------------------------------------------- tokenizer
PRETOK = re.compile(r" ?[A-Za-z_]+| ?[0-9]+| ?[^\sA-Za-z_0-9]+")


def pretokenize(text):
    return PRETOK.findall(text)


def train_bpe(texts):
    """Byte-pair encoding on characters. Most frequent adjacent pair wins;
    ties go to the pair that appears first in the corpus."""
    order, counts = [], Counter()
    for t in texts:
        for w in pretokenize(t):
            if w not in counts:
                order.append(w)
            counts[w] += 1
    splits = {w: list(w) for w in order}
    base = sorted({ch for w in order for ch in w})
    merges, log = [], []
    while True:
        pc, first = {}, {}
        for wi, w in enumerate(order):
            s = splits[w]
            for i in range(len(s) - 1):
                p = (s[i], s[i + 1])
                pc[p] = pc.get(p, 0) + counts[w]
                first.setdefault(p, (wi, i))
        if not pc:
            break
        ranked = sorted(pc, key=lambda p: (-pc[p], first[p]))
        best = ranked[0]
        log.append({"pair": list(best), "count": pc[best],
                    "top": [[a, b, pc[(a, b)]] for a, b in ranked[:5]]})
        merges.append(best)
        for w in order:
            s, out, i = splits[w], [], 0
            while i < len(s):
                if i < len(s) - 1 and (s[i], s[i + 1]) == best:
                    out.append(s[i] + s[i + 1])
                    i += 2
                else:
                    out.append(s[i])
                    i += 1
            splits[w] = out
    vocab = base + [a + b for a, b in merges]
    return vocab, merges, log, counts


def bpe_encode_word(word, ranks, upto=None):
    s = list(word)
    while len(s) > 1:
        best, best_rank = None, None
        for i in range(len(s) - 1):
            r = ranks.get((s[i], s[i + 1]))
            if r is not None and (upto is None or r < upto) and (best_rank is None or r < best_rank):
                best, best_rank = (s[i], s[i + 1]), r
        if best is None:
            break
        out, i = [], 0
        while i < len(s):
            if i < len(s) - 1 and (s[i], s[i + 1]) == best:
                out.append(s[i] + s[i + 1])
                i += 2
            else:
                out.append(s[i])
                i += 1
        s = out
    return s


def bpe_encode(text, ranks, upto=None):
    return [tok for w in pretokenize(text) for tok in bpe_encode_word(w, ranks, upto)]


# ---------------------------------------------------------------- model
PARAM_ORDER = None  # filled in by init_params


def init_params(V, T, d, L, dff, seed):
    g = torch.Generator().manual_seed(seed)

    def rnd(*shape, std):
        return (torch.randn(*shape, generator=g) * std).requires_grad_()

    p = {"wte": rnd(V, d, std=0.5), "wpe": rnd(T, d, std=0.5)}
    for l in range(L):
        p[f"l{l}.n1"] = torch.ones(d, requires_grad=True)
        p[f"l{l}.wq"] = rnd(d, d, std=d ** -0.5)
        p[f"l{l}.wk"] = rnd(d, d, std=d ** -0.5)
        p[f"l{l}.wv"] = rnd(d, d, std=d ** -0.5)
        p[f"l{l}.wo"] = rnd(d, d, std=(d ** -0.5) / math.sqrt(2 * L))
        p[f"l{l}.n2"] = torch.ones(d, requires_grad=True)
        p[f"l{l}.w1"] = rnd(d, dff, std=d ** -0.5)
        p[f"l{l}.w2"] = rnd(dff, d, std=(dff ** -0.5) / math.sqrt(2 * L))
    p["nf"] = torch.ones(d, requires_grad=True)
    p["wu"] = rnd(d, V, std=0.02)
    return p


def rmsnorm(x, gain, eps=1e-5):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * gain


def forward(p, ids, H, L, trace=False):
    B, T = ids.shape
    d = p["wte"].shape[1]
    hd = d // H
    x = p["wte"][ids] + p["wpe"][:T]
    tr = {"x0": x.detach().clone(), "att": [], "resid": [x.detach().clone()]}
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
    for l in range(L):
        h = rmsnorm(x, p[f"l{l}.n1"])
        q = (h @ p[f"l{l}.wq"]).view(B, T, H, hd).transpose(1, 2)
        k = (h @ p[f"l{l}.wk"]).view(B, T, H, hd).transpose(1, 2)
        v = (h @ p[f"l{l}.wv"]).view(B, T, H, hd).transpose(1, 2)
        s = (q @ k.transpose(-1, -2)) / math.sqrt(hd)
        s = s.masked_fill(mask, float("-inf"))
        a = s.softmax(-1)
        o = (a @ v).transpose(1, 2).reshape(B, T, d)
        x = x + o @ p[f"l{l}.wo"]
        if trace:
            tr["att"].append(a.detach().clone())
            tr["resid"].append(x.detach().clone())
        h = rmsnorm(x, p[f"l{l}.n2"])
        x = x + F.gelu(h @ p[f"l{l}.w1"], approximate="tanh") @ p[f"l{l}.w2"]
        if trace:
            tr["resid"].append(x.detach().clone())
    logits = rmsnorm(x, p["nf"]) @ p["wu"]
    return (logits, tr) if trace else logits


# ---------------------------------------------------------------- analysis
def entropy_floor(queries_p, ranks, stoi):
    """Exact per-position conditional entropy of the grammar (nats)."""
    by_prefix = defaultdict(Counter)
    pos_mass = defaultdict(float)
    for q, p in queries_p:
        ids = [stoi[t] for t in bpe_encode(q, ranks)]
        for i in range(1, len(ids)):
            by_prefix[tuple(ids[:i])][ids[i]] += p
    floor = defaultdict(float)
    for prefix, nxt in by_prefix.items():
        tot = sum(nxt.values())
        h = -sum((m / tot) * math.log(m / tot) for m in nxt.values())
        floor[len(prefix) - 1] += tot * h
        pos_mass[len(prefix) - 1] += tot
    return [floor[i] for i in sorted(floor)]


def exact_loss(p, all_ids, all_p, H, L):
    with torch.no_grad():
        logits = forward(p, all_ids[:, :-1], H, L)
        nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                              all_ids[:, 1:].reshape(-1), reduction="none").view(all_ids.shape[0], -1)
        per_pos = (nll * all_p[:, None]).sum(0)
    return per_pos.mean().item(), per_pos.tolist()


def pack(p, names, dtype=np.float32):
    flat = np.concatenate([p[n].detach().numpy().astype(dtype).ravel() for n in names])
    return base64.b64encode(flat.astype(flat.dtype.newbyteorder("<")).tobytes()).decode()


def unpack_like(b64, p, names, dtype):
    flat = np.frombuffer(base64.b64decode(b64), dtype=np.dtype(dtype).newbyteorder("<")).astype(np.float32)
    out, i = {}, 0
    for n in names:
        k = p[n].numel()
        out[n] = torch.tensor(flat[i:i + k].reshape(p[n].shape))
        i += k
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dff", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--explore", action="store_true")
    args = ap.parse_args()
    # one thread keeps floating-point summation order, and so the whole run, reproducible
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    # corpus for the tokenizer: the running example first, then sampled queries
    rng = random.Random(1234)
    corpus = [RUNNING] + [sample_query(rng) for _ in range(1999)]
    vocab, merges, bpe_log, word_counts = train_bpe(corpus)
    ranks = {m: i for i, m in enumerate(merges)}
    stoi = {s: i for i, s in enumerate(vocab)}
    assert len(stoi) == len(vocab), "duplicate token strings"

    queries_p = all_queries()
    enc = [[stoi[t] for t in bpe_encode(q, ranks)] for q, _ in queries_p]
    assert all(len(e) == 9 for e in enc), "every query should be 9 tokens"
    all_ids = torch.tensor(enc)
    all_p = torch.tensor([p for _, p in queries_p], dtype=torch.float32)
    floor = entropy_floor(queries_p, ranks, stoi)
    print(f"vocab {len(vocab)} (base {len(vocab) - len(merges)}, merges {len(merges)}); "
          f"queries {len(queries_p)}; floor {np.mean(floor):.4f} per-pos {[round(f, 4) for f in floor]}")

    V, T = len(vocab), 8
    H, L, d, dff = args.heads, args.layers, args.d, args.dff
    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        t = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))

    snap_steps = [0, 25, 50, 75, 100, 150, 200, 300, 400, 800, 1600, 3000, args.steps]

    def train(exclude=frozenset(), record=True):
        """Train one model. `exclude` lists queries the sampler never produces."""
        p = init_params(V, T, d, L, dff, args.seed)
        names = list(p)
        decay = [p[n] for n in names if p[n].dim() == 2]
        no_decay = [p[n] for n in names if p[n].dim() < 2]
        opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd},
                                 {"params": no_decay, "weight_decay": 0.0}],
                                lr=args.lr, betas=(0.9, 0.95))
        snaps, batch_loss, exact_curve = {}, [], []
        drng = random.Random(args.seed + 99)

        def draw():
            while True:
                q = sample_query(drng)
                if q not in exclude:
                    return q

        for step in range(args.steps + 1):
            if record and (step % 10 == 0 or step in snap_steps):
                ex, per = exact_loss(p, all_ids, all_p, H, L)
                exact_curve.append([step, round(ex, 5), [round(x, 5) for x in per]])
            if record and step in snap_steps:
                snaps[step] = pack(p, names, np.float32 if step == args.steps else np.float16)
            if step == args.steps:
                break
            qs = [draw() for _ in range(args.batch)]
            ids = torch.tensor([[stoi[t] for t in bpe_encode(q, ranks)] for q in qs])
            logits = forward(p, ids[:, :-1], H, L)
            loss = F.cross_entropy(logits.reshape(-1, V), ids[:, 1:].reshape(-1))
            for gpar in opt.param_groups:
                gpar["lr"] = lr_at(step)
            opt.zero_grad()
            loss.backward()
            opt.step()
            batch_loss.append(round(loss.item(), 4))
        return p, names, snaps, batch_loss, exact_curve

    p, names, snaps, batch_loss, exact_curve = train()
    nparams = sum(t.numel() for t in p.values())
    print(f"params {nparams}")

    # A second model trained on a random 90% of the queries; the held-out 10% always includes ours
    hrng = random.Random(2024)
    others = [q for q, _ in queries_p if q != RUNNING]
    held = frozenset([RUNNING] + hrng.sample(others, 26))
    pB, *_ = train(exclude=held, record=False)
    is_held = torch.tensor([q in held for q, _ in queries_p])

    def split_loss(params):
        with torch.no_grad():
            lg = forward(params, all_ids[:, :-1], H, L)
            nll = F.cross_entropy(lg.reshape(-1, V), all_ids[:, 1:].reshape(-1), reduction="none").view(all_ids.shape[0], -1).mean(1)
        def avg(mask):
            w = all_p * mask
            return (nll * w).sum().item() / w.sum().item()
        return {"held": round(avg(is_held.float()), 5), "seen": round(avg((~is_held).float()), 5)}
    holdout = {"queries": sorted(held), "dtype": "f32", "b64": pack(pB, names, np.float32),
               "lossA": split_loss(p), "lossB": split_loss(pB)}
    print("held-out experiment:", {k: v for k, v in holdout.items() if k.startswith("loss")})

    ex, per = exact_loss(p, all_ids, all_p, H, L)
    print(f"final exact loss {ex:.4f} vs floor {np.mean(floor):.4f}; per-pos gap "
          f"{[round(a - b, 4) for a, b in zip(per, floor)]}")
    for s_, e_, per_ in exact_curve:
        if s_ in snap_steps:
            print(f"  step {s_:5d} exact {e_:.4f} per-pos {[round(x, 3) for x in per_]}")

    # running example diagnostics
    run_tokens = bpe_encode(RUNNING, ranks)
    run_ids = torch.tensor([[stoi[t] for t in run_tokens]])
    logits, tr = forward(p, run_ids[:, :-1], H, L, trace=True)
    probs = logits.softmax(-1)[0]
    print("running example:", run_tokens)
    for i in range(8):
        top = probs[i].topk(4)
        tops = ", ".join(f"{vocab[j]!r}:{v:.3f}" for v, j in zip(top.values.tolist(), top.indices.tolist()))
        print(f"  pos {i} {run_tokens[i]!r:>10} -> target {run_tokens[i + 1]!r:>9} "
              f"p={probs[i, stoi[run_tokens[i + 1]]]:.3f}  top: {tops}")
    for l, a in enumerate(tr["att"]):
        for h in range(H):
            print(f"  layer {l} head {h}")
            for i in range(8):
                row = " ".join(f"{x:4.2f}" for x in a[0, h, i, : i + 1].tolist())
                print(f"     q={run_tokens[i]!r:>10}: {row}")

    if args.explore:
        return

    # mean batch loss per 10 steps keeps the curve honest but small
    batch10 = [round(float(np.mean(batch_loss[i:i + 10])), 4) for i in range(0, len(batch_loss), 10)]
    data = {
        "config": {"V": V, "T": T, "d": d, "H": H, "L": L, "dff": dff, "eps": 1e-5,
                   "params": nparams, "lr": args.lr, "steps": args.steps, "batch": args.batch,
                   "warmup": args.warmup, "seed": args.seed, "betas": [0.9, 0.95], "wd": args.wd,
                   "shapes": [[n, list(p[n].shape)] for n in names]},
        "schema": SCHEMA, "values": VALUES, "running": RUNNING,
        "vocab": vocab, "baseCount": len(vocab) - len(merges),
        "merges": [list(m) for m in merges], "bpeLog": bpe_log,
        "wordCounts": [[w, c] for w, c in word_counts.most_common()],
        "corpusHead": corpus[:16], "corpusSize": len(corpus), "corpusDistinct": len(set(corpus)),
        "distinctQueries": len(queries_p),
        "floor": [round(f, 6) for f in floor],
        "batchLoss10": batch10, "exactCurve": exact_curve,
        "snapSteps": snap_steps, "holdout": holdout,
        "snaps": {str(k): {"dtype": "f32" if k == args.steps else "f16", "b64": v} for k, v in snaps.items()},
    }
    out = ROOT / "data" / "model-data.js"
    out.write_text("// Generated by tools/build_model.py. Do not edit by hand.\n"
                   "window.MODEL_DATA = " + json.dumps(data, separators=(",", ":")) + ";\n")
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")

    # fixtures for checking the JavaScript forward pass and tokenizer, computed
    # from the exact weights the page decodes (float16 snapshots included)
    snap_fx = {}
    for k in snap_steps:
        dt = np.float32 if k == args.steps else np.float16
        w = unpack_like(snaps[k], p, names, dt)
        with torch.no_grad():
            lg, trk = forward(w, run_ids[:, :-1], H, L, trace=True)
        exk, perk = exact_loss(w, all_ids, all_p, H, L)
        snap_fx[str(k)] = {"logits": lg[0].tolist(), "att": [a[0].tolist() for a in trk["att"]],
                           "exact": exk, "per_pos": perk}
    fx = {"running_tokens": run_tokens, "running_ids": run_ids[0].tolist(),
          "snaps": snap_fx,
          "tok_tests": {q: bpe_encode(q, ranks) for q in [
              RUNNING, "SELECT total FROM orders WHERE status = 3;",
              "SELECT price FROM products WHERE title = 9;",
              "SELECT username FROM customers WHERE age = 42;"]},
          "tok_upto": {str(k): bpe_encode(RUNNING, ranks, upto=k) for k in range(len(merges) + 1)}}
    (ROOT / "tools" / "fixtures.json").write_text(json.dumps(fx))
    print("wrote tools/fixtures.json")


if __name__ == "__main__":
    main()
