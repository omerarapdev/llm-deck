#!/usr/bin/env python3
"""Part 9 experiments on the trained running-example model, in PyTorch.

    <venv>/bin/python tools/verify/part09_experiments.py

Loads the final weights from data/model-data.js, reuses the forward pass of
tools/build_model.py, and runs the fine-tuning experiments that Part 9 quotes:

  forget  supervised fine-tuning on a narrow slice of the grammar (orders
          queries only), with and without mixing the original distribution
          back in; loss per table before and after.
  dpo     Direct Preference Optimization on one prompt
          ("SELECT email FROM users WHERE"; chosen " id = 7;"), plain and with
          Llama 3's two changes (shared tokens masked, NLL term), several betas
          and two learning rates; where the probability mass goes.
  grpo    GRPO with every parameter trained by backpropagation on the task
          "email of user 7" (reward 1 only for " id = 7;").

Single-threaded and seeded, so every run prints the same numbers. The output is
recorded in tools/verify/part09_exp.mjs; tools/verify/part09.mjs checks that the
sheets state those numbers and, when this venv exists, re-runs this script and
checks that it reproduces the record exactly.
"""
import base64
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import build_model as bm  # noqa: E402  (the training script: forward pass, BPE, grammar)

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)

# ---------------------------------------------------------------- model and data
SRC = (ROOT / "data" / "model-data.js").read_text()
D = json.loads(SRC[SRC.index("{"): SRC.rstrip().rindex("}") + 1])
C = D["config"]
H, L, V = C["H"], C["L"], C["V"]
RANKS = {tuple(m): i for i, m in enumerate(D["merges"])}
STOI = {s: i for i, s in enumerate(D["vocab"])}


def load_weights():
    flat = np.frombuffer(base64.b64decode(D["snaps"]["6000"]["b64"]), dtype="<f4").astype(np.float32)
    p, i = {}, 0
    for name, shape in C["shapes"]:
        k = int(np.prod(shape))
        p[name] = torch.tensor(flat[i:i + k].reshape(shape))
        i += k
    assert i == flat.size
    return p


def enc(q):
    return [STOI[t] for t in bm.bpe_encode(q, RANKS)]


QUERIES = bm.all_queries()
ALL_IDS = torch.tensor([enc(q) for q, _ in QUERIES])
ALL_P = torch.tensor([pq for _, pq in QUERIES], dtype=torch.float64)
TABLE = [q.split(" ")[3] for q, _ in QUERIES]


def nll_matrix(p):
    """Cross-entropy at each of the 8 predictions, for each of the 270 queries."""
    with torch.no_grad():
        lg = bm.forward(p, ALL_IDS[:, :-1], H, L)
        nll = F.cross_entropy(lg.reshape(-1, V), ALL_IDS[:, 1:].reshape(-1), reduction="none")
    return nll.view(len(QUERIES), -1).double()


def table_losses(p):
    """Per table: mean loss over the 8 predictions, and the loss of the filter-column
    prediction (position 4: the token after WHERE), weighted by query probability."""
    nll = nll_matrix(p)
    out = {}
    for t in bm.SCHEMA:
        w = ALL_P * torch.tensor([x == t for x in TABLE], dtype=torch.float64)
        out[t] = {"mean": float((nll.mean(1) * w).sum() / w.sum()), "col": float((nll[:, 4] * w).sum() / w.sum())}
    out["all"] = float((nll.mean(1) * ALL_P).sum())
    return out


def clone(p):
    return {k: v.detach().clone().requires_grad_() for k, v in p.items()}


def adam(p, lr):
    return torch.optim.Adam(list(p.values()), lr=lr, betas=(0.9, 0.95))


# ---------------------------------------------------------------- forget: narrow SFT and data mixing
def sample_table(rng, t):
    c1 = rng.choice(bm.SCHEMA[t])
    c2 = rng.choice([c for c in bm.SCHEMA[t] if c != c1])
    return f"SELECT {c1} FROM {t} WHERE {c2} = {rng.choice(bm.VALUES)};"


def finetune(p0, mix, steps=300, batch=128, lr=1e-3, seed=11):
    """Next-token fine-tuning on orders queries; a fraction `mix` of each batch is
    drawn from the original grammar instead."""
    p, rng = clone(p0), random.Random(seed)
    opt = adam(p, lr)
    for _ in range(steps):
        qs = [bm.sample_query(rng) if rng.random() < mix else sample_table(rng, "orders") for _ in range(batch)]
        ids = torch.tensor([enc(q) for q in qs])
        lg = bm.forward(p, ids[:, :-1], H, L)
        loss = F.cross_entropy(lg.reshape(-1, V), ids[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return p


# ---------------------------------------------------------------- the one-prompt task
PROMPT = "SELECT email FROM users WHERE"
CONTS = [(c, v) for c in ["id", "name", "age"] for v in bm.VALUES]   # 27 grammatical answers
CHOSEN = CONTS.index(("id", "7"))
CONT_IDS = torch.tensor([enc(f"{PROMPT} {c} = {v};") for c, v in CONTS])   # 27 x 9 tokens


def answer_logp(p, ids=CONT_IDS):
    """log pi(answer | prompt): sum over the 4 answer tokens (positions 5..8)."""
    lg = bm.forward(p, ids[:, :-1], H, L).log_softmax(-1)
    tok = lg[:, 4:8].gather(-1, ids[:, 5:9, None]).squeeze(-1)
    return tok.sum(-1)


def answer_stats(p, ref_lp):
    with torch.no_grad():
        lp = answer_logp(p).double()
    pr = lp.exp()
    q = pr / pr.sum()                       # renormalized over the 27 answers
    qr = ref_lp.exp() / ref_lp.exp().sum()
    kl = float((q * (q.log() - qr.log())).sum())
    return {"p_chosen": float(pr[CHOSEN]), "mass27": float(pr.sum()), "kl27": kl,
            "lp_chosen": float(lp[CHOSEN]), "lp_rej_mean": float((lp.sum() - lp[CHOSEN]) / 26)}


def answer_token_logp(p, ids=CONT_IDS):
    """log pi of each answer token (column, ' =', digit, ';') given everything before it."""
    lg = bm.forward(p, ids[:, :-1], H, L).log_softmax(-1)
    return lg[:, 4:8].gather(-1, ids[:, 5:9, None]).squeeze(-1)


def dpo(p0, beta, mask_shared=False, nll=0.0, steps=100, batch=64, lr=1e-3, seed=5):
    """DPO on pairs (chosen ' id = 7;', rejected: one of the other 26 answers).
    mask_shared drops the tokens every answer shares (' =' and ';') from both
    log-probabilities, as Llama 3 masks its header and termination tokens; nll adds
    nll x (mean token NLL of the chosen answer), as Llama 3 does with 0.2."""
    p, rng = clone(p0), random.Random(seed)
    opt = adam(p, lr)
    keep = torch.tensor([1.0, 0.0, 1.0, 0.0]) if mask_shared else torch.ones(4)
    with torch.no_grad():
        ref = (answer_token_logp(p0) * keep).sum(-1)
    others = [i for i in range(len(CONTS)) if i != CHOSEN]
    for _ in range(steps):
        rej = torch.tensor([rng.choice(others) for _ in range(batch)])
        tok = answer_token_logp(p)
        lp = (tok * keep).sum(-1)
        m = (lp[CHOSEN] - ref[CHOSEN]) - (lp[rej] - ref[rej])
        loss = -F.logsigmoid(beta * m).mean() + nll * (-tok[CHOSEN].mean())
        opt.zero_grad()
        loss.backward()
        opt.step()
    return p


def top_answers(p, k=3):
    """The k most probable 4-token answers to the prompt (beam search over the full vocabulary)."""
    beams = [(0.0, torch.tensor([enc(PROMPT)]))]
    for _ in range(4):
        cand = []
        for lp, s in beams:
            with torch.no_grad():
                lg = bm.forward(p, s, H, L)[0, -1].log_softmax(-1)
            v, i = lg.topk(8)
            cand += [(lp + a, torch.cat([s, torch.tensor([[b]])], 1)) for a, b in zip(v.tolist(), i.tolist())]
        beams = sorted(cand, key=lambda x: -x[0])[:16]
    return [["".join(D["vocab"][j] for j in s[0, 5:].tolist()), round(math.exp(lp), 4)] for lp, s in beams[:k]]


# ---------------------------------------------------------------- grpo with every parameter
def grpo(p0, updates=60, G=8, lr=1e-3, seed=3):
    """One group of G sampled answers per update, group-normalized advantages
    (population std), one gradient step per group (so the PPO ratio is 1 and the
    clip is inactive), no KL term. Reward 1 only for ' id = 7;'."""
    p, rng = clone(p0), torch.Generator().manual_seed(seed)
    opt = adam(p, lr)
    curve, zero_groups = [], 0
    pre = torch.tensor([enc(PROMPT)])
    for _ in range(updates):
        with torch.no_grad():
            curve.append(float(answer_logp(p)[CHOSEN].exp()))
            seqs = pre.repeat(G, 1)
            for t in range(4):                     # sample 4 answer tokens
                lg = bm.forward(p, seqs, H, L)[:, -1]
                nxt = torch.multinomial(lg.softmax(-1), 1, generator=rng)
                seqs = torch.cat([seqs, nxt], 1)
        r = torch.tensor([1.0 if s == enc(f"{PROMPT} id = 7;") else 0.0 for s in seqs.tolist()])
        sd = r.std(unbiased=False)
        if sd == 0:
            zero_groups += 1
            continue
        A = (r - r.mean()) / sd
        lg = bm.forward(p, seqs[:, :-1], H, L).log_softmax(-1)
        lp = lg[:, 4:8].gather(-1, seqs[:, 5:9, None]).squeeze(-1).sum(-1)
        loss = -(A * lp).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    curve.append(float(answer_logp(p)[CHOSEN].exp().detach()))
    return curve, zero_groups


def p_next(p, text, tok):
    ids = torch.tensor([enc(text)])
    with torch.no_grad():
        return float(bm.forward(p, ids, H, L)[0, -1].softmax(-1)[STOI[tok]])


def main():
    p0 = load_weights()
    out = {}
    runs = {"base": p0, "narrow": finetune(p0, mix=0.0), "mix25": finetune(p0, mix=0.25)}
    out["forget"] = {k: dict(table_losses(p), p_users=p_next(p, "SELECT name FROM", " users")) for k, p in runs.items()}
    out["forget"]["setup"] = {"steps": 300, "batch": 128, "lr": 1e-3, "mix": 0.25}
    with torch.no_grad():
        ref = answer_logp(p0).double()
    out["dpo"] = {"ref": dict(answer_stats(p0, ref), top=top_answers(p0))}
    for name, kw in [("plain", {}), ("nll", {"nll": 0.2}), ("mask", {"mask_shared": True}), ("mask_nll", {"mask_shared": True, "nll": 0.2}),
                     ("plain_b0.5", {"beta_": 0.5}), ("mask_b0.05", {"mask_shared": True, "beta_": 0.05}),
                     ("mask_b0.5", {"mask_shared": True, "beta_": 0.5}), ("mask_b1", {"mask_shared": True, "beta_": 1.0}),
                     ("plain_lr1e-4", {"lr": 1e-4}), ("mask_lr1e-4", {"mask_shared": True, "lr": 1e-4})]:
        beta = kw.pop("beta_", 0.1)
        pd = dpo(p0, beta, **kw)
        out["dpo"][name] = dict(answer_stats(pd, ref), beta=beta, top=top_answers(pd))
    curve, zero = grpo(p0)
    out["grpo"] = {"start": curve[0], "end": curve[-1], "updates": len(curve) - 1, "zero_groups": zero}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
