"""Part 3: PyTorch behavior quoted on the sheets, recomputed with PyTorch itself.

Run with the project's venv (numpy + torch):
    python tools/verify/part03_torch.py
It prints one JSON object. Its values are pasted into TORCH in part03.mjs, which
checks them against the sheet text and against independent JavaScript math.
"""
import json
import math

import torch
import torch.nn.functional as F

torch.set_num_threads(1)
out = {"torch_version": torch.__version__}

# ------------------------------------------------------------ nn-mlp-code: the 2-3-1 network on a batch
X = torch.tensor([[1., 2.], [-1., 1.], [2., -1.]])
W1 = torch.tensor([[0.5, -0.5, 1.0], [0.2, 0.5, -1.0]])
b1 = torch.tensor([0.1, -0.2, 0.3])
W2 = torch.tensor([[0.6], [-0.4], [0.9]])
b2 = torch.tensor([0.05])
H = torch.relu(X @ W1 + b1)
Y = H @ W2 + b2
out["mlp_print"] = repr(Y.squeeze())
out["mlp_H"] = H.tolist()

# ------------------------------------------------------------ nn-bp-2layer: autograd in float64
d = torch.float64
p = {
    "W1": torch.tensor([[0.5, -0.5, 1.0], [0.2, 0.5, -1.0]], dtype=d, requires_grad=True),
    "b1": torch.tensor([0.1, -0.2, 0.3], dtype=d, requires_grad=True),
    "W2": torch.tensor([0.6, -0.4, 0.9], dtype=d, requires_grad=True),
    "b2": torch.tensor(0.05, dtype=d, requires_grad=True),
}
x = torch.tensor([1., 2.], dtype=d)
z = x @ p["W1"] + p["b1"]
h = torch.relu(z)
yh = h @ p["W2"] + p["b2"]
L = (yh - 1.0) ** 2
L.backward()
out["twolayer_L"] = L.item()
out["twolayer_grad"] = p["W1"].grad.flatten().tolist() + p["b1"].grad.tolist() + p["W2"].grad.tolist() + [p["b2"].grad.item()]

# ------------------------------------------------------------ nn-bp-relu: slope at exactly 0, what ReLU and GELU save
zz = torch.tensor([-1.0, 0.0, 2.0], requires_grad=True)
torch.relu(zz).sum().backward()
out["relu_grad_at_0"] = zz.grad.tolist()
a = torch.randn(3, requires_grad=True)
r = torch.relu(a)
g = F.gelu(a, approximate="tanh")
out["relu_saves"] = sorted(k for k in dir(r.grad_fn) if k.startswith("_saved_"))
out["relu_saves_output"] = r.grad_fn._saved_result.data_ptr() == r.data_ptr()
out["gelu_saves"] = sorted(k for k in dir(g.grad_fn) if k.startswith("_saved_"))

# ------------------------------------------------------------ nn-bp-memory: tensors saved by our MLP block, 1,024 tokens
torch.manual_seed(0)
T, dm, ff = 1024, 16, 64
resid = torch.randn(T, dm, requires_grad=True)
n2 = torch.ones(dm, requires_grad=True)
w1 = (torch.randn(dm, ff) * dm ** -0.5).requires_grad_()
w2 = (torch.randn(ff, dm) * ff ** -0.5).requires_grad_()
h2 = resid * torch.rsqrt(resid.pow(2).mean(-1, keepdim=True) + 1e-5) * n2  # stands in for the block's normalized input
params = {w1.data_ptr(), w2.data_ptr()}
saved = {}

def pack(t):
    if t.data_ptr() not in params:
        saved[t.data_ptr()] = (tuple(t.shape), t.numel(), t.element_size())
    return t

with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
    act = F.gelu(h2 @ w1, approximate="tanh")
    mlp_out = act @ w2
out["mlp_saved_shapes"] = sorted([list(v[0]) for v in saved.values()])
out["mlp_saved_numbers"] = sum(v[1] for v in saved.values())
out["mlp_saved_bytes"] = sum(v[1] * v[2] for v in saved.values())
saved.clear()
with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
    act = torch.relu(h2 @ w1)
    mlp_out = act @ w2
out["mlp_saved_numbers_relu"] = sum(v[1] for v in saved.values())

# ------------------------------------------------------------ nn-tensors
X = torch.tensor([[1., 2.], [-1., 1.], [2., -1.]])
out["tensor_shape"] = repr(X.shape)
out["tensor_dtype"] = repr(X.dtype)
out["tensor_bytes"] = X.element_size() * X.numel()
torch.manual_seed(1)
Wr = torch.randn(2, 3)
out["matmul_shape"] = repr((X @ Wr).shape)
out["broadcast_shape"] = repr((X @ Wr + torch.zeros(3)).shape)
out["bf16_size"] = X.to(torch.bfloat16).element_size()
wte = torch.zeros(113, 16)
out["wte_bytes"] = wte.numel() * wte.element_size()
# the broadcasting rule: sizes compared from the last dimension, equal or one of them 1
out["broadcast_31_13"] = list((torch.zeros(3, 1) + torch.zeros(1, 3)).shape)
out["broadcast_3_33"] = list((torch.zeros(3, 3) + torch.zeros(3)).shape)
try:
    torch.zeros(3, 2) + torch.zeros(3)
    out["broadcast_32_3_error"] = None
except RuntimeError as e:
    out["broadcast_32_3_error"] = str(e).split("\n")[0]

# ------------------------------------------------------------ nn-autograd
w = torch.tensor(0.8, requires_grad=True)
b = torch.tensor(0.2, requires_grad=True)
x, y = torch.tensor(1.5), torch.tensor(2.0)
L = (w * x + b - y) ** 2
out["autograd_L_repr"] = repr(L)
out["autograd_next"] = [(type(f).__name__ if f is not None else None, i) for f, i in L.grad_fn.next_functions]
out["autograd_next2"] = [(type(f).__name__ if f is not None else None, i) for f, i in L.grad_fn.next_functions[0][0].next_functions]
L.backward()
out["autograd_grads"] = [repr(w.grad), repr(b.grad)]

# ------------------------------------------------------------ nn-step: one SGD step, by hand and with torch.optim.SGD
with torch.no_grad():
    w2_, b2_ = w.clone(), b.clone()
    w -= 0.1 * w.grad
    b -= 0.1 * b.grad
L2 = (w * x + b - y) ** 2
out["sgd_manual"] = [w.item(), b.item()]
out["sgd_loss_repr"] = repr(L2)
wo = w2_.clone().requires_grad_()
bo = b2_.clone().requires_grad_()
Lo = (wo * x + bo - y) ** 2
Lo.backward()
opt = torch.optim.SGD([wo, bo], lr=0.1)
opt.step()
out["sgd_optim"] = [wo.item(), bo.item()]

# nn-step, the line "with torch.no_grad():": the update without it
wn = torch.tensor(0.8, requires_grad=True)
((wn * x - y) ** 2).backward()
try:
    wn -= 0.1 * wn.grad
    out["inplace_error"] = None
except RuntimeError as e:
    out["inplace_error"] = str(e).split("\n")[0]
wn2 = torch.tensor(0.8, requires_grad=True)
((wn2 * x - y) ** 2).backward()
wn2 = wn2 - 0.1 * wn2.grad  # out of place: a new tensor computed from the old one
out["outofplace_is_leaf"] = wn2.is_leaf
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    ((wn2 * x - y) ** 2).backward()
    out["outofplace_grad_is_none"] = wn2.grad is None

# ------------------------------------------------------------ nn-accum: .grad accumulates; micro-batches equal one batch
w = torch.tensor(0.8, requires_grad=True)
b = torch.tensor(0.2, requires_grad=True)
((w * x + b - y) ** 2).backward()
((w * x + b - y) ** 2).backward()
out["accum_twice"] = repr(w.grad)
opt = torch.optim.SGD([w, b], lr=0.1)
opt.zero_grad()
out["zero_grad_sets_none"] = w.grad is None

torch.manual_seed(0)
model = torch.nn.Sequential(torch.nn.Linear(16, 64), torch.nn.GELU(approximate="tanh"), torch.nn.Linear(64, 113))
Xb = torch.randn(128, 16)
Yb = torch.randint(0, 113, (128,))
model.zero_grad()
F.cross_entropy(model(Xb), Yb).backward()
full = [q.grad.clone() for q in model.parameters()]
model.zero_grad()
for i in range(4):
    (F.cross_entropy(model(Xb[32 * i:32 * (i + 1)]), Yb[32 * i:32 * (i + 1)]) / 4).backward()
acc = [q.grad.clone() for q in model.parameters()]
out["accum_max_abs_diff"] = max((u - v).abs().max().item() for u, v in zip(full, acc))
out["accum_max_abs_grad"] = max(u.abs().max().item() for u in full)
# the same in float64: equal to rounding of float64
model64 = model.double()
model64.zero_grad()
F.cross_entropy(model64(Xb.double()), Yb).backward()
full64 = [q.grad.clone() for q in model64.parameters()]
model64.zero_grad()
for i in range(4):
    (F.cross_entropy(model64(Xb.double()[32 * i:32 * (i + 1)]), Yb[32 * i:32 * (i + 1)]) / 4).backward()
out["accum_max_abs_diff_f64"] = max((u - q.grad).abs().max().item() for u, q in zip(full64, model64.parameters()))

# ------------------------------------------------------------ nn-step: Adam state size for our model's parameter shapes
shapes = [(113, 16), (8, 16)] + [s for _ in range(2) for s in [(16,), (16, 16), (16, 16), (16, 16), (16, 16), (16,), (16, 64), (64, 16)]] + [(16,), (16, 113)]
ps = [torch.zeros(*s, requires_grad=True) for s in shapes]
for q in ps:
    q.grad = torch.ones_like(q)
opt = torch.optim.AdamW(ps, lr=0.01, betas=(0.9, 0.95), weight_decay=0.0)
opt.step()
out["n_params"] = sum(q.numel() for q in ps)
out["adam_state_numbers"] = sum(v.numel() for st in opt.state.values() for k, v in st.items() if k in ("exp_avg", "exp_avg_sq"))
out["adam_state_keys"] = sorted(next(iter(opt.state.values())).keys())

# ------------------------------------------------------------ 3.5: momentum, Adam, AdamW, clipping on L = w1^2 + 10 w2^2
def valley(wv):
    return wv[0] ** 2 + 10 * wv[1] ** 2

def run(opt_fn, steps, start=(-4.0, 1.5)):
    wv = torch.tensor(start, dtype=torch.float64, requires_grad=True)
    o = opt_fn([wv])
    path = []
    for _ in range(steps):
        o.zero_grad()
        valley(wv).backward()
        o.step()
        path.append(wv.detach().tolist())
    return path, o

path, _ = run(lambda q: torch.optim.SGD(q, lr=0.09, momentum=0.5), 3)
out["momentum_3"] = path
path, _ = run(lambda q: torch.optim.SGD(q, lr=0.09), 3)
out["gd_3"] = path
path, o = run(lambda q: torch.optim.Adam(q, lr=0.1, betas=(0.9, 0.95), eps=1e-8), 3)
out["adam_3"] = path
st = next(iter(o.state.values()))
out["adam_3_m"] = st["exp_avg"].tolist()
out["adam_3_v"] = st["exp_avg_sq"].tolist()

def steps_to(opt_fn, thr, kappa=10.0, start=(-4.0, 1.5), maxsteps=20000):
    wv = torch.tensor(start, dtype=torch.float64, requires_grad=True)
    o = opt_fn([wv])
    for t in range(1, maxsteps + 1):
        o.zero_grad()
        (wv[0] ** 2 + kappa * wv[1] ** 2).backward()
        o.step()
        if (wv[0] ** 2 + kappa * wv[1] ** 2).item() < thr:
            return t
    return None

out["steps_gd"] = steps_to(lambda q: torch.optim.SGD(q, lr=0.09), 1e-3)
out["steps_mom"] = {str(bb): steps_to(lambda q, bb=bb: torch.optim.SGD(q, lr=0.09, momentum=bb), 1e-3) for bb in (0.3, 0.4, 0.5, 0.7, 0.9)}
out["steps_k100_gd"] = steps_to(lambda q: torch.optim.SGD(q, lr=0.0099), 1e-3, kappa=100.0)
out["steps_k100_mom09"] = steps_to(lambda q: torch.optim.SGD(q, lr=0.0099, momentum=0.9), 1e-3, kappa=100.0)

# AdamW vs Adam + L2 for a weight with zero loss gradient and one with a steady gradient of 10
def decay_run(cls, g, steps, **kw):
    wv = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
    o = cls([wv], lr=0.01, betas=(0.9, 0.95), eps=1e-8, **kw)
    for _ in range(steps):
        o.zero_grad()
        wv.grad = torch.tensor([g], dtype=torch.float64)
        o.step()
    return wv.item()

out["l2_g0_100"] = decay_run(torch.optim.Adam, 0.0, 100, weight_decay=0.1)
out["l2_g0_100_small"] = decay_run(torch.optim.Adam, 0.0, 100, weight_decay=0.001)
out["l2_g0_1"] = decay_run(torch.optim.Adam, 0.0, 1, weight_decay=0.1)
out["adamw_g0_1"] = decay_run(torch.optim.AdamW, 0.0, 1, weight_decay=0.1)
out["adamw_g0_100"] = decay_run(torch.optim.AdamW, 0.0, 100, weight_decay=0.1)
out["l2_g10_1"] = decay_run(torch.optim.Adam, 10.0, 1, weight_decay=0.1)
out["adam_g10_1"] = decay_run(torch.optim.Adam, 10.0, 1, weight_decay=0.0)
out["adamw_g10_1"] = decay_run(torch.optim.AdamW, 10.0, 1, weight_decay=0.1)
out["l2_g10_100"] = decay_run(torch.optim.Adam, 10.0, 100, weight_decay=0.1)
out["adam_g10_100"] = decay_run(torch.optim.Adam, 10.0, 100, weight_decay=0.0)
out["adamw_g10_100"] = decay_run(torch.optim.AdamW, 10.0, 100, weight_decay=0.1)

# clip_grad_norm_: two parameters with gradients 1.5 and 2, max_norm 1
pa = torch.zeros(1, requires_grad=True)
pb = torch.zeros(1, requires_grad=True)
pa.grad = torch.tensor([1.5])
pb.grad = torch.tensor([2.0])
total = torch.nn.utils.clip_grad_norm_([pa, pb], max_norm=1.0)
out["clip_total"] = total.item()
out["clip_after"] = pa.grad.tolist() + pb.grad.tolist()

# ------------------------------------------------------------ 3.5: activation std through 10 layers with torch's own random numbers
def depth_std(act, gain, n=512, B=512, L=10, seed=0):
    gen = torch.Generator().manual_seed(seed)
    hcur = torch.randn(B, n, generator=gen, dtype=torch.float64)
    stds = []
    for _ in range(L):
        Wl = torch.randn(n, n, generator=gen, dtype=torch.float64) * gain / math.sqrt(n)
        zl = hcur @ Wl
        stds.append(zl.std().item())
        hcur = {"linear": zl, "relu": torch.relu(zl), "tanh": torch.tanh(zl), "sigmoid": torch.sigmoid(zl)}[act]
    return stds

out["torch_depth"] = {f"{a}_{g:.3f}": depth_std(a, g) for a, g in [("linear", 0.5), ("linear", 1.0), ("linear", 2.0), ("relu", 1.0), ("relu", math.sqrt(2)), ("tanh", 1.0)]}

print(json.dumps(out, indent=1))
