"""GPT-2 small in pure NumPy: every layer with a hand-written backward pass,
plus a loader for OpenAI's pretrained weights. Built in notebooks 01-05; notebook 06 imports it."""
import math
import numpy as np


# ---------------------------------------------------------------- helpers
def numerical_grad(f, x, eps=1e-6):
    """Central-difference gradient of scalar f() w.r.t. array x (perturbed in place)."""
    grad = np.zeros_like(x, dtype=np.float64)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        i = it.multi_index
        old = x[i]
        x[i] = old + eps; fp = f()
        x[i] = old - eps; fm = f()
        x[i] = old
        grad[i] = (fp - fm) / (2 * eps)
        it.iternext()
    return grad


def rel_error(a, b):
    return np.linalg.norm(a - b) / max(1e-12, np.linalg.norm(a) + np.linalg.norm(b))


# ---------------------------------------------------------------- layers
class Linear:
    def __init__(self, in_f, out_f, bias=True, std=0.02, rng=None, dtype=np.float64):
        rng = rng if rng is not None else np.random.default_rng()
        self.params = {"W": rng.normal(0.0, std, (in_f, out_f)).astype(dtype)}
        if bias:
            self.params["b"] = np.zeros(out_f, dtype=dtype)
        self.grads = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.x = None

    def forward(self, x):
        self.x = x
        out = x @ self.params["W"]
        if "b" in self.params:
            out = out + self.params["b"]
        return out

    def backward(self, dout):
        x2 = self.x.reshape(-1, self.x.shape[-1])
        d2 = dout.reshape(-1, dout.shape[-1])
        self.grads["W"] += x2.T @ d2
        if "b" in self.params:
            self.grads["b"] += d2.sum(axis=0)
        return dout @ self.params["W"].T

    def parameters(self, prefix=""):
        return [(prefix + k, self.params[k], self.grads[k]) for k in self.params]


class GELU:
    C = math.sqrt(2.0 / math.pi)        # Python float: keeps float32 arrays float32

    def forward(self, x):
        self.x = x
        self.t = np.tanh(self.C * (x + 0.044715 * x * x * x))
        return 0.5 * x * (1.0 + self.t)

    def backward(self, dout):
        x, t = self.x, self.t
        du = self.C * (1.0 + 3 * 0.044715 * x * x)
        return dout * (0.5 * (1.0 + t) + 0.5 * x * (1.0 - t**2) * du)

    def parameters(self, prefix=""):
        return []


class MLP:
    def __init__(self, C, n_layer, rng, dtype=np.float64):
        self.c_fc = Linear(C, 4 * C, rng=rng, dtype=dtype)
        self.gelu = GELU()
        self.c_proj = Linear(4 * C, C, std=0.02 / math.sqrt(2 * n_layer), rng=rng, dtype=dtype)

    def forward(self, x):
        return self.c_proj.forward(self.gelu.forward(self.c_fc.forward(x)))

    def backward(self, dout):
        return self.c_fc.backward(self.gelu.backward(self.c_proj.backward(dout)))

    def parameters(self, prefix=""):
        return self.c_fc.parameters(prefix + "c_fc.") + self.c_proj.parameters(prefix + "c_proj.")


class Softmax:
    def forward(self, x):
        z = x - x.max(axis=-1, keepdims=True)
        e = np.exp(z)
        self.s = e / e.sum(axis=-1, keepdims=True)
        return self.s

    def backward(self, dout):
        s = self.s
        return s * (dout - (dout * s).sum(axis=-1, keepdims=True))


class CrossEntropyLoss:
    def forward(self, logits, targets):
        B, T, V = logits.shape
        N = B * T
        l = logits.reshape(N, V)
        y = targets.reshape(N)
        m = l.max(axis=1, keepdims=True)
        lse = m + np.log(np.exp(l - m).sum(axis=1, keepdims=True))
        logp = l - lse
        self.probs = np.exp(logp)
        self.y = y
        self.shape = (B, T, V)
        return -logp[np.arange(N), y].mean()

    def backward(self):
        N = self.y.shape[0]
        d = self.probs.copy()
        d[np.arange(N), self.y] -= 1.0
        d /= N
        return d.reshape(self.shape)


class LayerNorm:
    def __init__(self, C, eps=1e-5, dtype=np.float64):
        self.eps = eps
        self.params = {"g": np.ones(C, dtype=dtype), "b": np.zeros(C, dtype=dtype)}
        self.grads = {k: np.zeros_like(v) for k, v in self.params.items()}

    def forward(self, x):
        mu = x.mean(axis=-1, keepdims=True)
        xc = x - mu
        var = (xc**2).mean(axis=-1, keepdims=True)
        self.rstd = 1.0 / np.sqrt(var + self.eps)
        self.xhat = xc * self.rstd
        return self.xhat * self.params["g"] + self.params["b"]

    def backward(self, dout):
        C = dout.shape[-1]
        self.grads["g"] += (dout * self.xhat).reshape(-1, C).sum(axis=0)
        self.grads["b"] += dout.reshape(-1, C).sum(axis=0)
        h = dout * self.params["g"]
        return self.rstd * (h - h.mean(axis=-1, keepdims=True)
                            - self.xhat * (h * self.xhat).mean(axis=-1, keepdims=True))

    def parameters(self, prefix=""):
        return [(prefix + k, self.params[k], self.grads[k]) for k in self.params]


class CausalSelfAttention:
    def __init__(self, C, n_head, n_layer, rng, dtype=np.float64):
        assert C % n_head == 0
        self.C, self.H = C, n_head
        self.c_attn = Linear(C, 3 * C, rng=rng, dtype=dtype)
        self.c_proj = Linear(C, C, std=0.02 / math.sqrt(2 * n_layer), rng=rng, dtype=dtype)
        self.softmax = Softmax()

    def forward(self, x):
        B, T, C = x.shape
        H, hs = self.H, C // self.H
        qkv = self.c_attn.forward(x)
        q, k, v = np.split(qkv, 3, axis=-1)
        q = q.reshape(B, T, H, hs).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, H, hs).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, H, hs).transpose(0, 2, 1, 3)
        self.scale = 1.0 / math.sqrt(hs)    # Python float (np.sqrt would give float64 and upcast)
        att = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        att = np.where(mask, -np.inf, att)
        A = self.softmax.forward(att)
        y = A @ v
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)
        self.q, self.k, self.v, self.A = q, k, v, A
        return self.c_proj.forward(y)

    def backward(self, dout):
        B, T, C = dout.shape
        H, hs = self.H, C // self.H
        dy = self.c_proj.backward(dout)
        dy = dy.reshape(B, T, H, hs).transpose(0, 2, 1, 3)
        dA = dy @ self.v.transpose(0, 1, 3, 2)
        dv = self.A.transpose(0, 1, 3, 2) @ dy
        datt = self.softmax.backward(dA) * self.scale
        dq = datt @ self.k
        dk = datt.transpose(0, 1, 3, 2) @ self.q
        merge = lambda t: t.transpose(0, 2, 1, 3).reshape(B, T, C)
        dqkv = np.concatenate([merge(dq), merge(dk), merge(dv)], axis=-1)
        return self.c_attn.backward(dqkv)

    def parameters(self, prefix=""):
        return self.c_attn.parameters(prefix + "c_attn.") + self.c_proj.parameters(prefix + "c_proj.")


class Embedding:
    def __init__(self, num, C, std=0.02, rng=None, dtype=np.float64):
        rng = rng if rng is not None else np.random.default_rng()
        self.params = {"W": rng.normal(0.0, std, (num, C)).astype(dtype)}
        self.grads = {"W": np.zeros_like(self.params["W"])}

    def forward(self, idx):
        self.idx = idx
        return self.params["W"][idx]

    def backward(self, dout):
        np.add.at(self.grads["W"], self.idx, dout)

    def parameters(self, prefix=""):
        return [(prefix + "W", self.params["W"], self.grads["W"])]


class Block:
    def __init__(self, C, n_head, n_layer, rng, dtype=np.float64):
        self.ln_1 = LayerNorm(C, dtype=dtype)
        self.attn = CausalSelfAttention(C, n_head, n_layer, rng, dtype=dtype)
        self.ln_2 = LayerNorm(C, dtype=dtype)
        self.mlp = MLP(C, n_layer, rng, dtype=dtype)

    def forward(self, x):
        x = x + self.attn.forward(self.ln_1.forward(x))
        x = x + self.mlp.forward(self.ln_2.forward(x))
        return x

    def backward(self, dout):
        dx = dout + self.ln_2.backward(self.mlp.backward(dout))
        dx = dx + self.ln_1.backward(self.attn.backward(dx))
        return dx

    def parameters(self, prefix=""):
        return (self.ln_1.parameters(prefix + "ln_1.") + self.attn.parameters(prefix + "attn.")
                + self.ln_2.parameters(prefix + "ln_2.") + self.mlp.parameters(prefix + "mlp."))


class GPT:
    def __init__(self, vocab_size=50257, block_size=1024, n_layer=12, n_head=12, n_embd=768,
                 seed=0, dtype=np.float64):
        rng = np.random.default_rng(seed)
        self.block_size = block_size
        self.wte = Embedding(vocab_size, n_embd, rng=rng, dtype=dtype)
        self.wpe = Embedding(block_size, n_embd, rng=rng, dtype=dtype)
        self.h = [Block(n_embd, n_head, n_layer, rng, dtype=dtype) for _ in range(n_layer)]
        self.ln_f = LayerNorm(n_embd, dtype=dtype)

    def forward(self, idx):
        B, T = idx.shape
        assert T <= self.block_size
        x = self.wte.forward(idx) + self.wpe.forward(np.arange(T))
        for block in self.h:
            x = block.forward(x)
        x = self.ln_f.forward(x)
        self.x_final = x
        return x @ self.wte.params["W"].T

    def backward(self, dlogits):
        W = self.wte.params["W"]
        V, C = W.shape
        self.wte.grads["W"] += dlogits.reshape(-1, V).T @ self.x_final.reshape(-1, C)
        dx = dlogits @ W
        dx = self.ln_f.backward(dx)
        for block in reversed(self.h):
            dx = block.backward(dx)
        self.wte.backward(dx)
        self.wpe.backward(dx.sum(axis=0))

    def parameters(self):
        ps = self.wte.parameters("wte.") + self.wpe.parameters("wpe.")
        for i, block in enumerate(self.h):
            ps += block.parameters(f"h.{i}.")
        return ps + self.ln_f.parameters("ln_f.")

    def zero_grad(self):
        for _, _, g in self.parameters():
            g.fill(0.0)


# ---------------------------------------------------------------- pretrained weights
def load_safetensors(path):
    """Minimal .safetensors reader using only NumPy.
    File layout: 8-byte little-endian header length | JSON header | raw tensor bytes."""
    import json
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(n))
    raw = np.memmap(path, dtype=np.uint8, mode="r")
    dtypes = {"F32": np.float32, "F16": np.float16, "F64": np.float64}
    out = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        if info["dtype"] not in dtypes:
            raise ValueError(f"{name}: unsupported dtype {info['dtype']}")
        start, end = info["data_offsets"]
        out[name] = raw[8 + n + start: 8 + n + end].view(dtypes[info["dtype"]]).reshape(info["shape"])
    return out


def _hf_name_to_ours(k):
    """'h.3.attn.c_attn.weight' -> 'h.3.attn.c_attn.W', 'h.3.ln_1.weight' -> 'h.3.ln_1.g', etc."""
    if k.startswith("transformer."):
        k = k[len("transformer."):]
    parts = k.split(".")
    if k == "lm_head.weight" or parts[-2:] in (["attn", "bias"], ["attn", "masked_bias"]):
        return None                      # tied head / causal-mask buffers: not parameters
    stem, kind = ".".join(parts[:-1]), parts[-1]
    if kind == "weight":
        return stem + (".g" if parts[-2].startswith("ln_") else ".W")
    if kind == "bias":
        return stem + ".b"
    raise KeyError(k)


def load_hf_gpt2(model, path):
    """Copy HuggingFace GPT-2 weights into our model IN PLACE. Their Conv1D weights are stored
    (in, out), exactly like our x @ W, so no transposes are needed."""
    sd = load_safetensors(path)
    ours = {name: p for name, p, _ in model.parameters()}
    loaded = set()
    for k, arr in sd.items():
        name = _hf_name_to_ours(k)
        if name is None:
            continue
        p = ours[name]
        assert p.shape == arr.shape, f"{k}: file {arr.shape} vs model {p.shape}"
        p[...] = arr
        loaded.add(name)
    missing = set(ours) - loaded
    assert not missing, f"not found in file: {sorted(missing)[:5]}"
    return len(loaded)