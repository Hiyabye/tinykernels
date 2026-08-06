#!/usr/bin/env python3
"""Cross-check the C forward pass logits against an independent numpy re-reading
of the same GGUF.

C writes the full logits matrix to results/data/forward_logits.bin (float32,
rows = tokens, cols = vocab). This script re-reads the GGUF, re-computes the
logits in numpy (clean-room Qwen3 math), and reports the max abs difference and
the argmax agreement per position. It tokenizes via the C binary so both sides
see identical input ids.

Usage: python3 scripts/verify_forward.py [prompt]

Exits 0 when the max abs logit difference is within TOL.
"""
import numpy as np
import struct
import subprocess
import sys

MODEL = "Qwen3-0.6B-Q8_0.gguf"
BIN = "./tinykernels"
TOL = 1e-3  # Q8_0 weights + different float32 accumulation order between C and numpy

PROMPT = sys.argv[1] if len(sys.argv) > 1 else "The capital of France is"


# ---------------- GGUF reader (numpy, no deps) ----------------
def read_str(f):
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n).decode("utf-8", "replace")


def load(path):
    f = open(path, "rb")
    assert f.read(4) == b"GGUF"
    f.read(4)  # version
    n_tensors = struct.unpack("<Q", f.read(8))[0]
    n_kv = struct.unpack("<Q", f.read(8))[0]
    meta = {}
    for _ in range(n_kv):
        key = read_str(f)
        t = struct.unpack("<I", f.read(4))[0]
        if t == 8:
            meta[key] = read_str(f)
        elif t == 9:
            et = struct.unpack("<I", f.read(4))[0]
            cnt = struct.unpack("<Q", f.read(8))[0]
            if et == 8:
                meta[key] = [read_str(f) for _ in range(cnt)]
            else:
                sz = {0: 1, 1: 1, 7: 1, 2: 2, 3: 2, 6: 4, 4: 4, 5: 4, 10: 8, 11: 8, 12: 8}[et]
                f.read(sz * cnt)
        else:
            if t in (0, 1):  # uint8/int8 -> int
                meta[key] = struct.unpack("<b", f.read(1))[0]
            elif t in (2, 3):
                meta[key] = struct.unpack("<h", f.read(2))[0]
            elif t in (4, 5, 6):
                meta[key] = struct.unpack("<I", f.read(4))[0] if t in (4, 5) else struct.unpack("<f", f.read(4))[0]
            elif t in (10, 11):
                meta[key] = struct.unpack("<Q", f.read(8))[0]
            elif t == 12:
                meta[key] = struct.unpack("<d", f.read(8))[0]
            elif t == 7:
                meta[key] = bool(struct.unpack("<B", f.read(1))[0])
            else:
                raise ValueError("type %d" % t)
    tensors = {}
    for _ in range(n_tensors):
        name = read_str(f)
        nd = struct.unpack("<I", f.read(4))[0]
        dims = struct.unpack("<%dQ" % nd, f.read(8 * nd))
        typ = struct.unpack("<I", f.read(4))[0]
        off = struct.unpack("<Q", f.read(8))[0]
        tensors[name] = (dims, typ, off)
    pos = f.tell()
    data_start = (pos + 31) // 32 * 32
    return f, meta, tensors, data_start


def dequant(f, tensors, data_start, name):
    dims, typ, off = tensors[name]
    n = int(np.prod(dims))
    f.seek(data_start + off)
    if typ == 0:
        return np.frombuffer(f.read(4 * n), dtype="<f4").astype(np.float32)
    if typ == 8:  # Q8_0
        nblk = n // 32
        d = np.frombuffer(f.read(nblk * 34), dtype=np.uint8).reshape(nblk, 34)
        u = d[:, 0].astype(np.uint16) | (d[:, 1].astype(np.uint16) << 8)
        scales = u.view(np.float16).astype(np.float32)
        q = d[:, 2:].astype(np.int8)
        return (scales[:, None] * q).reshape(n)
    raise ValueError("unsupported type %d" % typ)


# ---------------- Qwen3 forward (numpy) ----------------
def rms_norm(x, w, eps):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * w


def softmax_rows(a):
    a = a - a.max(axis=-1, keepdims=True)
    e = np.exp(a)
    return e / e.sum(axis=-1, keepdims=True)


def main():
    f, meta, tensors, data_start = load(MODEL)

    hidden = meta["qwen3.embedding_length"]
    nlayers = meta["qwen3.block_count"]
    n_heads = meta["qwen3.attention.head_count"]
    n_kv = meta["qwen3.attention.head_count_kv"]
    hd = meta["qwen3.attention.key_length"]
    inter = meta["qwen3.feed_forward_length"]
    vocab = len(meta["tokenizer.ggml.tokens"])
    eps = meta["qwen3.attention.layer_norm_rms_epsilon"]
    theta = meta["qwen3.rope.freq_base"]

    # run the C forward (writes results/data/forward_logits.bin) and fetch the
    # input ids from the tokenizer so both sides see identical tokens.
    out = subprocess.run([BIN, "forward", PROMPT], capture_output=True)
    if out.returncode != 0:
        sys.stderr.write(out.stderr.decode())
        sys.exit(2)
    tok = subprocess.run([BIN, "tokenize", PROMPT], capture_output=True, text=True)
    ids = np.array([int(x) for x in tok.stdout.split()], dtype=np.uint32)

    qwidth, kvwidth = n_heads * hd, n_kv * hd

    def w(name, out_d, in_d):
        return dequant(f, tensors, data_start, name).reshape(out_d, in_d)

    emb_wt = dequant(f, tensors, data_start, "token_embd.weight").reshape(vocab, hidden).T
    H = emb_wt[:, ids].T.astype(np.float32)  # (T, hidden)

    # rotary cos/sin tables
    half = hd // 2
    inv = theta ** (-np.arange(0, hd, 2) / hd)
    pos = np.arange(H.shape[0])[:, None]
    freqs = pos * inv[None, :]
    cosm, sinn = np.cos(freqs), np.sin(freqs)

    def rope_apply(x):  # x: (T, nheads, hd)
        a, b = x[..., :half], x[..., half:]
        c, s = cosm[:, None, :], sinn[:, None, :]
        return np.concatenate([c * a - s * b, s * a + c * b], axis=-1)

    T = H.shape[0]
    for l in range(nlayers):
        # attention
        xn = rms_norm(H, dequant(f, tensors, data_start, "blk.%d.attn_norm.weight" % l).astype(np.float32).reshape(hidden), eps)
        q = xn @ w("blk.%d.attn_q.weight" % l, qwidth, hidden).T
        k = xn @ w("blk.%d.attn_k.weight" % l, kvwidth, hidden).T
        v = (xn @ w("blk.%d.attn_v.weight" % l, kvwidth, hidden).T).reshape(T, n_kv, hd)
        q = rope_apply(rms_norm(q.reshape(T, n_heads, hd), dequant(f, tensors, data_start, "blk.%d.attn_q_norm.weight" % l).astype(np.float32).reshape(hd), eps))
        k = rope_apply(rms_norm(k.reshape(T, n_kv, hd), dequant(f, tensors, data_start, "blk.%d.attn_k_norm.weight" % l).astype(np.float32).reshape(hd), eps))

        o = np.zeros((T, n_heads, hd), dtype=np.float32)
        for h in range(n_heads):
            kvh = h // (n_heads // n_kv)
            sc = q[:, h] @ k[:, kvh].T / np.sqrt(hd)
            sc[np.triu_indices(T, 1)] = -np.inf
            o[:, h] = softmax_rows(sc) @ v[:, kvh]
        H += o.reshape(T, qwidth) @ w("blk.%d.attn_output.weight" % l, hidden, qwidth).T

        # MLP
        xf = rms_norm(H, dequant(f, tensors, data_start, "blk.%d.ffn_norm.weight" % l).astype(np.float32).reshape(hidden), eps)
        gate = xf @ w("blk.%d.ffn_gate.weight" % l, inter, hidden).T
        up = xf @ w("blk.%d.ffn_up.weight" % l, inter, hidden).T
        with np.errstate(over="ignore"):  # silu saturates to 0 like the C float path
            act = (gate / (1.0 + np.exp(-gate))) * up
        H += act @ w("blk.%d.ffn_down.weight" % l, hidden, inter).T

    hf = rms_norm(H, dequant(f, tensors, data_start, "output_norm.weight").astype(np.float32).reshape(hidden), eps)
    ref = hf @ emb_wt  # (T, vocab)

    got = np.fromfile("results/data/forward_logits.bin", dtype="<f4").reshape(T, vocab)

    diff = np.abs(got - ref)
    maxd = diff.max()
    agree = (np.argmax(got, axis=1) == np.argmax(ref, axis=1)).sum()
    print("prompt tokens (%d): %s" % (T, " ".join(map(str, ids.tolist()))))
    print("per-position max abs diff:", ["%.2e" % d for d in diff.max(axis=1).tolist()])
    print("global max abs diff: %.3e" % maxd)
    print("argmax agreement: %d/%d" % (agree, T))
    ref_next = int(np.argmax(ref[-1]))
    print("reference next token: %d ; C next token: %d" % (ref_next, int(np.argmax(got[-1]))))

    ok = maxd <= TOL and agree == T
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
