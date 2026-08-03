"""
Logit lens for the trained character-level Shakespeare model.

THE IDEA
--------
A transformer builds its answer gradually. The input text becomes a list of
vectors (one per character), and each of the 6 layers *adds* a correction to
those vectors. The final answer is read off the vector after layer 6, by
normalising it (ln_f) and multiplying by the output head (lm_head).

The logit lens asks: what if we did that read-off EARLY? Take the vector after
layer 2, run it through the same ln_f and lm_head, and see what character the
model would have guessed if it had stopped there. Doing this for every layer
shows you the model making up its mind.

This works here because nanoGPT is a "pre-LN" transformer: each block computes
    x = x + attn(ln_1(x))
    x = x + mlp(ln_2(x))
so x is a running total that every layer writes into, and all layers share the
same vector space. That shared space is what lets one output head read all of
them.

NOTHING IS TRAINED OR MODIFIED. The model is in eval() mode, everything runs
under torch.no_grad(), and no file is written.

Usage:
    python logit_lens.py
    python logit_lens.py --text "ROMEO: But soft" --topk 3
"""
import argparse
import pickle
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent
NANOGPT = REPO / "nanoGPT"
sys.path.insert(0, str(NANOGPT))

from model import GPT, GPTConfig  # noqa: E402  (needs sys.path set first)

DEFAULT_TEXT = "ROMEO: But soft, what light through yonder"


def load_model(out_dir, device="cpu"):
    """Rebuild the model and load the trained weights. Read-only."""
    ckpt_path = NANOGPT / out_dir / "ckpt.pt"
    if not ckpt_path.exists():
        sys.exit(f"No checkpoint at {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = GPT(GPTConfig(**ckpt["model_args"]))

    state_dict = ckpt["model"]
    for k in list(state_dict):  # strip the torch.compile prefix if present
        if k.startswith("_orig_mod."):
            state_dict[k[len("_orig_mod."):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

    # eval() matters a lot here: training used dropout=0.2, which randomly zeroes
    # activations. Leaving it on would make every run give different answers.
    model.eval().to(device)

    meta_path = NANOGPT / "data" / ckpt["config"]["dataset"] / "meta.pkl"
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    best = ckpt.get("best_val_loss")
    best = best.item() if hasattr(best, "item") else best
    return model, meta["stoi"], meta["itos"], ckpt.get("iter_num"), best


def capture_hidden_states(model, idx):
    """Run the model once and grab the residual stream after every layer.

    A "forward hook" is a callback PyTorch runs whenever a module finishes its
    forward pass. We attach one to each of the 6 blocks, so we collect the real
    intermediate values from the real forward pass -- no reimplementation that
    could silently drift from the actual model.
    """
    hidden = []
    handles = [
        block.register_forward_hook(lambda mod, inp, out: hidden.append(out.detach()))
        for block in model.transformer.h
    ]
    try:
        with torch.no_grad():
            model(idx)  # return value ignored; the hooks are what we want
    finally:
        for h in handles:  # always remove hooks, even if the forward pass raises
            h.remove()
    return hidden


def lens(model, hidden_state):
    """Decode one layer's hidden state into character probabilities.

    This is the whole trick, and it is only two operations -- the exact same two
    the model itself applies at the end:
        ln_f     normalise the vector
        lm_head  project 192 numbers onto 65 character scores
    """
    with torch.no_grad():
        logits = model.lm_head(model.transformer.ln_f(hidden_state))
        return F.softmax(logits, dim=-1)


def show(ch):
    """Make whitespace visible in the table."""
    return {" ": "_", "\n": "\\n", "\t": "\\t"}.get(ch, ch)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="out-shakespeare-char-cpu-8k")
    p.add_argument("--text", default=DEFAULT_TEXT)
    p.add_argument("--topk", type=int, default=1,
                   help="how many predictions to show per cell")
    args = p.parse_args()

    model, stoi, itos, step, best = load_model(args.out_dir)
    n_layer = model.config.n_layer

    text = args.text
    unknown = sorted({c for c in text if c not in stoi})
    if unknown:
        sys.exit(f"characters not in the 65-character vocabulary: {unknown}")
    if len(text) > model.config.block_size:
        sys.exit(f"text is {len(text)} chars but block_size is "
                 f"{model.config.block_size}; shorten it")

    idx = torch.tensor([[stoi[c] for c in text]], dtype=torch.long)

    print(f"checkpoint : {args.out_dir} (step {step}, best val loss {best:.4f})")
    print(f"model      : {n_layer} layers, {model.config.n_embd} embd, "
          f"{model.config.vocab_size} vocab")
    print(f"input      : {text!r} ({len(text)} characters)\n")

    hidden = capture_hidden_states(model, idx)
    assert len(hidden) == n_layer, f"expected {n_layer} hidden states, got {len(hidden)}"

    # probs[layer][position] -> distribution over the 65 characters
    probs = [lens(model, h)[0] for h in hidden]

    # Sanity check: layer 6 through the lens must equal the model's own output,
    # because that IS what the model does. If this fails, the lens is wrong.
    with torch.no_grad():
        real_logits, _ = model(idx, targets=torch.zeros_like(idx))
        real_last = real_logits[0, -1].argmax().item()
    lens_last = probs[-1][-1].argmax().item()
    ok = "OK" if real_last == lens_last else "MISMATCH"
    print(f"self-check : final-layer lens vs model's own output -> {ok} "
          f"(both predict {show(itos[lens_last])!r})\n")

    # ---- build the table ----
    top = [[probs[l][i].argmax().item() for i in range(len(text))]
           for l in range(n_layer)]

    col_w = 9
    header = f"{'pos':>3} {'char':>5} {'next':>5}  " + "".join(
        f"{'layer ' + str(l + 1):>{col_w}}" for l in range(n_layer)
    ) + f"{'first=':>8}{'stays=':>8}"
    print(header)
    print("-" * len(header))

    converged_at, settled_at = [], []
    for i in range(len(text)):
        final = top[-1][i]
        # earliest layer whose top guess already equals the final layer's guess
        first = next((l for l in range(n_layer) if top[l][i] == final), n_layer - 1)
        # earliest layer that matches AND never changes its mind afterwards.
        # "first" can fire by coincidence -- a layer can agree, drift away, and
        # come back -- so this stricter number is the more honest one.
        settled = next(l for l in range(n_layer)
                       if all(top[j][i] == final for j in range(l, n_layer)))
        converged_at.append(first + 1)
        settled_at.append(settled + 1)

        actual = show(text[i + 1]) if i + 1 < len(text) else "?"
        row = f"{i:>3} {show(text[i]):>5} {actual:>5}  "
        for l in range(n_layer):
            c = show(itos[top[l][i]])
            pct = probs[l][i, top[l][i]].item() * 100
            cell = f"{c}{pct:>4.0f}%"
            # mark the earliest layer that already agrees with the final answer
            row += f"{'[' + cell + ']' if l == first else ' ' + cell + ' ':>{col_w}}"
        row += f"{first + 1:>8}{settled + 1:>8}"
        print(row)

    print("-" * len(header))
    print("[ ] = earliest layer whose top prediction already matches layer 6's.")
    print("'_' = space.  Percentages are the model's confidence in that character.\n")

    avg = sum(converged_at) / len(converged_at)
    avg_settled = sum(settled_at) / len(settled_at)
    hits = sum(1 for i in range(len(text) - 1)
               if itos[top[-1][i]] == text[i + 1])
    flaky = sum(1 for f, s in zip(converged_at, settled_at) if f != s)
    print(f"average first-agreement layer : {avg:.2f} of {n_layer}")
    print(f"average settle-for-good layer : {avg_settled:.2f} of {n_layer}")
    print(f"positions that changed mind   : {flaky}/{len(text)} agreed early, "
          f"then drifted away again")
    print(f"final layer correct next-char : {hits}/{len(text) - 1} "
          f"({hits / (len(text) - 1) * 100:.0f}%)")
    dist = {l: settled_at.count(l) for l in sorted(set(settled_at))}
    print(f"settle distribution           : "
          + ", ".join(f"layer {l}: {n}" for l, n in dist.items()))


if __name__ == "__main__":
    main()
