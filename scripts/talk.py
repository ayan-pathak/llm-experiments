"""
Prompt the trained Shakespeare model and watch it continue your text.

This is NOT a chatbot. The model was trained to do exactly one thing: given some
characters, guess the next character. So it *continues* whatever you type rather
than answering it. Type a line of dialogue or a speaker name and it carries on.

    python scripts/talk.py
    python scripts/talk.py --temperature 0.6 --tokens 300

Type 'quit' to exit.
"""
import argparse
import pickle
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
NANOGPT = REPO / "nanoGPT"
sys.path.insert(0, str(NANOGPT))

from model import GPT, GPTConfig  # noqa: E402  (needs sys.path set first)


def load(out_dir, device):
    ckpt_path = NANOGPT / out_dir / "ckpt.pt"
    if not ckpt_path.exists():
        sys.exit(f"No checkpoint at {ckpt_path}\nTrain a model first.")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = GPT(GPTConfig(**ckpt["model_args"]))

    state_dict = ckpt["model"]
    for k in list(state_dict):  # strip the torch.compile prefix if present
        if k.startswith("_orig_mod."):
            state_dict[k[len("_orig_mod."):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval().to(device)

    meta_path = NANOGPT / "data" / ckpt["config"]["dataset"] / "meta.pkl"
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    step = ckpt.get("iter_num", "?")
    val = ckpt.get("best_val_loss")
    val = f"{val:.4f}" if isinstance(val, (int, float)) else (
        f"{val.item():.4f}" if hasattr(val, "item") else "?")
    return model, meta["stoi"], meta["itos"], step, val


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="out-shakespeare-char-cpu-8k")
    p.add_argument("--temperature", type=float, default=0.8,
                   help="lower = safer and more repetitive, higher = wilder")
    p.add_argument("--tokens", type=int, default=400, help="characters to generate")
    p.add_argument("--top_k", type=int, default=200)
    args = p.parse_args()

    device = "cpu"
    torch.manual_seed(1337)
    model, stoi, itos, step, val = load(args.out_dir, device)

    print(f"\nLoaded {args.out_dir} (step {step}, best val loss {val})")
    print(f"temperature {args.temperature} | generating {args.tokens} characters")
    print("\nThis model CONTINUES your text, it does not answer questions.")
    print("Try a speaker name like 'ROMEO:' or the start of a line.")
    print("Type 'quit' to exit.\n")

    while True:
        try:
            prompt = input("you> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if prompt.strip().lower() in {"quit", "exit"}:
            break
        if not prompt:
            prompt = "\n"

        # The model only knows 65 characters. Anything else has no encoding.
        unknown = sorted({c for c in prompt if c not in stoi})
        if unknown:
            shown = " ".join(repr(c) for c in unknown)
            print(f"  (dropped characters the model has never seen: {shown})")
            prompt = "".join(c for c in prompt if c in stoi)
            if not prompt:
                print("  nothing left to work with -- try plain letters.\n")
                continue

        ids = torch.tensor([[stoi[c] for c in prompt]], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model.generate(ids, args.tokens,
                                 temperature=args.temperature, top_k=args.top_k)
        text = "".join(itos[i] for i in out[0].tolist())
        print(f"\n{text}\n{'-' * 60}\n")


if __name__ == "__main__":
    main()
