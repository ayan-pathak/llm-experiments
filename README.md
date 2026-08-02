# llm-experiments

Training a small character-level GPT on Shakespeare, **CPU-only**, on modest hardware
(Intel i5, 8 GB RAM, no GPU).

## Result

A 2.67M-parameter GPT trained for 3000 iterations in roughly **11 minutes** on 4 CPU threads.

| Metric | Value |
| --- | --- |
| Final validation loss | **1.8762** |
| Final training loss | 1.7134 |
| Starting loss (random) | 4.1993 |
| Parameters | 2.67M |
| Peak RAM | ~410 MB |
| Time per iteration | ~200 ms |

Validation loss was still falling at iteration 3000, so this model is **undertrained** —
more iterations would keep improving it.

### Loss curve

| Step | Train | Val |
| --- | --- | --- |
| 0 | 4.1995 | 4.1993 |
| 500 | 2.2999 | 2.3384 |
| 1000 | 2.0983 | 2.1492 |
| 1500 | 1.9073 | 2.0103 |
| 2000 | 1.8044 | 1.9244 |
| 2500 | 1.7416 | 1.8865 |
| 3000 | 1.7134 | 1.8762 |

The gap between train and val loss widens over time (0.00 → 0.16), the beginning of
overfitting on this small 1.1M-character dataset.

## Sample output

See [sample_output.txt](sample_output.txt). Excerpt:

```
CLOUCESTER:
Now love
To prove conters, was upon off this gove but him.
```

The model learned play formatting, capitalized speaker names followed by a colon, line
breaks, and English-like spelling — but not real words or meaning. That is the expected
outcome at a validation loss near 1.9.

## Configuration

See [configs/train_shakespeare_char_cpu.py](configs/train_shakespeare_char_cpu.py).

Three settings matter most for CPU:

- `gradient_accumulation_steps = 1` — nanoGPT defaults this to **40**, which would make
  every iteration 40x slower.
- `dtype = 'float32'` — nanoGPT defaults to float16 when no GPU is found, which behaves
  badly on CPU.
- `eval_iters = 20` — the default 200 spends more time measuring than training.

## Reproducing

Requires Python 3.12.

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

git clone https://github.com/karpathy/nanoGPT.git
copy configs\train_shakespeare_char_cpu.py nanoGPT\config\
cd nanoGPT
python data\shakespeare_char\prepare.py
python train.py config/train_shakespeare_char_cpu.py
python sample.py --out_dir=out-shakespeare-char-cpu --device=cpu
```

The `--index-url` matters: the default PyTorch package bundles ~2.5 GB of NVIDIA GPU
libraries that are useless without a GPU. The CPU index is a 122 MB download.

nanoGPT itself is not committed here — it is upstream code with its own history.

## Environment

| Package | Version |
| --- | --- |
| Python | 3.12.10 |
| torch | 2.13.0+cpu |
| numpy | 2.5.1 |
| matplotlib | 3.11.1 |

## Credit

Model code from [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) (MIT).
