# Small character-level Shakespeare GPT tuned for an i5 CPU with 8GB RAM, no GPU.
# Target: finish 3000 iterations in well under 30 minutes.

out_dir = 'out-shakespeare-char-cpu'

eval_interval = 250   # check validation loss every 250 iterations
eval_iters = 20       # only 20 batches per check -- 200 (the default) is far too slow on CPU
log_interval = 50     # print progress every 50 iterations

# Only write a checkpoint when validation loss actually improves.
always_save_checkpoint = False

wandb_log = False

dataset = 'shakespeare_char'

# CRITICAL: nanoGPT defaults this to 40, which would make every iteration 40x slower.
gradient_accumulation_steps = 1

batch_size = 12
block_size = 64       # model sees 64 previous characters

# model size
n_layer = 6
n_head = 6
n_embd = 192
dropout = 0.2

# optimizer
learning_rate = 1e-3
max_iters = 3000
lr_decay_iters = 3000
min_lr = 1e-4
beta2 = 0.99          # small number of tokens per step, so raise this
warmup_iters = 100

# CPU-specific
device = 'cpu'
compile = False       # torch.compile gives no benefit here and is unreliable on Windows CPU
dtype = 'float32'     # float16 is the no-GPU default and behaves badly on CPU
