# Longer version of train_shakespeare_char_cpu.py: 8000 iterations instead of 3000.
# Same model and hardware assumptions (i5 CPU, 8GB RAM, no GPU). Expect ~30 minutes.

out_dir = 'out-shakespeare-char-cpu-8k'

eval_interval = 250
eval_iters = 20
log_interval = 100

# Only save when validation loss improves, so the final checkpoint is the BEST
# model seen, not the last one. This matters more here: with 8000 iterations on a
# 1.1M-character dataset the model will start overfitting, and this setting means
# we automatically keep the checkpoint from before that happened.
always_save_checkpoint = False

wandb_log = False

dataset = 'shakespeare_char'

gradient_accumulation_steps = 1
batch_size = 12
block_size = 64

n_layer = 6
n_head = 6
n_embd = 192
dropout = 0.2

learning_rate = 1e-3
max_iters = 8000
lr_decay_iters = 8000   # keep equal to max_iters
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100

device = 'cpu'
compile = False
dtype = 'float32'
