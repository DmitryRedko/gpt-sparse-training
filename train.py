import os
import math
import pickle

import numpy as np
import torch

from model import GPTConfig, GPT
from sparsity_scheduler import SparsityScheduler

from training.model_trainer import Trainer

# -----------------------------------------------------------------------------
# default config values designed to train a GPT‑2 (124 M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False  # if True, script exits right after the first eval
always_save_checkpoint = True  # if True, always save a checkpoint after each eval
init_from = 'scratch'  # 'scratch' | 'resume' | 'gpt2*'
# wandb logging
wandb_log = False
wandb_project = 'owt'
wandb_run_name = 'gpt2'
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8  # simulates larger batch sizes
batch_size = 12  # micro‑batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
bias = False  # use bias inside LayerNorm and Linear layers?
activation_function = "gelu"  # "gelu", "relu", "relu^2"
# AdamW optimizer
learning_rate = 6e-4  # max learning rate
max_iters = 600000  # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
# learning‑rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 2000  # warm‑up steps
lr_decay_iters = 600000  # should be ~= max_iters per Chinchilla
min_lr = 6e-5  # ~= learning_rate/10 per Chinchilla
sparsity_mode = "static"
sparsity_type = "masked-activations-layer" # "orig"
sparsity_ratio: float | dict | list = 0.2

# HotFix потому что  забыли завести соответствующие переменные.
mode: str = 'all'  # "all", "exclude-first-last", or "custom"
custom_slice = None

save_best_model = True
always_save_checkpoint = True

# EARLY‑STOPPING --------------------------------------------------------------
early_stop_mode = True
early_stop_patience = 3  # number of consecutive evals with rising perplexity
# -----------------------------------------------------------------------------


eval_ckpt_name = "best_model.pt"

save_gradients = False
gradient_save_interval = 250

# system
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 'cpu', 'cuda', 'cuda:0', etc.
dtype = (
    'bfloat16'
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else 'float16'
)  # 'float32', 'bfloat16', or 'float16'
compile = True  # use PyTorch 2.0 compile to speed up training
# -----------------------------------------------------------------------------
# Allow command‑line or config‑file overrides via configurator.py
config_keys = [
    k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))
]
exec(open('configurator.py').read())  # may overwrite globals
config = {k: globals()[k] for k in config_keys}  # handy for logging
# -----------------------------------------------------------------------------

# Set up reproducibility and I/O
os.makedirs(out_dir, exist_ok=True)
seed_offset = 0

torch.manual_seed(1337 + seed_offset)

# Enable TF32 where available
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Autocast context and dtype handling
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]


tokens_per_iter = gradient_accumulation_steps * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

iter_num = 0
best_val_loss = 1e9

meta_path = os.path.join('data', dataset, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=bias,
    vocab_size=None,
    dropout=dropout,
    activation_function=activation_function,
)

checkpoint = None

if init_from == 'scratch':
    print("Initializing a new model from scratch")
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT‑2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    if sparsity_mode == "uniform":
        assert isinstance(sparsity_ratio, dict), "For uniform sparsity, sparsity_ratio should be a dict with 'start' and 'end' keys."
        model = GPT(gptconf, sparsity_ratio['start'], sparsity_type, mode, custom_slice)
    elif sparsity_mode == "grid":
        assert isinstance(sparsity_ratio, list), "For grid sparsity, sparsity_ratio should be a list of sparsity levels."
        model = GPT(gptconf, sparsity_ratio[0], sparsity_type, mode, custom_slice)
    elif sparsity_mode == "static":
        assert isinstance(sparsity_ratio, float), "For static sparsity, sparsity_ratio should be a single float value."
        model = GPT(gptconf, sparsity_ratio, sparsity_type, mode, custom_slice)
    else: 
        raise ValueError("Could not classify such sparsity_mode")
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, eval_ckpt_name)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    checkpoint_model_args = checkpoint['model_args']
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
        
    gptconf = GPTConfig(**model_args)
    
    if sparsity_mode == "uniform":
        assert isinstance(sparsity_ratio, dict), "For uniform sparsity, sparsity_ratio should be a dict with 'start' and 'end' keys."
        model = GPT(gptconf, sparsity_ratio['end'], sparsity_type, mode, custom_slice)
    elif sparsity_mode == "grid":
        assert isinstance(sparsity_ratio, list), "For grid sparsity, sparsity_ratio should be a list of sparsity levels."
        model = GPT(gptconf, sparsity_ratio[max(sparsity_ratio)], sparsity_type, mode, custom_slice)
    elif sparsity_mode == "static":
        assert isinstance(sparsity_ratio, float), "For static sparsity, sparsity_ratio should be a single float value."
        model = GPT(gptconf, sparsity_ratio, sparsity_type, mode, custom_slice)
    else: 
        raise ValueError("Could not classify such sparsity_mode")

    
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
else:
    raise ValueError(f"init_from {init_from} not supported")

if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size

model.to(device)

print("======================================")
print("MODEL:")
print(model)
print("======================================")

sparsity_scheduler = None
if sparsity_mode == "uniform":
    assert isinstance(sparsity_ratio, dict), "For uniform sparsity, sparsity_ratio should be a dict with 'start' and 'end' keys."
    sparsity_scheduler = SparsityScheduler(model, mode="uniform", start=sparsity_ratio['start'], end=sparsity_ratio['end'], total_steps=max_iters)
if sparsity_mode == "grid":
    assert isinstance(sparsity_ratio, list), "For grid sparsity, sparsity_ratio should be a list of sparsity levels."
    sparsity_scheduler = SparsityScheduler(model, mode="grid", grid=sparsity_ratio)
if sparsity_mode == "static":
    assert isinstance(sparsity_ratio, float), "For static sparsity, sparsity_ratio should be a single float value."
    sparsity_scheduler = SparsityScheduler(model, mode="static", start=sparsity_ratio, end=sparsity_ratio, total_steps=max_iters)

optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type=device.split(':')[0])
if checkpoint is not None:
    optimizer.load_state_dict(checkpoint['optimizer'])

if compile:
    print("compiling the model... (takes a ~minute)")
    model = torch.compile(model)

def get_lr(it):
    """Cosine LR schedule with warm‑up."""
    if not decay_lr:
        return learning_rate
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

wandb=None

if wandb_log and not eval_only:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)
    wandb.watch(model, log="all", log_freq=100)

train_data = np.memmap(os.path.join('data', dataset, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join('data', dataset, 'val.bin'), dtype=np.uint16, mode='r')
val_data = val_data[:len(val_data) - (len(val_data) % block_size)]

trainer = Trainer(model, optimizer, device=device, dtype=ptdtype)
if not eval_only:
    trainer.train(
        train_data,
        val_data,

        eval_interval=eval_interval,
        steps=max_iters,
        start_step=iter_num,
        batch_size=batch_size * gradient_accumulation_steps,
        mini_batch_size=batch_size,
        grad_clip=grad_clip,
        early_stop_patience=early_stop_patience if early_stop_mode else 0,

        sparsity_scheduler=sparsity_scheduler,
        lr_scheduler=get_lr,

        best_model_dir=out_dir if save_best_model else None,
        checkpoint_dir=out_dir if always_save_checkpoint else None,

        l1_target="weight",

        wandb=wandb,
    ).visualize(os.path.join(out_dir, 'plots'))
else:
    print("Running evaluation only")
    val_loss, ppl_val = trainer.evaluation_step(val_data, batch_size)
    print(f"step {iter_num}: val loss {val_loss:.4f}, val ppl {ppl_val:.4f}")

if wandb: wandb.finish()
