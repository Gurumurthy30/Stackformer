# Configuration Guide

StackFormer provides three structured dataclasses under `stackformer.config` to manage hyperparameters across model definitions, training loops, and sequence decoding:

1. **`ModelConfig`**: Architectural hyperparameters (embedding dimensions, layer counts, heads, context lengths).
2. **`TrainingConfig`**: Optimization parameters (epochs, learning rates, gradient accumulation, schedulers).
3. **`GenerationConfig`**: Decoding parameters (max tokens, temperature, top-k, top-p, EOS tokens).

All three classes use Python `slots=True` for reduced memory overhead and attribute access speed.

---

## 1. `ModelConfig`

`ModelConfig` serves as a structured container for storing, serializing, and passing model architecture hyperparameters.

### Field Definitions

| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `vocab_size` | `int` | *(Required)* | Total vocabulary size (number of tokens). |
| `embed_dim` | `int` | *(Required)* | Hidden state embedding dimension $C$. |
| `num_layers` | `int` | *(Required)* | Number of stacked Transformer blocks $N$. |
| `num_heads` | `int` | *(Required)* | Number of query attention heads $H$. |
| `seq_len` | `int` | *(Required)* | Maximum context sequence length $T$. |
| `hidden_dim` | `int` | *(Required)* | Feed-forward inner hidden layer dimension $M$. |
| `dropout` | `float` | `0.0` | Dropout probability across attention and FFN layers. |

### Usage Example

```python
from dataclasses import asdict
from stackformer.config import ModelConfig
from stackformer.models import GPT2

# 1. Define model configuration
cfg = ModelConfig(
    vocab_size=50257,
    embed_dim=768,
    num_layers=12,
    num_heads=12,
    seq_len=1024,
    hidden_dim=3072,
    dropout=0.1,
)

# 2. Unpack into model constructor
model = GPT2(**asdict(cfg))
print(f"Instantiated GPT-2 with {cfg.num_layers} layers and {cfg.embed_dim} embed dim.")
```

---

## 2. `TrainingConfig`

`TrainingConfig` centralizes all hyperparameters governing the training execution loop.

### Field Definitions

| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `max_epochs` | `int` | `1` | Maximum number of training epochs to run. |
| `max_train_steps` | `int \| None` | `None` | Optional hard limit on total training steps. |
| `max_eval_steps` | `int \| None` | `None` | Optional limit on evaluation batches per validation phase. |
| `eval_every_n_epochs` | `int` | `1` | Validation frequency in epochs. |
| `save_every_n_epochs` | `int` | `1` | Checkpoint save frequency in epochs. |
| `grad_accumulation_step` | `int` | `1` | Number of batches to accumulate before optimizer step. |
| `max_grad_norm` | `float \| None` | `None` | Threshold for gradient norm clipping (e.g. `1.0`). |
| `lr` | `float` | `3e-4` | Peak learning rate. |
| `weight_decay` | `float` | `0.01` | Weight decay coefficient for decoupled AdamW. |
| `optimizer_name` | `str` | `"adamw"` | Name of optimizer (`"adamw"`, `"adam"`, `"sgd"`, `"adafactor"`). |
| `scheduler_name` | `str` | `"none"` | LR schedule policy (`"cosine"`, `"linear"`, `"step"`, `"none"`). |
| `warmup_steps` | `int` | `0` | Number of linear learning rate warmup steps. |
| `total_steps` | `int \| None` | `None` | Total steps used to compute cosine/linear decay schedules. |

### Usage Example

```python
from stackformer.config import TrainingConfig
from stackformer.engine import Trainer

train_cfg = TrainingConfig(
    max_epochs=5,
    lr=1e-4,
    grad_accumulation_step=4,
    max_grad_norm=1.0,
    optimizer_name="adamw",
    scheduler_name="cosine",
    warmup_steps=100,
    total_steps=1000,
)

# Pass directly to Trainer via training_config
trainer = Trainer(
    model=model,
    train_dataloader=dataloader,
    training_config=train_cfg,
)
```

---

## 3. `GenerationConfig`

`GenerationConfig` controls sequence decoding behavior when invoking `text_generate()`.

### Field Definitions

| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `max_context_len` | `int` | `128` | Maximum sequence context window length. |
| `max_new_tokens` | `int` | `50` | Maximum number of new tokens to generate. |
| `temperature` | `float` | `1.0` | Logit scaling factor ($>0$). Values $<1$ make output more focused; $>1$ more random. |
| `top_k` | `int \| None` | `None` | Restrict sampling to the top $k$ highest-probability tokens. |
| `top_p` | `float` | `1.0` | Nucleus sampling: restrict sampling to cumulative top $p$ probability mass. |
| `eos_token_id` | `int \| None` | `None` | Optional token ID that halts generation when sampled. |

### Usage Example

```python
from stackformer import text_generate
from stackformer.config import GenerationConfig

# Define creative sampling profile
gen_cfg = GenerationConfig(
    max_context_len=256,
    max_new_tokens=40,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    eos_token_id=50256,  # GPT-2 <|endoftext|>
)

prompt = torch.randint(0, 10000, (1, 8))
output = text_generate(
    model=model,
    prompt_ids=prompt,
    generation_config=gen_cfg,
)
```

