# Stackformer Data Subsystem — Architecture & Design

## 1. Overview & Objectives

The Stackformer data subsystem is designed to deliver high-throughput, deterministic, and distributed-ready data streams to `Trainer` with zero external binary runtime dependencies.

### Design Principles:
1. **Zero Mandatory Dependencies**: Core data loading uses standard PyTorch (`torch.utils.data`) and Python standard library primitives.
2. **Deterministic Resumption**: Native support for resuming dataloaders from arbitrary mid-epoch step boundaries (`step_in_epoch`) with bit-identical batch sequences.
3. **Hardware-Efficient Packing**: Variable-length tokenized sequences are packed into fixed context windows ($L$) to eliminate padding waste during pretraining.
4. **Clean Distributed Partitioning**: Sharding across data-parallel ranks (`DP`, `DP_SHARD`) handles both map-style indexing and streaming datasets without duplicate batch delivery.

---

## 2. Pipeline Mental Model

```text
1. Data Source (JSONL / Parquet / Memory-Mapped Binaries)
      │
      ▼
2. Transform & Tokenization (Tokenize -> Token IDs + Attention Bounds)
      │
      ▼
3. Packing (First-Fit Decreasing or Concat-Then-Split)
      │
      ▼
4. Collation & Sharding (DataCollator -> TrainerBatch dict)
      │
      ▼
5. Distributed DataLoader (Prefetch + Pin Memory + Device Transfer)
      │
      ▼
6. Trainer forward/backward
```

---

## 3. Core Components

### 3.1 Sources
- **`IndexedJsonlSource`**: Reads `.jsonl` files indexed by byte offset for $O(1)$ random access by index without loading the entire document set into memory.
- **`BinaryMemmapSource`**: Memory-maps pre-tokenized `uint16` / `int32` token arrays directly from disk, enabling instantaneous dataset initialization and shared page caching across processes.

### 3.2 Sequence Packing & Masking
Variable-length sequences are packed into fixed sequence lengths (e.g., 2048 or 4096 tokens) using two strategies:
1. **First-Fit Decreasing (FFD)**: Knapsack-style packing that minimizes document fragmentation while fitting within maximum sequence length.
2. **Concat-then-Split**: Continuous token stream split into chunks of size $L$.
3. **Document Boundary Masking**: Produces 2D attention document masks (or FlexAttention `BlockMask`) preventing cross-document attention leakage without requiring padding tokens.

### 3.3 Mid-Epoch Resumption
- The data loader state tracks `epoch`, `batches_yielded_in_epoch`, and worker RNG states.
- On resumption from checkpoint (`TrainState.step_in_epoch`), the dataloader can fast-forward past already-processed micro-batches or reconstruct the generator permutation deterministically.

---

## 4. API Specification (Planned for Phase 4)

```python
from stackformer.data import (
    IndexedJsonlSource,
    PackedSequenceCollator,
    create_dataloader,
)

# 1. Source
source = IndexedJsonlSource("data/pretrain/*.jsonl", text_key="text")

# 2. Packing Collator
collator = PackedSequenceCollator(
    max_seq_len=2048,
    pad_token_id=0,
    packing_strategy="first_fit",
)

# 3. DataLoader
loader = create_dataloader(
    source,
    batch_size=32,
    collator=collator,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
)
```

