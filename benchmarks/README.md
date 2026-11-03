# Stackformer Benchmarks

Status: Planned, not implemented

## Purpose
Provides reproducible throughput, memory, and scaling benchmarks across models, batch sizes, sequence lengths, and distributed training configurations.

## Planned Contents
- Throughput benchmarking scripts (tokens/sec, images/sec).
- Peak memory profiling scripts (CUDA memory tracking, activation memory).
- Distributed scaling benchmarks (weak and strong scaling curves for DDP and FSDP2).

## Checklist
- [ ] Implement single-GPU throughput benchmark harness.
- [ ] Implement multi-GPU DDP / FSDP2 throughput benchmark.
- [ ] Add peak memory tracking via `torch.cuda.max_memory_allocated()`.
- [ ] Document hardware baselines and reproduction commands.

