# Serving Subsystem

Status: Planned, not implemented

## Purpose
Provides continuous-batching inference serving primitives, integrating Stackformer's paged KV cache (`stackformer.cache.paged`) with asynchronous request scheduling.

## Planned Contents
- Asynchronous continuous batching scheduler.
- Paged KV cache block allocator and manager.
- Streaming token generation server.

## Checklist
- [ ] Integrate `stackformer.cache.PagedKVCache` with request batching queue.
- [ ] Implement continuous batching iteration loop.
- [ ] Add token streaming output interface.
- [ ] Add latency and throughput benchmark tests.
