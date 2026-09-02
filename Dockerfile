# =============================================================================
# Stackformer — Container Image
# =============================================================================
# This Dockerfile packages the stackformer library into a minimal, production-
# ready image.  It is NOT a training-ready image (no GPU drivers / CUDA) — it
# is designed to expose stackformer as a Python environment that downstream
# users can import or script against on CPU.
#
# Usage examples
# --------------
# Run a one-off Python script:
#   docker run --rm ghcr.io/stackformer-labs/stackformer:latest \
#       python -c "import stackformer; print(stackformer.__version__)"
#
# Interactive shell:
#   docker run --rm -it ghcr.io/stackformer-labs/stackformer:latest python
#
# Mount local code and run a script:
#   docker run --rm -v $(pwd):/workspace ghcr.io/stackformer-labs/stackformer:latest \
#       python /workspace/my_script.py
# =============================================================================

# ── Build arg: version is injected by the CD workflow (optional, cosmetic) ──
ARG STACKFORMER_VERSION=dev

# ── Base: slim Python 3.11 for a small final image footprint ────────────────
FROM python:3.11-slim AS base

LABEL org.opencontainers.image.title="stackformer"
LABEL org.opencontainers.image.description="Modular transformer blocks built in PyTorch"
LABEL org.opencontainers.image.source="https://github.com/stackformer-labs/Stackformer"
LABEL org.opencontainers.image.licenses="MIT"

ARG STACKFORMER_VERSION
ENV STACKFORMER_IMAGE_VERSION=${STACKFORMER_VERSION}

# ── System dependencies ───────────────────────────────────────────────────────
# git is needed only if you later want to pip-install from VCS inside the
# container; remove if not required to keep the image smaller.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
    && rm -rf /var/lib/apt/lists/*

# ── Create a non-root user ────────────────────────────────────────────────────
RUN useradd --create-home --shell /bin/bash stackformer
WORKDIR /home/stackformer

# ── Install PyTorch (CPU-only) and then stackformer ──────────────────────────
# We install PyTorch's CPU wheel explicitly before stackformer so that pip
# resolves against the CPU variant rather than downloading the full CUDA
# wheel (~2 GB) — that would make the image unnecessarily large for a library
# image.  CUDA support can be added in a derived image.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        "torch>=2.3,<3.0" \
    && pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        stackformer==${STACKFORMER_VERSION}

# ── Switch to non-root user ───────────────────────────────────────────────────
USER stackformer

# ── Default entrypoint: interactive Python ────────────────────────────────────
CMD ["python"]
