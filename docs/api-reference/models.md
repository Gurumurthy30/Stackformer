# Models API Reference

This page documents the canonical language and vision architectures implemented in `stackformer.models` and `stackformer.vision`.

---

## Base Model Lifecycle

::: stackformer.models.base
    options:
      show_root_heading: true
      members:
        - BaseModel
        - resolve_model_config

---

## Language Models

### GPT Architectures

::: stackformer.models.gpt
    options:
      show_root_heading: true
      members:
        - GPT1
        - GPT2

---

### LLaMA Architectures

::: stackformer.models.llama
    options:
      show_root_heading: true
      members:
        - Llama1
        - Llama2

---

### Gemma Architectures

::: stackformer.models.gemma
    options:
      show_root_heading: true
      members:
        - Gemma1_2B
        - Gemma1_7B

---

### BERT & RoBERTa Encoders

::: stackformer.models.bert
    options:
      show_root_heading: true
      members:
        - BERT

::: stackformer.models.roberta
    options:
      show_root_heading: true
      members:
        - RoBERTa

---

### Sequence-to-Sequence Transformer

::: stackformer.models.transformer
    options:
      show_root_heading: true
      members:
        - Transformer

---

## Vision Models

### Vision Transformer (ViT)

::: stackformer.vision.vit
    options:
      show_root_heading: true
      members:
        - PatchEmbedding
        - Block
        - Encoder
        - ViT

---

### SegFormer

::: stackformer.vision.segformer
    options:
      show_root_heading: true
      members:
        - OverlapPatchEmbeddings
        - SequenceReduction
        - EfficientAttention
        - MixVisionEncoder
        - DecodeHead
        - SegFormer

