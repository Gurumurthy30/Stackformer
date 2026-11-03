# Models API Reference

This page documents the canonical language and vision architectures implemented in `stackformer.models.language` and `stackformer.models.vision`.

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

::: stackformer.models.language.gpt
    options:
      show_root_heading: true
      members:
        - GPT1
        - GPT2

---

### LLaMA Architectures

::: stackformer.models.language.llama
    options:
      show_root_heading: true
      members:
        - Llama1
        - Llama2

---

### Gemma Architectures

::: stackformer.models.language.gemma
    options:
      show_root_heading: true
      members:
        - Gemma1_2B
        - Gemma1_7B

---

### BERT & RoBERTa Encoders

::: stackformer.models.language.bert
    options:
      show_root_heading: true
      members:
        - BERT

---

::: stackformer.models.language.roberta
    options:
      show_root_heading: true
      members:
        - RoBERTa

---

### Sequence-to-Sequence Transformer

::: stackformer.models.language.transformer
    options:
      show_root_heading: true
      members:
        - Transformer

---

## Vision Models

### Vision Transformer (ViT)

::: stackformer.models.vision.vit
    options:
      show_root_heading: true
      members:
        - PatchEmbedding
        - Block
        - Encoder
        - ViT

---

### SegFormer

::: stackformer.models.vision.segformer
    options:
      show_root_heading: true
      members:
        - OverlapPatchEmbeddings
        - SequenceReduction
        - EfficientAttention
        - MixVisionEncoder
        - DecodeHead
        - SegFormer
