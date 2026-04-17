# Lab 3 Short Answers

Answer the required questions from `docs/LAB_QUESTIONS.md` here.
Focus on conceptual or methodological reasoning. Put full empirical detail in
`report/main.md`.

## Q1


## Q2
The main difference between Multi-Head Attention (MHA) and Grouped Query Attention (GQA) is that the query and key value are set up. This makes it a trade-off between speed and flexibility. In standard MHA, each attention head has its own set of projections for Q, K, and V, which means that there are the same number of Q, K, and V heads. 

GQA, on the other hand, keeps multiple query heads but cuts down on the number of key/value heads. This means that multiple query heads share the same Key Value representation. The configuration (Model Config) shows this change in architecture by setting attention_type to "gqa" and num_kv_heads to less than num_heads. The main benefit of GQA is that it cuts down on the number of parameters and memory use during inference, especially when using KV-cache storage for inference.

This makes it easier for big models to handle more data and grow. But the downside is that sharing K/V representations could make attention patterns less diverse, which could affect the quality of the model's answers. I think the baseline uses MHA (as it should), but in theory, the configuration and code structure could clearly show how GQA would lower K/V projections and memory usage. So, GQA makes things more efficient, but it also reduces the amount of information it can represent. This makes it a useful optimization for large-scale deployments

## Q3
Sinusoidal positional encoding and Rotary Positional Embedding (RoPE) are very different in how they add positional information to the model. In the baseline implementation, sinusoidal encoding is used. A deterministic position-dependent vector is added directly to token embeddings before they go through the transformer layers (src/components/positional.py). This encoding is computed ahead of time using sine and cosine functions and added in the forward pass. This lets the model use absolute positional information. RoPE, on the other hand, doesn't add positional vectors to embeddings. Instead, it uses position-dependent rotations to change the way queries and keys are represented in the attention mechanism.
This means that the attention computation itself, not the input representation, holds positional information. The model code (language_model.py) makes this distinction clear: embeddings get sinusoidal encoding, while RoPE is meant to be used in attention layers. In this submission, all experiments (tiny, small, pre-norm, post-norm) employ sinusoidal encoding (pos_encoding_type="sinusoidal"), thus precluding any direct empirical comparison with RoPE. But in theory, RoPE should do a better job of capturing relative positional relationships, while sinusoidal encoding gives a simple and stable baseline. So, the main difference is between adding at the embedding level and changing at the attention level. This changes how positional information works with the model.

## Q4

## Q5

## Q6

## Optional Q7

## Optional Q8
