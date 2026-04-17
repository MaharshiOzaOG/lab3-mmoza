# Lab 3 Short Answers

Answer the required questions from `docs/LAB_QUESTIONS.md` here.
Focus on conceptual or methodological reasoning. Put full empirical detail in
`report/main.md`.

## Q1
In a decoder-only transformer, we need to apply a causal mask to satisfy the autoregressive condition that each position only attends to its past positions and not to its future. Specifically, this mask should be applied to the attention logits, not after the softmax. In the case of scaled dot-product attention, logits are calculated as `QK^T/(d k )` and converted to probabilities via a softmax. If masking is applied before the softmax (as implemented in ScaledDotProductAttention.forward where we use scores.masked_fill(mask == 0, -inf)) then for the masked positions we obtain − ∞, which becomes 0 when the softmax is applied and sums of the remaining probabilities are still 1 after renormalization. In other words, we get a valid probability distribution over attentions. If masking is applied after the softmax operation, then the sum of probabilities would not be 1 (or the scaled probability distribution). As in the baseline attention module (src/components/attention.py) and the causal attention module, we ensure masking is performed before the softmax operation.

## Q2
The main difference between Multi-Head Attention (MHA) and Grouped Query Attention (GQA) is that the query and key value are set up. This makes it a trade-off between speed and flexibility. In standard MHA, each attention head has its own set of projections for Q, K, and V, which means that there are the same number of Q, K, and V heads. 

GQA, on the other hand, keeps multiple query heads but cuts down on the number of key/value heads. This means that multiple query heads share the same Key Value representation. The configuration (Model Config) shows this change in architecture by setting attention_type to "gqa" and num_kv_heads to less than num_heads. The main benefit of GQA is that it cuts down on the number of parameters and memory use during inference, especially when using KV-cache storage for inference.

This makes it easier for big models to handle more data and grow. But the downside is that sharing K/V representations could make attention patterns less diverse, which could affect the quality of the model's answers. I think the baseline uses MHA (as it should), but in theory, the configuration and code structure could clearly show how GQA would lower K/V projections and memory usage. So, GQA makes things more efficient, but it also reduces the amount of information it can represent. This makes it a useful optimization for large-scale deployments

## Q3
Sinusoidal positional encoding and Rotary Positional Embedding (RoPE) are very different in how they add positional information to the model. In the baseline implementation, sinusoidal encoding is used. A deterministic position-dependent vector is added directly to token embeddings before they go through the transformer layers (src/components/positional.py). This encoding is computed ahead of time using sine and cosine functions and added in the forward pass. This lets the model use absolute positional information. RoPE, on the other hand, doesn't add positional vectors to embeddings. Instead, it uses position-dependent rotations to change the way queries and keys are represented in the attention mechanism.
This means that the attention computation itself, not the input representation, holds positional information. The model code (language_model.py) makes this distinction clear: embeddings get sinusoidal encoding, while RoPE is meant to be used in attention layers. In this submission, all experiments (tiny, small, pre-norm, post-norm) employ sinusoidal encoding (pos_encoding_type="sinusoidal"), thus precluding any direct empirical comparison with RoPE. But in theory, RoPE should do a better job of capturing relative positional relationships, while sinusoidal encoding gives a simple and stable baseline. So, the main difference is between adding at the embedding level and changing at the attention level. This changes how positional information works with the model.

## Q4
Token-level perplexity is not a good way to compare models that were trained with different tokenizers because it depends on how the text is broken up into tokens. Perplexity tells you how uncertain each token is on average, but different tokenizers make different token sequences for the same text. A tokenizer that breaks text into many small subword units makes the sequences longer and the number of prediction steps higher. On the other hand, a tokenizer that uses bigger tokens makes the sequences shorter. This means that the model's prediction difficulty and entropy per token change, which makes it impossible to compare perplexity values across different tokenization schemes. The lab rules make it clear that perplexity should only be compared when tokenization is set.

In my submission, a consistent tokenizer (english_bytebpe_8k.json) is used in all experiments. This ensures fair comparisons between models. If different tokenizers were used, better comparisons would include qualitative generation quality, tokens-per-sequence efficiency, or evaluations under a fixed context length. We could also look at throughput or memory efficiency because tokenization impacts sequence length and computing costs. Also my tokenizer has a Compression Ratio of 0.80. If I change to ratio of 0.40, the perplexity would drop significantly just because the units got smaller, not because the model got smarter. That is why we only use Validation Loss as a primary metric when the tokenization is held fixed.

## Q5
Shifted input and target sequences sit at the core of autoregressive LM's. The basic idea is tht the model tries to learns to guess the next token by looking what came before. 

For Example 
Input (What the robot sees): The, cat, sat, on, the
Target (The correct answers): cat, sat, on, the, mat


Each token in the input points at the next token in the target, lining things up so the model has to predict what comes next. This is essential if we didn’t shift things, the model could just learn to copy tokens, not actually predict anything, and we’d end up with a useless, trivial identity mapping. 
for example: 
Input: The, cat, sat
Target: The, cat, sat

There will be no real learning asthe model wouldn’t pick up on any patterns or dependencies in the data. We use cross-entropy loss to check how well the model’s guesses match up with these shifted targets. So, shifting isn’t just a random implementation choice it’s actually a direct way to force the model into its real job which is predicting what’s next in a sequence. Skip this step, and you break the training process entirely.

## Q6
One cherry-picked generation example doesn’t prove much about how well a model actually works. Generated text can change wildly depending on the prompt or even some random chance. If you only show one sample, you’re not getting the full picture. Real evidence means checking both hard numbers and a whole set of actual outputs. In this submission, there’s a solid approach: models get evaluated with metrics like validation loss, perplexity, accuracy, and top-k accuracy across all epochs. 

For instance, in Experiment 1, the small model beats the tiny model on both validation loss and perplexity by the third epoch, clearly showing it predicts better. Different generation methods like greedy, top-k, temperature help round out the picture, so you’re not just cherry-picking one kind of output. 

In Experiment 2, post-norm scores slightly better, but the actual generated samples are all over the place, which shows why you can’t just trust the metrics without seeing some examples too. For really solid claims, you’d want outputs from multiple prompts and several samples per prompt, all compared under controlled conditions. In short, strong evidence means you look at stats, compare under fair settings, and check plenty of real outputs and not just a unique result.

## Optional Q7

## Optional Q8
