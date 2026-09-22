# LLM Pretraining — A Study Guide

> Pretraining is the phase where a randomly initialized transformer is trained on
> trillions of tokens with a single objective: predict the next token. Everything
> else — instruction tuning, RLHF, tool use — is a thin layer on top of what
> happens here.

**Who this is for:** you can write PyTorch and remember basic linear algebra and
calculus. You do not need prior transformer experience — Part 2 builds it.

**How to use it.** The document has two tracks:

- **The process track — [Part 16](#16-the-end-to-end-process--notebook-blueprint).**
  A linear, runnable walkthrough of the whole pipeline in the order you actually
  execute it, from raw text to a trained, evaluated base model. Every stage is a
  notebook cell. **This is the spine — start here.**
- **The reference track — Parts 1–15.** Topic-by-topic depth. Each process stage
  links back to the part that explains *why*. Read these when a stage raises a
  question, or read them straight through first if you prefer theory before
  keyboard.

Parts 1–15 also carry *Checkpoints* — small exercises that are the actual
learning; the prose is scaffolding.

---

## Table of contents

1. [The mental model](#1-the-mental-model)
2. [Prerequisites](#2-prerequisites)
3. [Data](#3-data)
4. [Tokenization](#4-tokenization)
5. [Architecture](#5-architecture)
6. [The objective and the loss](#6-the-objective-and-the-loss)
7. [Optimization](#7-optimization)
8. [Scaling laws](#8-scaling-laws)
9. [Systems and parallelism](#9-systems-and-parallelism)
10. [Stability: when training breaks](#10-stability-when-training-breaks)
11. [Evaluation](#11-evaluation)
12. [The hands-on ladder](#12-the-hands-on-ladder)
13. [After pretraining](#13-after-pretraining)
14. [Reading list](#14-reading-list)
15. [Open-source repositories](#15-open-source-repositories)
16. [**The end-to-end process — notebook blueprint**](#16-the-end-to-end-process--notebook-blueprint)
17. [Glossary](#17-glossary)

---

## 1. The mental model

Hold these five facts in your head; most of the field is elaboration on them.

**1. The objective is trivial.** Given tokens `x₁…xₜ`, predict `xₜ₊₁`. Loss is
cross-entropy, averaged over every position in every sequence. That's it. There
is no auxiliary trick doing the heavy lifting.

**2. Compute is the currency.** A useful approximation for a dense transformer:

```
C ≈ 6 · N · D        FLOPs
```

where `N` = non-embedding parameters, `D` = training tokens. The 6 comes from
2 FLOPs per multiply-accumulate × 3 passes (forward, backward w.r.t. inputs,
backward w.r.t. weights). Every architectural or data decision is really a
question of *how to spend a fixed C*.

**3. Loss is predictable.** Loss vs. compute follows a power law across ~6 orders
of magnitude. A *sweep* of small models, trained with one consistent recipe, lets
you forecast the loss of a run far larger than any of them. The forecast is
not a guarantee: the further you extrapolate, the more a change in data, recipe or
batch regime can bend the curve. Still, this is why pretraining is an
engineering discipline and not alchemy — you de-risk at small scale.

**4. Data quality moves the curve, not just the point.** Better data doesn't only
give a lower loss at the same compute; it changes the slope of what you get per
FLOP on downstream tasks. The last five years of open-model progress is more a
data story than an architecture story.

**5. At scale, the bottleneck is systems.** A 70B-parameter run is a distributed
systems problem — memory hierarchies, collective communication, fault tolerance
over thousands of GPUs for months — wearing a machine learning costume.

### The pipeline, end to end

```
 raw web dumps (petabytes)
        │  extract text, language ID
        ▼
 quality filtering  ──► heuristics + learned classifiers
        │
        ▼
 deduplication      ──► MinHash LSH, exact substring
        │
        ▼
 decontamination    ──► remove eval-set overlap
        │
        ▼
 mixing             ──► web / code / math / books, with weights
        │
        ▼
 tokenization       ──► BPE → uint16/uint32 token shards
        │
        ▼
 ┌──────────────────────────────────────┐
 │  TRAINING LOOP                       │
 │  sample batch → forward → loss       │
 │  → backward → all-reduce → optimizer │
 │  → checkpoint  (repeat ~10⁵–10⁶ ×)   │
 └──────────────────────────────────────┘
        │
        ▼
 anneal / midtrain on high-quality data
        │
        ▼
 base model ──► evals ──► post-training
```

### Checkpoint 1

Without looking: estimate the training FLOPs for a 7B model on 2T tokens. Then
work out roughly how many H100-days that is at 40% MFU. (Peak bf16 on an H100 is
~990 TFLOP/s dense on the SXM variant — the 1,979 on spec sheets assumes 2:4
sparsity — and ~756 TFLOP/s dense on the PCIe variant. Use whichever card you are
pricing.) You should land in the low thousands of GPU-days.

---

## 2. Prerequisites

Don't over-invest here. You need working fluency, not mastery.

| Area | What you actually need | Skip |
|---|---|---|
| Linear algebra | Matmul shapes, why `(B,T,C) @ (C,C)` works, broadcasting | Eigendecompositions, SVD proofs |
| Calculus | Chain rule, what a gradient is | Manual backprop derivations (do it once, then use autograd) |
| Probability | Cross-entropy, KL, softmax as a distribution, perplexity | Measure theory |
| PyTorch | `nn.Module`, autograd, `DataLoader`, `.to(device)`, mixed precision | Custom C++ extensions |
| Systems | What a GPU SM is, HBM vs SRAM bandwidth, what an all-reduce does | CUDA kernel authoring (until Part 9) |

**The one derivation worth doing by hand:** cross-entropy loss of a softmax
output. The gradient w.r.t. the logits is `softmax(z) − y_onehot`. It falls out
in three lines and it explains why training is numerically well-behaved.

**Fastest path if rusty:** Karpathy's *Neural Networks: Zero to Hero* videos 1–4
(micrograd → makemore → attention). ~8 hours, and you write every line.

### Checkpoint 2

Implement `micrograd` from scratch — a scalar autograd engine in <150 lines.
Verify gradients against PyTorch on a two-layer MLP. If you can do this, you will
never again be confused about what `.backward()` does.

---

## 3. Data

This is the part that is least glamorous and most determines the result. Budget
your learning time accordingly: if architecture gets 20% of your attention, data
should get 40%.

### 3.1 Where tokens come from

| Source | Rough scale | Notes |
|---|---|---|
| Common Crawl | ~100B pages, petabytes of WARC | The base of nearly every open corpus. ~90%+ gets filtered away. |
| Curated web | FineWeb (15T), DCLM-Baseline (4T), RefinedWeb (5T) | Pre-filtered CC derivatives. **Start here** — don't re-do CC processing. |
| Code | The Stack v2, GitHub | Improves reasoning even on non-code evals. |
| Math | OpenWebMath, proof corpora, arXiv | Small volume, outsized effect. |
| Books / papers | arXiv, PubMed, public-domain books | Long-form coherence, rare vocabulary. |
| Reference | Wikipedia, StackExchange | High quality, tiny; usually upweighted. |
| Synthetic | Model-generated rephrasing, textbooks | Increasingly used in midtraining. Contamination risk. |

### 3.2 Filtering

Two families, used together:

**Heuristic filters** — cheap, interpretable, run first. The canonical sets are
the C4 rules and the Gopher quality rules:

- drop documents outside a length band (e.g. 50–100k words)
- mean word length outside ~3–10 characters
- symbol-to-word ratio (`#`, `…`) above ~0.1
- too few lines ending in terminal punctuation (a C4-style line rule, not a
  Gopher one; the threshold is a tuning knob — Stage 3 uses 60%)
- stopword count below a floor (catches keyword spam)
- bullet-heavy or ellipsis-heavy documents
- language ID confidence below threshold (fastText `lid.176`)

**Learned classifiers** — a fastText or small-transformer classifier trained to
distinguish "reference-like" text (Wikipedia, well-upvoted forum posts,
instruction data) from raw crawl, then used to keep the top *k*% of documents.
DCLM showed this single choice was worth more than most architecture tuning.

> **Failure mode to internalize:** aggressive quality filtering collapses
> diversity. A classifier trained on Wikipedia-like text will quietly delete
> dialect, informal registers, and niche domains. Check what your filter *removes*,
> not just what it keeps — sample 100 rejected documents and read them.

### 3.3 Deduplication

Duplicated data causes memorization, wasted compute, and degraded loss curves.
Two levels:

- **Fuzzy, document-level:** MinHash + LSH. Shingle each doc into n-grams
  (n≈5–13), hash, keep a signature of ~100–256 permutations, band the signature
  into buckets, treat colliding docs as near-duplicates at a Jaccard threshold
  (~0.7–0.8). Scales to trillions of tokens with MapReduce-style sharding.
- **Exact, substring-level:** suffix-array based removal of repeated spans over
  ~50 tokens. Catches boilerplate that survives doc-level dedup.

Also decide *scope*: dedup within a crawl dump, or globally across all dumps?
FineWeb found per-dump dedup outperformed global dedup — over-deduplication
strips away genuinely useful repeated content (e.g. widely-mirrored reference
text). This is a real, counterintuitive result worth remembering.

### 3.4 Decontamination

Remove training documents that overlap your evaluation sets (n-gram overlap,
typically 13-gram matching against every eval instance). Do this *before* you
report numbers, and report the method. Contamination is the single most common
reason a published eval score is meaningless.

### 3.5 Mixing

You have domain buckets and a token budget. Weights matter:

- Upsample high-quality, low-volume sources (Wikipedia, math) — but epoching the
  same tokens >~4 times shows clear diminishing returns and eventually hurts.
- Code fraction of ~10–20% is common even for general-purpose models.
- Domain weights can be tuned by proxy: train many small models on candidate
  mixes, fit the loss, extrapolate (this is what DoReMi and related methods do).
- The mix usually *changes over training*: more web early, more high-quality and
  domain-specific data in the final anneal phase.

### 3.6 Practical shape of the data pipeline

Tokenized data is stored as flat binary shards of `uint16` (vocab < 65536) or
`uint32`, concatenated with a document separator token, then read as a
memory-mapped array. A batch is `B` random offsets into that array, each of
length `T+1`; inputs are `[:-1]`, targets are `[1:]`.

```python
import numpy as np, torch

data = np.memmap("train.bin", dtype=np.uint16, mode="r")

def get_batch(B, T, device):
    ix = torch.randint(len(data) - T - 1, (B,))
    x = torch.stack([torch.from_numpy(data[i:i+T].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+T].astype(np.int64)) for i in ix])
    return x.pin_memory().to(device, non_blocking=True), \
           y.pin_memory().to(device, non_blocking=True)
```

Note what this does *not* do: it doesn't respect document boundaries. Sequences
cross documents. Most large runs accept this (with a separator token and
sometimes an attention mask that blocks cross-document attention — "document
masking", which measurably helps).

### Checkpoint 3

Take 10k documents from FineWeb. Implement: (a) three Gopher heuristic filters,
(b) MinHash dedup at threshold 0.8. Report how many documents each stage removes,
then **read 20 of the removed documents**. Write one paragraph on what your
filters are biased against.

---

## 4. Tokenization

An underrated source of model pathologies.

**Byte-Pair Encoding (BPE)** is the standard. Start from bytes; repeatedly merge
the most frequent adjacent pair; stop at a target vocabulary size. Byte-level BPE
(GPT-2 onward) guarantees no out-of-vocabulary input.

Key decisions:

| Decision | Typical | Why it matters |
|---|---|---|
| Vocab size | 32k–256k | Larger vocab → fewer tokens per document (cheaper training and inference) but a bigger embedding/output matrix and rarer per-token updates. Modern models trend larger (128k+). |
| Pre-tokenization regex | GPT-4 style split | Controls whether digits, whitespace and punctuation can merge into words. Splitting digits individually (or in groups of 3) measurably improves arithmetic. |
| Whitespace handling | Leading-space attached | `" the"` and `"the"` are different tokens — a classic source of prompt-sensitivity bugs. |
| Training corpus for the tokenizer | Sample of the real mix | A tokenizer trained on English-only text makes other languages 2–4× more expensive in tokens. |
| Special tokens | BOS/EOS/separator, reserved slots | Reserve spare slots up front; adding tokens later means resizing embeddings. |

**Metric to know:** *fertility* = tokens per word (or per byte). Compare
tokenizers by fertility on held-out text from each domain you care about. A
tokenizer with 15% lower fertility on your mix means ~15% fewer tokens for the
same text, so roughly 15% less token-proportional training compute — at a fixed
architecture, and ignoring the vocabulary-dependent cost of a larger
embedding/LM head.

> **How tokenization can make arithmetic harder:** with merged multi-digit
> tokens, `1234` might segment as `123`+`4` in one context and `12`+`34` in
> another, so the model never sees a stable place-value representation. Splitting
> digits individually (or in fixed groups of 3) removes that raggedness and
> measurably improves arithmetic. Note the limit of this argument: specific
> failures like `9.11 > 9.9` are *contested* — version-string and date priors in
> the training data are at least as plausible a cause. Tokenization is a real
> contributor to arithmetic difficulty, not a complete explanation of reasoning
> failure.

### Checkpoint 4

Train two BPE tokenizers (use `tokenizers` or `sentencepiece`) at 8k and 48k
vocab on the same 200MB corpus. Compare fertility on English, code, and a
non-English sample. Estimate the training-FLOP difference for a fixed document
set. Then find three inputs where one tokenizer segments in a way that would
obviously hurt the model.

---

## 5. Architecture

Modern pretrained LLMs are decoder-only transformers. The 2017 design has
accumulated a specific set of changes; learn a *representative modern* stack, then
the history.

### 5.1 The block

```
x ─┬─────────────────────────────────────┐
   │                                     │
   ├─► RMSNorm ─► Attention (RoPE, GQA) ─┴─► + ─┬───────────────────┐
                                                │                   │
                                                ├─► RMSNorm ─► MLP ─┴─► +  ─► x'
                                                              (SwiGLU)
```

Pre-norm (normalize *before* the sublayer, not after) with residual connections.
This is what makes deep stacks trainable — the residual stream is an
uninterrupted identity path from input to output.

### 5.2 Attention

Self-attention for a single head:

```
Attention(Q,K,V) = softmax( QKᵀ / √d_head + M ) V
```

`M` is the causal mask (`−inf` above the diagonal), which is what makes the model
autoregressive: position `t` sees only `≤ t`.

Variants, in the order they were adopted:

- **MHA** (multi-head): `n_heads` separate Q, K, V projections.
- **MQA** (multi-query): one shared K/V head. Shrinks the KV cache ~`n_heads`×;
  costs quality.
- **GQA** (grouped-query): `n_kv_heads` groups, e.g. 8 KV heads for 64 Q heads.
  The most common choice in open dense models — nearly MHA quality, near-MQA
  cache size.
- **MLA** (multi-head latent attention, DeepSeek): compress K/V into a
  low-rank latent, cache the latent. Smaller cache than GQA at better quality.

The KV cache is an *inference* concern, but it constrains *pretraining*
architecture choices, because you must serve what you train.

**Positional information.** Vanilla transformers add position embeddings.
Current practice:

- **RoPE** (rotary): rotate Q and K by a position-dependent angle so that the
  dot product depends only on *relative* position. Dominant choice. The `theta`
  base (10000 originally, often 500k+ for long-context models) sets the
  wavelength range.
- **ALiBi**: a linear distance penalty added to attention scores. Simple,
  extrapolates, largely superseded.
- **NoPE**: no positional encoding at all — causal masking alone leaks position.
  Works surprisingly well; appears in hybrid layer schemes.

### 5.3 The MLP

```
SwiGLU(x) = ( Swish(x W_gate) ⊙ (x W_up) ) W_down
```

Gated activations such as SwiGLU/GeGLU commonly outperform plain GELU at a
comparable parameter budget and are the usual choice in modern LLMs — though not
the only good one: `modded-nanogpt` ([Part 15](#15-open-source-repositories))
uses ReLU². Because the gate
adds a third matrix, the hidden dimension is set to `(8/3)·d_model` rounded to a
hardware-friendly multiple, keeping the parameter count comparable to a `4·d_model`
GELU MLP.

### 5.4 Normalization

**RMSNorm** is the dominant choice in modern decoder-only LLMs: `x / rms(x) · g`, no mean
subtraction, no bias. Cheaper and empirically equivalent. Related tricks:

- **QK-norm:** normalize Q and K before the dot product. A strong stabilizer for
  large runs; increasingly standard.
- **Logit soft-capping / z-loss:** keep output logits from drifting large.

Biases are generally removed from all linear layers — they cost parameters and
slightly hurt stability.

### 5.5 Mixture of Experts

Replace the MLP with `E` expert MLPs and a router that sends each token to the
top-`k` (typically 1–8, often with a few always-on "shared" experts). The *MLP*
parameters grow by ~`E`× — attention and embeddings are unchanged, so total
parameters grow by less — while FLOPs per token grow by only ~`k`×. You buy
capacity with memory instead of compute.

The hard part is **load balancing** — routers collapse onto a few experts. Two
approaches:

1. An auxiliary balancing loss added to the objective (classic; interferes with
   the language-modeling gradient).
2. Loss-free balancing: a per-expert bias on the routing scores, adjusted
   online to equalize load (DeepSeek-V3). Cleaner gradients.

MoE is now standard at frontier scale. Learn dense first — everything about MoE
is a modification of it.

### 5.6 Choosing a shape

For a dense model, given a parameter budget:

- `d_model` and `n_layers` trade off; aspect ratio `d_model / n_layers` around
  100–200 is the well-trodden band.
- `d_head` typically 64 or 128.
- Make the large matmul dimensions (`d_model`, `d_ff`, vocab size) multiples of
  64–128 so tensor cores and tensor parallelism divide cleanly. The exact
  requirement depends on dtype, GPU generation and TP degree; small dimensions
  like `d_head` just need to suit the attention kernel (64 or 128).
- Tied vs. untied input/output embeddings: tying saves `V·d_model` parameters and
  helps small models; large models usually untie.

Non-embedding parameter count for a dense SwiGLU model:

```
N ≈ n_layers · ( (2 + 2·n_kv_head/n_head)·d_model²  +  3·d_model·d_ff )
```

The attention term is Q and O at full width (`2·d²`) plus K and V shrunk by the
GQA ratio — with MHA (`n_kv_head = n_head`) it reduces to the familiar `4·d²`.
With `d_ff ≈ (8/3)·d_model` the MLP term is `≈ 8·d²`. Embeddings (`V·d_model`,
twice if untied) are excluded by convention, as in [Part 1](#1-the-mental-model).
Still, compute it programmatically — hand formulas drift with architecture.

### Checkpoint 5

Implement a decoder-only transformer from scratch in a single file: RMSNorm,
RoPE, GQA, SwiGLU, weight tying, causal masking. Train it on TinyStories until it
produces coherent sentences. ~300 lines. Do **not** copy nanoGPT — write it, then
diff against nanoGPT and understand every difference.

---

## 6. The objective and the loss

```python
logits = model(x)                       # (B, T, V)
loss = F.cross_entropy(
    logits.view(-1, V).float(),         # compute CE in fp32
    y.view(-1),
)
```

Every position contributes a prediction — a `(B, T)` batch gives `B·T` training
signals per forward pass. This density is why language modeling is so
sample-efficient as a pretext task.

**Teacher forcing.** During training the model always conditions on the *true*
prefix, never on its own samples. This creates train/inference mismatch
(exposure bias) — a known and largely tolerated cost.

**Units of loss.** Three ways of saying the same thing:

| Unit | Definition | Use |
|---|---|---|
| Cross-entropy (nats/token) | The raw loss | Training curves |
| Perplexity | `exp(loss)` | Intuition; "effective branching factor" |
| Bits per byte (BPB) | `loss · n_tokens / (ln2 · n_bytes)` | **Comparing across tokenizers** |

> Always compare models with BPB, never raw loss, unless the tokenizers are
> identical. A bigger vocabulary packs more text into each token, so per-token
> loss is mechanically higher even when the model is equally good or better on
> the same underlying text. BPB removes that artifact.

**Interpreting the curve.** A healthy run drops very fast for the first ~1% of
steps (learning unigram frequencies), then settles into a near-straight line on a
log-log plot. Deviations from that line are the signal you watch for.

### Checkpoint 6

Compute the loss of a model that outputs the unigram token distribution on your
corpus, and of a uniform model (`ln V`). Your training curve should pass through
the unigram baseline within a few hundred steps. If it doesn't, something is
broken — this is the cheapest sanity check in pretraining.

---

## 7. Optimization

### 7.1 The optimizer

**AdamW**, essentially universally. Standard hyperparameters for LLM pretraining:

```python
optimizer = torch.optim.AdamW(
    param_groups,           # see decay/no-decay split below
    lr=3e-4,                # peak; scale down as model grows
    betas=(0.9, 0.95),      # note: β₂=0.95, not the 0.999 default
    eps=1e-8,
    weight_decay=0.1,
    fused=True,
)
```

- `β₂ = 0.95` (rather than 0.999) shortens the second-moment window, which
  improves responsiveness and stability at large batch sizes.
- **Weight decay applies to matrices, not to norms and biases.** The usual
  implementation splits on `dim >= 2`. Note what that rule actually does:
  `nn.Embedding.weight` is 2-D, so it *is* decayed — nanoGPT and many production
  recipes do exactly this. Whether embeddings should be decayed is genuinely
  unsettled; some recipes exclude them (especially when tied to the LM head).
  Pick one and make your code and your notes agree — the common bug is a comment
  that says "no decay on embeddings" sitting above a rule that decays them.
- Gradient clipping at global norm `1.0`. A strong default that nearly every
  large-scale recipe uses (or replaces with an equivalent stability mechanism),
  and a useful telemetry signal (see Part 10).

Newer optimizers (Muon, Shampoo/SOAP, Lion) show real speedups, especially on
small/medium runs — Muon in particular has become the thing to beat in speedrun
benchmarks. Learn AdamW thoroughly first; it is still the safe default at scale.

### 7.2 Learning rate schedule

Two schedules dominate:

**Cosine decay with warmup** — the classic.
```
lr(t) = lr_peak · t/t_warmup                                  , t < t_warmup
      = lr_min + 0.5(lr_peak − lr_min)(1 + cos(π·progress))   , otherwise
```
Warmup is typically 0.1–2% of total steps (or a fixed ~2000 steps). Decay to
~10% of peak. The catch: the schedule depends on the *total* step count, so you
cannot cleanly extend a run.

**WSD (Warmup–Stable–Decay)** — warm up, hold a constant LR for most of training,
then decay sharply over the last ~10–20%. Advantages: you can stop at any point
by triggering the decay, intermediate checkpoints are reusable, and the decay
phase is a natural place to switch to high-quality data. This is the modern
default for runs whose length isn't fixed in advance.

```
lr │      ┌─────────────────────────┐
   │     /                           \
   │    /                              \___
   └───┴──────────────────────────────────── steps
       warmup        stable            decay
```

### 7.3 Batch size

Measured in **tokens per optimizer step**, not sequences. Typical: 0.5M–4M
tokens for mid-size models, growing with model size.

- Larger batches → less gradient noise → you can use a higher LR, up to a point.
- Beyond the **critical batch size**, doubling the batch stops buying you a
  proportional reduction in steps; you're burning compute for wall-clock.
- Critical batch size grows as loss falls, which is why some runs ramp batch size
  over training.

**Gradient accumulation** decouples the batch size you want from the memory you
have:

```python
for micro in range(accum_steps):
    x, y = get_batch(...)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(x, y) / accum_steps      # ← scale: see the note below
    loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step(); optimizer.zero_grad(set_to_none=True)
```

> **What forgetting `/ accum_steps` actually does.** It is widely said that this
> makes your effective learning rate `accum_steps`× too high. That is true for
> SGD and **false for Adam.** Adam's update is `m / (√v + ε)`; multiply every
> gradient by a constant `c` and, ε aside, you get `cm / √(c²v) = m / √v` —
> unchanged. (Bias correction scales both moments identically, and AdamW's
> weight decay is decoupled from the gradient, so neither breaks this. ε only
> matters when gradients are tiny — and scaling them *up* makes it matter less.) The
> real damage is to *clipping and telemetry*: the gradient norm is inflated `c`×,
> so `clip_grad_norm_(…, 1.0)` fires far more often — on most steps, unless the
> true norm is already below `1/accum_steps`. Clipping rescales each step's
> gradient by a *different* factor, which the moment history then mixes, so the
> clean cancellation above no longer describes the trajectory. Once clipping is
> nearly always active the optimizer is effectively following a fixed-norm direction,
> you have lost clipping as an early-warning signal ([Part 10](#10-stability-when-training-breaks)),
> and your logged grad-norm plot is meaningless. Scale the loss anyway — but know
> which failure you are preventing.

With DDP, disable gradient sync on all but the last micro-step
(`model.no_sync()`) or you pay `accum_steps`× the communication.

### 7.4 Initialization

- Normal with `std ≈ 0.02`, or `1/√d_model`.
- Scale residual-output projections (attention out-proj, MLP down-proj) by
  `1/√(2·n_layers)`. This keeps the variance of the residual stream from growing
  with depth and is the difference between a stable and an unstable deep model.
- Embeddings: same small normal init.

### 7.5 Hyperparameter transfer

You cannot tune hyperparameters at target scale. Two strategies:

- **Empirical scaling of HPs:** fit `lr*(N)` from a sweep at several small sizes
  and extrapolate. LR typically decreases roughly as a power of `N`.
- **μP (maximal update parametrization):** reparametrize initialization and
  per-layer learning rates so that the optimal LR is *invariant* to width. Tune
  on a 40M model, transfer the LR to a 40B model. Real, used in production, and
  worth understanding even if you don't adopt it.

### Checkpoint 7

On a small model, run a 6-point LR sweep at two widths (e.g. `d_model` 256 and
512). Plot final loss vs. LR. Confirm the optimum shifts with width under
standard parametrization. If you're ambitious, implement μP and confirm it
doesn't.

---

## 8. Scaling laws

### 8.1 The shape of the law

Loss as a function of parameters and data:

```
L(N, D) = E + A/N^α + B/D^β
```

`E` is the irreducible entropy of the text. The two terms are the penalties for
finite model size and finite data. Fitted exponents land near `α ≈ 0.34`,
`β ≈ 0.28`.

### 8.2 Chinchilla

Minimize `L(N, D)` subject to `C = 6ND`. The result: **`N` and `D` should scale
in roughly equal proportion**, giving a compute-optimal ratio of about

```
D ≈ 20 · N      tokens per parameter
```

Treat the 20 as an empirical fit, not a constant of nature. It comes from
Chinchilla's particular data, tokenizer and recipe, and later refits on other
setups land at different ratios. It is the right *order of magnitude* to reason
with.

This corrected the earlier Kaplan-era practice of training large models on too
few tokens. GPT-3 is the canonical example: 175B parameters on 300B tokens, which
is ~1.7 tokens per parameter — more than 10× under-trained by this criterion.

### 8.3 Why most modern models train past Chinchilla-optimal

Chinchilla minimizes *training* compute. Real deployments care about *inference*
compute, which is paid forever. A smaller model trained far past the optimal
point — Llama-3-8B saw 15T tokens, ~1875 tokens/param, ~90× Chinchilla — is worse
per training FLOP but much cheaper to serve.

**The practical rule:** use Chinchilla to reason about what a *fixed training
budget* buys. Use inference economics to choose the actual model size. They are
different questions.

### 8.4 How to actually use scaling laws

1. Pick 5–8 model sizes spanning ~2 orders of magnitude (e.g. 20M → 1B).
2. Train each on a compute-matched budget with a consistent recipe.
3. Fit the power law; check the residuals.
4. Extrapolate to your target — and **predict the target loss before you launch**.
5. If the real run departs from the prediction, investigate — it may be a bug,
   or a genuine change in regime your small runs could not see. Either way this
   is your primary early-warning system on a run that costs six figures.

Scaling laws also work for comparing *data mixes* and *architectures*: fit a
curve for each candidate and compare slopes, not single points. A recipe that
wins at 100M can lose at 10B.

### Checkpoint 8

Train 5 models from ~2M to ~50M parameters on a fixed tokens-per-param ratio.
Fit `L = E + A/N^α`. Then train a 6th model ~3× larger than your biggest and
check whether your fit predicted its loss to within a few percent.

---

## 9. Systems and parallelism

### 9.1 The memory budget

One common setup for a dense model trained with AdamW and bf16 mixed precision
(the exact number depends on the recipe — see the note below the table), per
parameter:

| Item | Bytes/param |
|---|---|
| bf16 weights | 2 |
| fp32 master weights | 4 |
| fp32 gradients | 4 |
| Adam `m` (fp32) | 4 |
| Adam `v` (fp32) | 4 |
| **Total** | **~18** |

You will also see **16 bytes/param** quoted (e.g. in the ZeRO paper): that
convention keeps gradients in bf16 (2 bytes) rather than fp32. PyTorch autocast —
what the Part 16 notebook uses — also lands at 16, but differently: it keeps a
single fp32 copy of the weights and casts to bf16 on the fly, so there is no
separate bf16 weight copy. Other optimizers and sharding schemes give anything
from ~12 to 18+. Check which setup a source is using before comparing numbers.

Plus activations, which scale with `batch × seq_len × d_model × n_layers`, and at
long context can dwarf everything else.

Under the 18-byte setup above, a 7B model needs ~126 GB for states alone (~112 GB
at 16 bytes) — already past a single 80GB
GPU before a single activation is stored. **This is why parallelism exists.**

### 9.2 Mixed precision

Compute in **bf16** (wide exponent, no loss scaling needed — unlike fp16) and
accumulate matmuls in fp32. Keep the weights the optimizer updates in fp32 —
either as a separate master copy or, with autocast, as the only copy (the recipes
differ; see 9.1). Keep the final logits and the cross-entropy in fp32. fp8 training is now in production at frontier labs
(DeepSeek-V3 trained the bulk of GEMMs in fp8) but requires careful per-block
scaling; treat it as advanced.

### 9.3 The parallelism dimensions

| Kind | Splits | Communicates | Use when |
|---|---|---|---|
| **Data (DDP)** | the batch | gradients (all-reduce, once per step) | Always, as the outer dimension |
| **ZeRO / FSDP** | optimizer states, grads, params across DP ranks | params (all-gather) + grads (reduce-scatter) | Model doesn't fit; the default way to scale |
| **Tensor (TP)** | individual weight matrices (attention QKV/out, MLP up/down) across GPUs | activations via all-reduce inside every layer — twice per layer forward, twice backward | Within a node. It is on the critical path of every layer, so it needs NVLink-class bandwidth; across the inter-node network it is too slow |
| **Pipeline (PP)** | layers into stages (e.g. layers 1–8 on GPU 0, 9–16 on GPU 1) | point-to-point sends: activations forward, gradients backward, only at stage boundaries | Across nodes, where bandwidth can't sustain TP; introduces "bubble" idle time |
| **Sequence / Context** | the sequence dimension | attention K/V (ring attention) | Long context |
| **Expert (EP)** | MoE experts | all-to-all token routing | MoE models |

Real frontier runs combine 4–5 of these ("5D parallelism"). The standard layout
is TP within a node, PP across a small group of nodes, DP/FSDP across everything
else.

**ZeRO stages**, worth knowing precisely:
- Stage 1: shard optimizer states across the data-parallel ranks. Optimizer
  memory shrinks in proportion to the number of ranks. Total model-state memory
  shrinks less: the ZeRO paper quotes ~4× at large data-parallel degree under its
  16-byte accounting, closer to ~3× with the 18-byte table above. Same
  communication volume as plain DDP.
- Stage 2: also shard gradients.
- Stage 3 (≈ FSDP's full-shard mode): also shard parameters; gather them
  just-in-time per layer. Maximum savings, most communication. (FSDP also has
  a mode that shards only gradients and optimizer states — roughly Stage 2.)

### 9.4 Making it fast

- **FlashAttention** — compute attention in tiles that fit in SRAM, never
  materializing the `T×T` matrix. Turns attention memory from `O(T²)` to `O(T)`
  and is substantially faster. Use it (via `F.scaled_dot_product_attention` or
  the `flash-attn` package); never hand-roll attention for a real run.
- **Activation checkpointing** — discard activations in the forward pass,
  recompute them in the backward. Trades ~30% more compute for a large memory
  saving. Selective checkpointing (only the cheap-to-recompute layers) is better
  than full.
- **`torch.compile`** — kernel fusion; often 1.2–1.8× for free.
- **Overlap communication with compute** — FSDP prefetching, gradient bucketing.
  Without overlap, comms can be 30%+ of step time.

### 9.5 MFU — the number that matters

```
MFU = (6 · N · tokens_per_second) / (n_gpus · peak_FLOPs_per_gpu)
```

Interpretation: what fraction of your hardware's theoretical peak you are
actually converting into model FLOPs.

`6N` counts only the parameter matmuls, which is accurate for large models. It
undercounts when other FLOPs are a big share: the LM-head projection (`6·V·d`
per token, left out of non-embedding `N`) and attention's `QKᵀ`/`AV` (roughly
`12·n_layer·d_model·seq_len` per token, which grows with context). For small
models or long sequences, add both terms — the Stage 11 notebook does.

This makes published MFU numbers only loosely comparable: some use bare `6N`,
PaLM's original definition adds the attention term, and some count the LM head.
When you compare against a paper, use *its* formula — which is why Stage 11 logs
both the bare `6N` figure and the fuller estimate.

**On large, well-shaped training workloads, 35–55% is good.** Below ~25% on such
a workload usually means a fixable problem — dataloading, missing overlap,
unfused kernels, or bad shapes. Small models are different: at the Part 16
notebook's size, 15–25% is normal, because kernel-launch overhead dominates.
Expected MFU depends on model size, sequence length, hardware and parallelism,
so compare against runs like yours. It should be on your dashboard from step one.

### 9.6 Fault tolerance

At 1000+ GPUs for months, hardware *will* fail — Meta reported hundreds of
interruptions over the Llama-3-405B run, the majority hardware-related.
Requirements:

- Checkpoint frequently and asynchronously (write to local NVMe, upload in
  background).
- Save model, optimizer, LR schedule, **and dataloader position**. Forgetting
  the dataloader state means silently re-training on the same tokens.
- Automatic restart from last checkpoint with node replacement.
- Detect stragglers and silent data corruption (a single flaky GPU producing bad
  gradients can poison a run for hours before it shows in the loss).

### Checkpoint 9

Take your Part 5 model. Measure MFU on one GPU. Then: enable `torch.compile`,
swap in FlashAttention, tune batch size, and fix the dataloader. Record MFU after
each change. Then scale to 2 GPUs with DDP and verify the loss curve matches the
1-GPU run at the same token count.

---

## 10. Stability: when training breaks

The characteristic failure of a large pretraining run is a **loss spike** — the
loss jumps by 0.5–3 nats and either recovers over thousands of steps or diverges.

**Leading indicators**, all of which you should be logging:

- gradient norm (pre-clip) — spikes here precede loss spikes
- fraction of steps being clipped
- per-layer activation RMS and max
- attention logit maximum (the classic culprit: attention logits growing until
  softmax saturates)
- output logit magnitude
- Adam second-moment `v` distribution
- parameter update / parameter magnitude ratio (should sit near `10⁻³`)

**Preventive measures**, roughly in order of adoption:

| Measure | What it does |
|---|---|
| QK-norm | Bounds attention logits directly. Very effective. |
| z-loss (`10⁻⁴ · log²Z`) | Keeps the softmax normalizer near 1, preventing logit drift. |
| Lower `β₂` (0.95) | Faster adaptation to gradient-scale changes. |
| Tuning Adam `eps` | Two *opposite* adjustments exist and get confused. **Raising** eps (say 1e-8 → 1e-6) damps updates when `v` is tiny, guarding against huge steps on near-zero gradients. **Lowering** it (1e-8 → 1e-15, as in some large-model recipes) stops eps from dominating when gradient RMS is genuinely small, which would otherwise shrink updates. Know which problem you have before changing it. |
| Residual-scaled init | Keeps deep residual streams bounded. |
| Longer warmup | Cheapest fix for early-training instability. |
| Embedding norm / logit soft-cap | Bounds the two ends of the network. |

**When a spike happens anyway:** the standard playbook is to rewind to a
checkpoint ~100–500 steps before the spike, skip the data batches involved, and
resume (optionally at a lower LR). Spikes are frequently triggered by specific
pathological documents — long runs of repeated characters, corrupted encodings.

> **Debugging heuristic:** if the loss diverges immediately, it's your LR or
> init. If it diverges after hours of healthy training, it's data or numerics.
> If loss sits at exactly `ln V` and never moves, **no learning is happening at
> all** — LR is 0, the optimizer was handed the wrong parameter list,
> `zero_grad()` runs after `backward()` instead of after `step()`, or the graph
> is detached. Note what this is *not*: misaligned labels still let the model fit
> unigram frequencies, so a shifted-target bug drops the loss below `ln V` and
> then stalls at a visibly-too-high plateau. Flat-at-`ln V` means dead, not
> misaligned.

### Checkpoint 10

Deliberately break your training run four ways: (1) LR 30× too high, (2) no
residual init scaling at 24 layers, (3) shift targets by one extra position,
(4) inject a document of 50k repeated `"a"` tokens. Record the loss curve and
grad-norm signature of each. You are building a diagnostic vocabulary — these
four shapes cover most real failures.

---

## 11. Evaluation

Base models are evaluated differently from chat models. They cannot follow
instructions; you evaluate them by likelihood and few-shot completion.

### 11.1 Loss-based

The most reliable signal during training. Held-out **bits-per-byte** on each
domain you care about (web, code, math, books, and any target domain). Low
variance, immediately interpretable, no prompt sensitivity. Track a per-domain
panel, not one aggregate number.

### 11.2 Benchmarks

Scored by comparing the model's likelihood of each answer option, usually with
few-shot prompting:

| Benchmark | Measures | Note |
|---|---|---|
| HellaSwag | Commonsense completion | Smooth and early-signal; good for small models |
| ARC-Easy/Challenge | Science QA | |
| PIQA, WinoGrande | Physical/coreference commonsense | |
| MMLU | 57-subject knowledge | Historically near-random below ~7B params, but that threshold has moved a long way down — well-trained 0.5–2B models now score far above chance. Still noisy at *your* notebook scale. |
| GSM8K, MATH | Math reasoning | Needs generation. Weak in small base models, but modern small models with math-heavy pretraining are no longer at zero. |
| HumanEval, MBPP | Code | |

**Scoring detail that matters:** normalize option likelihood by token count or
by unconditional likelihood, or you systematically favor short answers. Different
harnesses make different choices, which is why "the same model" scores differently
across leaderboards. Use a single harness (`lm-evaluation-harness` is standard)
and report its settings.

### 11.3 Traps

- **Contamination.** Re-check after every data change, not once.
- **Small-model noise.** At the scale you can train in a notebook (tens of
  millions of parameters), multiple-choice benchmarks are pure noise. Don't steer
  a recipe on them; use loss.
- **Eval on the anneal.** Benchmark scores move a lot during the final decay
  phase. Mid-run scores under-predict final scores.
- **Prompt-format sensitivity.** Base models can swing several points on
  formatting alone. Fix the format and never change it mid-project.

### Checkpoint 11

Run `lm-evaluation-harness` on two public base models of different sizes (e.g.
Pythia-160M and Pythia-1.4B). Reproduce the published HellaSwag numbers. Then
change the few-shot count and the normalization and see how much you can move
the score without touching the model.

---

## 12. The hands-on ladder

Each rung is a complete, finishable project. Don't skip; each one teaches
something the next assumes.

> **[Part 16](#16-the-end-to-end-process--notebook-blueprint) is a scaled-down
> version of Rungs 3–6** — same stages, same order, runnable code, but at
> ~31M parameters on TinyStories rather than the 100–350M on FineWeb that Rung 5
> calls for. Do Part 16 first to get the whole pipeline working end to end, then
> use these rungs to scale each piece up to something real.

**Rung 1 — Autograd (1 day).**
Write micrograd. Train an MLP on a toy dataset. *You now know what backprop is.*

**Rung 2 — Character transformer (2 days).**
Karpathy's `nanoGPT`-from-scratch on Shakespeare. ~10M params in the default
char-level config, one modest GPU. *You now know what attention computes.*

**Rung 3 — Modern architecture (3 days).**
Rewrite it with RMSNorm, RoPE, GQA, SwiGLU. Train on TinyStories. Generate
coherent text. *You now know the current stack, not the 2017 one.*

**Rung 4 — Real data pipeline (3 days).**
Download 10GB of FineWeb. Filter, dedup, train a BPE tokenizer, tokenize to
binary shards, build a memmap loader. *This is the part most tutorials skip and
most real work consists of.*

**Rung 5 — A real (small) pretraining run (1 week).**
~100–350M params, ~5–10B tokens, single GPU or one node. Pick the pair
deliberately: those ranges span ~14 to ~100 tokens/param, from roughly
Chinchilla-optimal (350M on 7B) to well past it (100M on 10B) — see [Part 8.3](#83-why-most-modern-models-train-past-chinchilla-optimal).
Full instrumentation: loss, grad norm, LR, MFU, throughput, per-domain eval loss,
periodic benchmark runs, resumable checkpoints. *This is a complete pretraining
run in miniature; everything above it is scale.*

**Rung 6 — Scaling law (4 days).**
Five sizes, fit the curve, predict a sixth before training it. *You now have the
core professional skill: forecasting instead of guessing.*

**Rung 7 — Multi-GPU (1 week).**
DDP → FSDP. Verify loss-curve equivalence. Profile MFU, find the dominant
bottleneck, and improve it measurably toward a sensible baseline for your model
size and hardware ([Part 9.5](#95-mfu--the-number-that-matters)).
Introduce a fault mid-run and recover cleanly from checkpoint. *You now
understand the systems layer.*

**Rung 8 — Pick one depth (open-ended).**
- MoE: implement top-k routing with loss-free balancing.
- Long context: extend your model to 32k via RoPE base scaling + a fine-tune.
- Efficiency: enter the modded-nanogpt speedrun mindset — fixed target loss,
  minimize wall-clock.
- Data: reproduce a published filtering ablation and see if it holds at your scale.

**Compute reality check.** Rungs 1–4 run on a laptop or a single consumer GPU.
Rung 5 is ~1 GPU-week on a 4090, or a few hundred dollars of rented A100/H100
time. Rung 7 needs a multi-GPU node for a day or two. Budget accordingly — the
learning is front-loaded into the cheap rungs.

---

## 13. After pretraining

Where the boundary is, so you know what pretraining is *not* responsible for.

**Midtraining / annealing.** The final 10–20% of tokens, at decaying LR, on a
much higher-quality mix — textbooks, curated math and code, some
instruction-formatted data, long-context data. This phase moves benchmark scores
disproportionately and is increasingly treated as its own stage with its own
data recipe, not as "the end of pretraining."

**Long-context extension.** Usually done after the main run: increase RoPE
`theta` (or apply YaRN/position interpolation), continue training on a few
billion long-document tokens at the new context length. Cheaper and more stable
than training long-context from scratch.

**Post-training.** Typically some combination of: SFT on instruction data;
offline preference optimization (DPO and relatives — no online RL loop); and
reinforcement learning (PPO, GRPO), including RL on verifiable rewards for
reasoning. The mix and order vary by model and goal. This is where the base
model becomes an assistant. It's a different literature with a different rhythm —
small data, fast iteration — and is where "reasoning models" come from.

**The dividing line:** pretraining supplies most of *what the model knows and can
represent*. Post-training mostly shapes *how those capabilities are elicited and
used*. Post-training can add capabilities, but teaching a genuinely new domain
there is usually far less efficient than learning it in pretraining or continued
pretraining — and shaping behavior well is its own hard problem, not a
formality.

---

## 14. Reading list

Read in this order. Papers marked ★ are the ones to read fully; others you can
skim for their core result.

**Foundations**
1. ★ *Attention Is All You Need* (2017) — the architecture.
2. ★ *Language Models are Unsupervised Multitask Learners* (GPT-2, 2019) — the framing.
3. *Language Models are Few-Shot Learners* (GPT-3, 2020) — scale as capability.

**Scaling**
4. *Scaling Laws for Neural Language Models* (Kaplan et al., 2020).
5. ★ *Training Compute-Optimal Large Language Models* (Chinchilla, 2022) — the correction.
6. *Tensor Programs V / μTransfer* (2022) — hyperparameter transfer.

**Recipes (read at least two end to end)**
7. ★ *LLaMA* and *Llama 2 / 3* papers — the canonical open recipe and its evolution.
8. ★ *OLMo 2* — the most transparent open recipe; data, stability fixes, and ablations all published.
9. *DeepSeek-V3* — MoE at frontier scale, MLA, fp8, loss-free balancing.

**Data**
10. ★ *The FineWeb Datasets* (2024) — read the blog version; the best practical writeup on web data processing that exists.
11. *DataComp-LM (DCLM)* (2024) — filtering as the dominant variable.
12. *Deduplicating Training Data Makes Language Models Better* (2022).
13. *The Pile* / *Dolma* — corpus construction documentation.

**Architecture components**
14. *RoFormer* (RoPE), *GLU Variants Improve Transformer* (SwiGLU), *Root Mean Square Layer Normalization* (RMSNorm) — short, read all three in an hour.
15. *GQA: Training Generalized Multi-Query Transformer Models* (2023).

**Systems**
16. ★ *FlashAttention* (and v2/v3) — the key efficiency idea.
17. *Megatron-LM* — tensor and pipeline parallelism.
18. *ZeRO* — memory sharding.
19. ★ *The Ultra-Scale Playbook* (Hugging Face) — the best single practical resource on distributed training; effectively a textbook.

**Code to read** — see [Part 15](#15-open-source-repositories).

---

## 15. Open-source repositories

The field's real documentation is its code. This is a map of it, organized by
what each repo is *for*. You do not need most of these — the ★ entries are the
ones worth actually reading line by line.

### 15.1 Minimal and educational

Read these first. They fit in your head, which is the entire point.

| Repo | What it is | Why it matters |
|---|---|---|
| ★ `karpathy/nanoGPT` | ~600 lines: GPT-2 training + sampling | The reference minimal implementation. Everyone's mental model of a training loop comes from here. |
| ★ `karpathy/build-nanogpt` | nanoGPT rebuilt commit-by-commit, with a 4-hour video | Reproduces GPT-2 124M for ~$10 on rented GPUs. The best guided first run in existence. |
| ★ `karpathy/nanochat` | The *whole* pipeline in ~8k readable lines: tokenizer → pretrain → midtrain → SFT → RL → inference + web UI | "The best ChatGPT $100 can buy." The single best repo for seeing how all the stages connect. |
| `karpathy/llm.c` | GPT-2/GPT-3 pretraining in raw C/CUDA, no PyTorch | Strips away the framework. Read it when you want to know what the framework was doing. |
| `karpathy/minGPT` | nanoGPT's predecessor, even smaller | Superseded, but a clean 300-line read. |
| ★ `KellerJordan/modded-nanogpt` | Speedrun: fixed target loss, minimize wall-clock | A live, competitive catalogue of what actually works — Muon, QK-norm, architecture tweaks — validated by leaderboard rather than by paper. Read the record history as a changelog of the field. |

**Start here:** `build-nanogpt` to learn, `nanochat` to see the full picture,
`modded-nanogpt` to find out what's current.

### 15.1b After nanoGPT: where to go next

nanoGPT teaches one thing completely — a single-node training loop over a 2019
architecture. Everything it leaves out falls into four directions. They are
different skills; pick by what you want to be good at, don't try to do all four
at once.

**Direction A — the recipe frontier (what's actually in a modern model).**

| Repo | What it adds over nanoGPT | Difficulty |
|---|---|---|
| ★ `KellerJordan/modded-nanogpt` | Same task, same data, same target loss — but Muon, QK-norm, ReLU², untied and value-residual embeddings, U-net-style inter-layer skips, FlexAttention with sliding-window block masks, fp8 head, warmup-free schedules, batch-size ramp. The record has fallen by more than an order of magnitude since the nanoGPT baseline. | Low — it's still one file |
| `facebookresearch/lingua` | Meta's "lean and hackable" research codebase: modern arch, real configs, proper checkpointing and eval, but still small enough to read. | Low-medium |
| `allenai/OLMo-core` | A current production recipe in full: annealing, data mix scheduling, stability fixes, resumption. | Medium |

This is the highest value-per-hour direction and the natural immediate next step.
`modded-nanogpt` is *literally* nanoGPT with six years of progress applied, so
every diff is legible to you already. Read the record-holder's `train_gpt.py`,
then walk backwards through the PR history — each record is a one-idea
ablation with a measured result attached. It is the best-annotated
architecture-ablation dataset that exists, and none of it is in papers.

**Direction B — distributed systems (the actual bottleneck at scale).**

| Repo | What it teaches | Difficulty |
|---|---|---|
| ★ `huggingface/nanotron` | 3D parallelism (DP/TP/PP) in a few thousand readable lines. Read with the *Ultra-Scale Playbook*, which is its textbook. | Medium |
| ★ `pytorch/torchtitan` | FSDP2 + TP + PP + context parallel + fp8 + `torch.compile`, written to be read. Includes Llama-3/4 and MoE. The reference for how PyTorch itself thinks parallelism should look. | Medium |
| `NVIDIA/Megatron-LM` | The canonical implementation everything descends from (Megatron-Core is not a separate repo — it lives here under `megatron/core`). Not pleasant to read, but you should be able to navigate it. | High |
| `deepseek-ai/DualPipe`, `DeepEP`, `FlashMLA`, `DeepGEMM` | Frontier-lab systems work, released standalone: bidirectional pipeline scheduling, expert-parallel all-to-all comms, MLA kernels, fp8 GEMMs. Small, focused, and *far* past tutorial level. | High |
| `stanford-crfm/levanter`, `google/maxtext` | JAX. Different mental model — `jit`, sharding annotations instead of manual collectives. Worth a day even if you never use JAX. | Medium |

Order: nanotron → torchtitan → Megatron when you need something specific.

**Direction C — below the framework.**

| Repo | What it teaches | Difficulty |
|---|---|---|
| ★ `karpathy/llm.c` | GPT-2/3 pretraining in raw C/CUDA. Hand-written kernels for every op. Removes all magic — after this you know exactly what PyTorch was doing. | Medium-high |
| `linkedin/Liger-Kernel` | Production Triton kernels (fused RMSNorm, RoPE, SwiGLU, fused linear+cross-entropy). The most readable real Triton in the ecosystem, and the fused-CE trick alone is worth the read. | Medium |
| `Dao-AILab/flash-attention` | The tiling/online-softmax idea in its native habitat. Read the v2 paper alongside the CUDA. | High |
| `NVIDIA/TransformerEngine` | How fp8 training actually works: per-tensor scaling, delayed scaling, recipe management. | High |

**Direction D — data at scale.**

| Repo | What it teaches | Difficulty |
|---|---|---|
| ★ `huggingface/datatrove` | The FineWeb pipeline. Distributed filtering, MinHash dedup, tokenization, over Slurm/S3. | Medium |
| `mlfoundations/dclm` | Data curation as a controlled benchmark — how to *prove* a filtering change helps. | Medium |
| `allenai/dolma` | Rust-backed taggers and dedup at trillion-token scale. | Medium |

Least glamorous, most underrated. If you want to be employable in pretraining
specifically, this is the direction with the least competition.

**A concrete next two weeks.** Read `modded-nanogpt`'s current record file and
port three of its ideas into your own nanoGPT fork, measuring each in isolation
(Direction A). Then read `nanotron` alongside the Ultra-Scale Playbook and
convert your fork from DDP to FSDP2 + TP on two GPUs, verifying the loss curve is
unchanged (Direction B). That combination takes you from "I can train a small
GPT" to "I can read and modify a real pretraining stack."

### 15.2 Complete open research stacks

Real pretraining runs with published code, data, checkpoints and logs. These are
what you imitate for a serious project.

| Repo | Notes |
|---|---|
| ★ `allenai/OLMo` + `allenai/OLMo-core` | The most transparent full stack in the world: training code, data, intermediate checkpoints, W&B logs, and honest writeups of what broke. If you read one production codebase, read this one. |
| `EleutherAI/gpt-neox` | Megatron + DeepSpeed based; trained GPT-NeoX-20B and the Pythia suite. Battle-tested, widely forked. |
| ★ `EleutherAI/pythia` | 16 models, 70M→12B, identical data in identical order, 154 checkpoints each. Not a training framework — a *research instrument*. The standard substrate for studying training dynamics and memorization. |
| `huggingface/smollm` | SmolLM2/3 — full small-model recipes, data mixes and ablations, published in detail. Accompanied by the "Smol Training Playbook" writeup, which is excellent. |
| `LLM360/*` (Amber, CrystalCoder, K2) | Fully open: data ordering, all intermediate checkpoints, metrics. Explicitly built for reproducibility. |
| `mosaicml/llm-foundry` | The MPT models. Production-shaped, opinionated, good YAML-driven config patterns. |
| `bigscience-workshop/Megatron-DeepSpeed` | BLOOM. Historically important; the [training chronicles](https://github.com/bigscience-workshop/bigscience/blob/master/train/tr11-176B-ml/chronicles.md) documenting every failure of a 176B run are required reading for anyone about to launch something large. |

### 15.3 Training frameworks and parallelism

What you use when the model stops fitting on one GPU.

| Repo | Role |
|---|---|
| `NVIDIA/Megatron-LM` | Origin of tensor/pipeline/sequence parallelism. Dense, but canonical — most large-scale code descends from it. |
| `NVIDIA/NeMo` | The productized, modular framework built on Megatron-Core. What many labs actually run. |
| `microsoft/DeepSpeed` | ZeRO stages 1–3, offload, and a large bag of training optimizations. |
| ★ `pytorch/torchtitan` | PyTorch-native reference for 4D parallelism (FSDP2 + TP + PP + CP), `torch.compile`, fp8. Written to be *read* — the best modern entry point to distributed training. |
| ★ `huggingface/nanotron` | Minimal 3D parallelism, few thousand lines. The companion code to the *Ultra-Scale Playbook*. Read alongside it. |
| `google/maxtext` | JAX/TPU, pure-Python, exceptionally high MFU. The reference if you're on TPUs. |
| `stanford-crfm/levanter` | JAX; emphasizes bitwise reproducibility and resumability across hardware configs. A genuinely different set of ideas. |
| `hpcaitech/ColossalAI`, `InternLM/InternEvo` | Alternative full-featured distributed stacks. |
| `facebookresearch/metaseq` | OPT-175B, plus the famous logbook of a real large run going wrong. |

**Start here:** `nanotron` to understand parallelism, `torchtitan` to use it.

### 15.4 Data

Where most of your actual engineering time will go.

| Repo | Role |
|---|---|
| ★ `huggingface/datatrove` | The pipeline that built FineWeb: filtering, dedup, tokenization, distributed over Slurm/local/S3. The practical default for web-scale data processing. |
| `allenai/dolma` | AI2's data toolkit + the Dolma corpus. Rust-backed taggers and dedup. |
| `mlfoundations/dclm` | DataComp-LM: a *benchmark* for data curation, with the winning filtering recipe. Read this to see filtering evaluated scientifically. |
| `google-research/deduplicate-text-datasets` | Suffix-array exact substring dedup. The reference implementation. |
| `ChenghaoMou/text-dedup` | Practical MinHash/SimHash/suffix-array dedup collection. Easy starting point. |
| `togethercomputer/RedPajama-Data` | Full open reproduction of the LLaMA data recipe, with quality signals precomputed. |
| `huggingface/tokenizers`, `google/sentencepiece`, `openai/tiktoken` | Tokenizer training (first two) and fast inference-time encoding (third). |

### 15.5 Kernels and efficiency

| Repo | Role |
|---|---|
| ★ `Dao-AILab/flash-attention` | The attention implementation. Non-optional for real runs. |
| `linkedin/Liger-Kernel` | Drop-in fused Triton kernels (RMSNorm, RoPE, SwiGLU, fused cross-entropy). Meaningful memory and throughput wins for a one-line change — and readable Triton to learn from. |
| `NVIDIA/TransformerEngine` | fp8 training primitives on Hopper/Blackwell. |
| `pytorch/ao` | Quantization and low-precision training in native PyTorch. |
| `triton-lang/triton` | Write your own kernels. The realistic path into custom GPU code. |
| `KellerJordan/Muon`, `facebookresearch/optimizers`, `nikhilvyas/SOAP` | Muon; Distributed Shampoo; SOAP (its own repo, not part of the Meta one). The live frontier of optimizer work. |

### 15.6 Evaluation

| Repo | Role |
|---|---|
| ★ `EleutherAI/lm-evaluation-harness` | The de facto standard. Whatever you report, report it with this and state the settings. |
| `huggingface/lighteval` | Lighter, configurable alternative; backs the HF leaderboards. |
| `allenai/OLMES` | A *standardized* eval recipe — fixes prompt format and normalization so numbers are actually comparable across models. Addresses the trap in [Part 11](#11-evaluation). |

### 15.7 How to read a repo like this

Don't start at `main()`. In any training codebase, four files hold nearly all the
information:

1. **The model definition** (`model.py`, `modeling_*.py`) — confirms the actual
   architecture, which often differs from the paper.
2. **The training loop** — where the optimizer step, clipping, accumulation and
   schedule actually live.
3. **The config for a real run** (`configs/*.yaml`) — the ground truth on
   hyperparameters. Diff two labs' configs against each other; the disagreements
   are where the open questions are.
4. **The dataloader** — usually the least glamorous and most bug-prone file, and
   the one that determines whether your run is reproducible.

### Checkpoint 15

Diff the configs of a real OLMo 2 run against a SmolLM run at comparable size.
List every hyperparameter where they disagree — LR, batch size, warmup, weight
decay, z-loss, init, data mix — and for each, write one sentence on why they
might have chosen differently. This is the fastest way to learn what is settled
practice and what is still taste.

---

## 16. The end-to-end process — notebook blueprint

This part is the pipeline in execution order. Each **Stage** is designed to become
one notebook section, and they run top to bottom: state created in Stage 3 is used
in Stage 5, never the reverse. The one exception is that Stage 12's helper cell
(12a) must sit above Stage 11 — see the [cell map](#160-cell-map).

Every stage has the same fields:

- **Goal** — the one thing this stage produces.
- **What's happening** — the concept, in three sentences.
- **Code** — a runnable cell.
- **Predict before you run** *(most stages)* — write down what you expect first;
  the gap between your guess and the output is where the learning is.
- **Verify** — how you know it worked *before* moving on. Do not skip these; a
  pretraining bug found three stages late costs hours.
- **Breaks like this** — the failure signatures specific to this stage.

### 16.0 Cell map

GPU column: **yes** means impractical without one at `small` or `base`. Every
stage runs on CPU at the `tiny` preset — that is what `tiny` is for.

| Cell | Stage | Runtime (`small` preset) | GPU |
|---|---|---|---|
| 1 | [Setup and config](#stage-1--setup-and-config) | seconds | no |
| 2 | [Acquire raw text](#stage-2--acquire-raw-text) | 5–20 min | no |
| 3 | [Inspect and filter](#stage-3--inspect-and-filter) | 1–5 min | no |
| 4 | [Deduplicate](#stage-4--deduplicate) | 2–10 min | no |
| 5 | [Train the tokenizer](#stage-5--train-the-tokenizer) | 2–10 min | no |
| 6 | [Tokenize to shards](#stage-6--tokenize-to-shards) | 5–20 min | no |
| 7 | [Dataloader](#stage-7--the-dataloader) | seconds | no |
| 8 | [Build the model](#stage-8--build-the-model) | seconds | yes |
| 9 | [Pre-flight sanity checks](#stage-9--pre-flight-sanity-checks) | 1–2 min | yes |
| 12a | [Checkpoint helpers](#stage-12--checkpoint-and-resume) | seconds | no |
| 10 | [Optimizer and schedule](#stage-10--optimizer-and-schedule) | seconds | yes |
| 11 | [The training loop](#stage-11--the-training-loop) | ~1 h (A100) – 5 h (T4) | yes |
| 12b | [Verify checkpoint round trip](#stage-12--checkpoint-and-resume) | seconds | yes |
| 12c | [Fast resume after restart](#stage-12--checkpoint-and-resume) (replaces 2–6) | seconds | no |
| 13 | [Evaluate](#stage-13--evaluate) | 1–5 min | yes |
| 14 | [Anneal (two arms)](#stage-14--anneal) | 2 × the decay phase | yes |
| 15 | [Fit a scaling law](#stage-15--fit-a-scaling-law) | ~3–4× cell 11 | yes |

**Cell 12a runs before cell 10** — Stage 11 calls `save_ckpt`/`load_ckpt`, so
their definitions must already exist. That is the only ordering exception on a
first run; every other stage runs in numeric order. Cell 12c is only for resuming
after a kernel restart.

**Deliberately out of scope for a notebook:** multi-node parallelism, fp8,
trillion-token pipelines, fault tolerance. Those need a cluster, not a cell.
[Part 9](#9-systems-and-parallelism) covers the concepts; Stage 16 says how to
graduate.

---

### Stage 1 — Setup and config

**Goal.** One config object every later cell reads from, with size presets so the
same notebook runs on a laptop or an A100.

**What's happening.** Pretraining has ~30 coupled hyperparameters. Fixing them in
one place — and deriving everything you can rather than hardcoding it — is the
difference between a notebook you can experiment with and one you can only run
once. The presets let you debug at `tiny` (seconds per step) and then scale the
exact same code.

```python
# Cell 1 — config
!pip install -q torch numpy scipy datasets tokenizers tqdm matplotlib

import math, os, time, json, random
from dataclasses import dataclass, asdict, field
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

@dataclass
class Config:
    # --- data ---
    dataset: str = "roneneldan/TinyStories"   # swap for a FineWeb slice later
    n_docs: int = 400_000
    data_dir: str = "data"

    # --- tokenizer ---
    vocab_size: int = 16_000

    # --- model ---
    d_model: int = 512
    n_layer: int = 8
    n_head: int = 8
    n_kv_head: int = 2          # GQA: n_head must be divisible by this
    seq_len: int = 512
    rope_theta: float = 10_000.0
    ffn_multiple: int = 128     # round SwiGLU hidden dim to this
    tie_embeddings: bool = True

    # --- optimization ---
    batch_size: int = 24        # micro-batch (per forward)
    grad_accum: int = 4         # tokens/step = batch_size*grad_accum*seq_len
    lr: float = 6e-4
    min_lr_frac: float = 0.1
    warmup_frac: float = 0.02
    decay_frac: float = 0.20    # WSD: final fraction spent decaying
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    max_steps: int = 6_000

    # --- runtime ---
    seed: int = 1337
    compile: bool = True
    eval_every: int = 250
    ckpt_every: int = 1_000
    out_dir: str = "runs/small"

PRESETS = {
    # ~0.8M non-emb params — CPU-viable. A SMOKE TEST, not a real run:
    # deliberately far below Chinchilla, just enough to prove the pipeline works.
    "tiny":  dict(d_model=128, n_layer=4,  n_head=4,  n_kv_head=2,
                  seq_len=256, vocab_size=8_000, batch_size=8, grad_accum=1,
                  max_steps=2_000, lr=1e-3, n_docs=20_000, out_dir="runs/tiny"),
    # ~22M non-emb params, ~295M tokens (~13 tok/param).
    # ~1h on an A100, more like 3-5h on a T4.
    "small": dict(),
    # ~75M non-emb params, ~1.5B tokens (~20 tok/param = Chinchilla).
    # A100-class, the better part of a day. Really wants FineWeb, not TinyStories.
    "base":  dict(d_model=768, n_layer=12, n_head=12, n_kv_head=4,
                  seq_len=1024, vocab_size=32_000, batch_size=24, grad_accum=8,
                  max_steps=7_700, lr=4e-4, n_docs=2_000_000, out_dir="runs/base"),
}

PRESET = "small"                                   # <-- the one knob
cfg = Config(**PRESETS[PRESET])

torch.manual_seed(cfg.seed); np.random.seed(cfg.seed); random.seed(cfg.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(cfg.data_dir, exist_ok=True); os.makedirs(cfg.out_dir, exist_ok=True)

# bf16 needs Ampere (sm_80+). A T4 is Turing and must use fp16 + GradScaler.
# Do NOT use torch.cuda.is_bf16_supported(): by default it counts *emulated*
# bf16 and returns True on a T4, which then runs bf16 slowly in software.
use_bf16   = device == "cuda" and torch.cuda.get_device_capability()[0] >= 8
amp_dtype  = torch.bfloat16 if use_bf16 else torch.float16
gpu_name   = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"

TOKENS_PER_STEP = cfg.batch_size * cfg.grad_accum * cfg.seq_len
total_tokens    = TOKENS_PER_STEP * cfg.max_steps
print(json.dumps(asdict(cfg), indent=2))
print(f"\ndevice={device} ({gpu_name})  amp={amp_dtype}")
print(f"tokens/step={TOKENS_PER_STEP:,}   total tokens={total_tokens/1e6:.0f}M")
```

**Predict before you run.** Write down your guesses, then check them against the
printout:

1. How many tokens per optimizer step? (`batch_size × grad_accum × seq_len`)
2. Roughly how many parameters will the model have? Full multi-head attention is
   ~`4·d²` per layer, but GQA shrinks K and V by `n_kv_head/n_head`: with 2 of 8
   heads that is `2d² + 2d²·(2/8) = 2.5·d²`. The SwiGLU MLP is ~`3·d·(8d/3) = 8d²`
   (a little more after rounding up to `ffn_multiple`). So ~`10.5·d²·n_layer`.
3. Divide total tokens by that. Are you above or below Chinchilla's ~20?

**Verify.** `n_head % n_kv_head == 0`. Then compare your three guesses. For
`small` you should land near 49k tokens/step, ~22M non-embedding parameters, and
~13 tokens/parameter — i.e. **deliberately below Chinchilla**, trading some final
quality for a run that finishes in an afternoon. `base` is set at ~20. Knowing
which side of that line you are on is the point of the exercise; there is no
single correct value ([Part 8.3](#83-why-most-modern-models-train-past-chinchilla-optimal)).

**Breaks like this.** Silent CPU fallback (check `device`). Assuming bf16 exists
— on a T4 it does not, which is why `amp_dtype` is detected rather than
hardcoded, and detected from compute capability rather than
`is_bf16_supported()`, which reports emulated support as real. `seq_len` set beyond what memory allows, which won't fail until
Stage 11.

> **Reference:** [Part 5.6](#56-choosing-a-shape) for shape choices,
> [Part 7](#7-optimization) for the optimizer values.

---

### Stage 2 — Acquire raw text

**Goal.** A list of raw document strings on disk.

**What's happening.** Every pretraining corpus starts as documents, not tokens.
TinyStories is the right starting corpus because it is small, clean, and a
30M-param model can genuinely master it — you will see coherent English, which is
the motivating payoff. Swap in a FineWeb slice once the pipeline works end to end.

```python
# Cell 2 — raw text
from datasets import load_dataset

# TinyStories: clean, tiny, learnable at 30M params.
ds = load_dataset(cfg.dataset, split="train", streaming=True)
docs = []
for i, ex in enumerate(ds):
    if i >= cfg.n_docs: break
    docs.append(ex["text"])

# --- swap for real web data once the pipeline runs: ---
# ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT",
#                   split="train", streaming=True)
# docs = [ex["text"] for i, ex in zip(range(cfg.n_docs), ds)]

print(f"{len(docs):,} docs, {sum(len(d) for d in docs)/1e6:.1f}M chars")
print("-" * 60); print(docs[0][:600])
```

**Verify.** Print three full documents and read them. You are looking for
encoding mojibake, HTML remnants, and truncation. Ten seconds here saves a
retrain.

**Breaks like this.** Streaming datasets silently yielding fewer docs than asked;
a field name that isn't `"text"` for your chosen dataset.

---

### Stage 3 — Inspect and filter

**Goal.** A filtered document list, plus a record of what each rule removed.

**What's happening.** Raw web text is mostly junk: navigation bars, keyword spam,
truncated boilerplate. Heuristic filters are cheap and interpretable, so they run
first. The important discipline is measuring what each rule *removes*, because
over-filtering quietly destroys diversity ([Part 3.2](#32-filtering)).

```python
# Cell 3 — heuristic quality filters (a small Gopher/C4-style subset)
import re
from collections import Counter

STOPWORDS = {"the","be","to","of","and","that","have","with","this","it","is","in"}
TERMINAL  = (".", "!", "?", '"', "'")

def quality_signals(doc):
    words = doc.split()
    n = len(words)
    lines = [l for l in doc.split("\n") if l.strip()]
    return {
        "n_words":        n,
        "mean_word_len":  (sum(map(len, words)) / n) if n else 0.0,
        "symbol_ratio":   (doc.count("#") + doc.count("...")) / max(n, 1),
        "frac_terminal":  (sum(l.rstrip().endswith(TERMINAL) for l in lines)
                           / max(len(lines), 1)),
        "n_stopwords":    sum(w.lower() in STOPWORDS for w in words),
        "frac_dup_lines": 1 - (len(set(lines)) / max(len(lines), 1)),
    }

RULES = {   # name -> predicate that is True when the doc should be DROPPED
    "too_short":      lambda s: s["n_words"] < 30,
    "too_long":       lambda s: s["n_words"] > 100_000,
    "odd_word_len":   lambda s: not (3.0 <= s["mean_word_len"] <= 10.0),
    "symbol_spam":    lambda s: s["symbol_ratio"] > 0.10,
    "no_punctuation": lambda s: s["frac_terminal"] < 0.60,
    "few_stopwords":  lambda s: s["n_stopwords"] < 2,
    "repetitive":     lambda s: s["frac_dup_lines"] > 0.30,
}

kept, removed, counts = [], [], Counter()
for d in docs:
    s = quality_signals(d)
    fired = [name for name, rule in RULES.items() if rule(s)]
    if fired:
        counts.update(fired); removed.append((fired[0], d))
    else:
        kept.append(d)

print(f"kept {len(kept):,} / {len(docs):,}  ({100*len(kept)/len(docs):.1f}%)")
for name, c in counts.most_common():
    print(f"  {name:<16} {c:>7,}  ({100*c/len(docs):.2f}%)")

# THE IMPORTANT PART: read what you threw away.
for name, d in random.sample(removed, min(5, len(removed))):
    print(f"\n--- dropped by {name} ---\n{d[:300]}")
```

**Verify.** Keep rate should be 50–90% on clean data, 10–40% on raw web. Then
read five rejected documents. If any look like text you'd want the model to
learn, loosen that rule.

**Breaks like this.** A single over-aggressive rule eating most of the corpus —
which the per-rule counter makes obvious, and an aggregate keep-rate would hide.

---

### Stage 4 — Deduplicate

**Goal.** Near-duplicate documents removed.

**What's happening.** Duplicated text causes memorization and wastes compute.
MinHash estimates Jaccard similarity between documents cheaply: shingle each doc
into n-grams, hash them, keep the minimum hash per permutation, then band the
signature so similar docs collide in a bucket ([Part 3.3](#33-deduplication)).

```python
# Cell 4 — MinHash LSH near-duplicate removal
import hashlib

NUM_PERM, N_GRAM, BANDS = 128, 5, 16
ROWS      = NUM_PERM // BANDS
THRESHOLD = (1 / BANDS) ** (1 / ROWS)      # the S-curve midpoint: ~0.71 here
JACCARD_MIN = 0.7                          # verification cutoff (see below)

rng = np.random.default_rng(cfg.seed)
# Keep a,h < 2**31 so a*h + b stays inside uint64 and the modulus is real
# arithmetic rather than silent wraparound.
A = rng.integers(1, 1 << 31, NUM_PERM, dtype=np.uint64)
B = rng.integers(0, 1 << 31, NUM_PERM, dtype=np.uint64)
MERSENNE = np.uint64((1 << 61) - 1)

def signature(doc):
    toks = doc.lower().split()
    if len(toks) < N_GRAM:
        shingles = {" ".join(toks)}
    else:
        shingles = {" ".join(toks[i:i+N_GRAM]) for i in range(len(toks)-N_GRAM+1)}
    # blake2b, not hash(): Python's hash() is salted per process, so a notebook
    # restart would silently give different results.
    h = np.array([int.from_bytes(hashlib.blake2b(s.encode(), digest_size=4).digest(),
                                 "big") for s in shingles], dtype=np.uint64)
    return ((A[:, None] * h[None, :] + B[:, None]) % MERSENNE).min(axis=1)

sigs = [signature(d) for d in kept]

buckets, dup_of = {}, {}
for i, sig in enumerate(sigs):
    for b in range(BANDS):
        key = (b, sig[b*ROWS:(b+1)*ROWS].tobytes())
        cand = buckets.get(key)
        if cand is not None:
            # A band collision is a CANDIDATE, not a duplicate. Verify with the
            # signature Jaccard estimate before discarding a document.
            est = float((sigs[i] == sigs[cand]).mean())
            if est >= JACCARD_MIN:
                dup_of[i] = (cand, est); break
        else:
            buckets[key] = i

deduped = [d for i, d in enumerate(kept) if i not in dup_of]
print(f"threshold≈{THRESHOLD:.2f} (verified at ≥{JACCARD_MIN})")
print(f"removed {len(dup_of):,} near-duplicates -> {len(deduped):,} docs")

if dup_of:                       # always eyeball a matched pair
    i, (j, est) = next(iter(dup_of.items()))
    print(f"\nestimated Jaccard {est:.3f}")
    print(f"DUP:\n{kept[i][:200]}\n\nORIGINAL:\n{kept[j][:200]}")
```

**Predict before you run.** With `BANDS=16` and `ROWS=8`, what similarity does
this catch about half the time? The LSH S-curve midpoint is `(1/BANDS)^(1/ROWS)`
— compute it before looking. (Raising `BANDS` at fixed `NUM_PERM` *lowers* the
threshold, catching more but with more false positives.)

**Verify.** Print a matched pair and confirm they really are near-duplicates. On
TinyStories expect a few percent; on raw Common Crawl, 30–60% is normal.

**Breaks like this.** Treating a band collision as a confirmed duplicate — LSH
gives you *candidates*, and skipping the verification step deletes unrelated
documents. Silent integer overflow: drawing `A` up to 2⁶¹ (as `datasketch` does)
with 32-bit hashes makes `A*h` exceed uint64 and wrap before the modulus applies,
so "mod p" is not what runs. That is why this cell draws `A` and `B` below 2³¹.
It is mostly harmless in practice, but it means the code is not doing what it
claims.

---

### Stage 5 — Train the tokenizer

**Goal.** A BPE tokenizer saved to disk.

**What's happening.** BPE starts from bytes and repeatedly merges the most
frequent adjacent pair until it hits the vocab size. Train it on *your* corpus —
a mismatched tokenizer taxes every token you will ever process
([Part 4](#4-tokenization)).

```python
# Cell 5 — BPE tokenizer
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

tok = Tokenizer(models.BPE(unk_token=None))
tok.pre_tokenizer = pre_tokenizers.Sequence([
    # Split digits out individually: measurably improves arithmetic [Part 4].
    pre_tokenizers.Digits(individual_digits=True),
    # ByteLevel applies the GPT-2 word/punctuation regex and maps bytes -> chars,
    # which guarantees no input is ever out-of-vocabulary.
    pre_tokenizers.ByteLevel(add_prefix_space=False),
])
tok.decoder = decoders.ByteLevel()

SPECIALS = ["<|endoftext|>"] + [f"<|reserved_{i}|>" for i in range(8)]
trainer = trainers.BpeTrainer(vocab_size=cfg.vocab_size,
                              special_tokens=SPECIALS,
                              initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                              show_progress=True)
tok.train_from_iterator(deduped, trainer=trainer, length=len(deduped))
tok.save(f"{cfg.data_dir}/tokenizer.json")

EOT = tok.token_to_id("<|endoftext|>")
assert tok.get_vocab_size() <= 65535, "vocab > uint16; use uint32 in Stage 6"

# fertility = tokens per word: the number that decides your effective compute
sample = deduped[:2000]
n_tok = sum(len(tok.encode(d).ids) for d in sample)
n_word = sum(len(d.split()) for d in sample)
print(f"vocab={tok.get_vocab_size()}  fertility={n_tok/n_word:.3f} tok/word")
print(tok.encode("The year 1987 cost $4.50.").tokens)
```

**Verify.** Round-trip: `tok.decode(tok.encode(s).ids) == s` for several
documents including punctuation and unicode. Fertility should be ~1.2–1.5 for
English. Look at the printed token list — digits should be split sensibly.

**Breaks like this.** Forgetting to reserve special-token slots (adding them
later means resizing embeddings); training the tokenizer on unfiltered data so
merges are spent on junk.

---

### Stage 6 — Tokenize to shards

**Goal.** `train.bin` and `val.bin` — flat `uint16` token arrays.

**What's happening.** Training reads tokens millions of times, so you tokenize
once, up front, into a flat binary file that can be memory-mapped. Documents are
concatenated with an end-of-text token between them ([Part 3.6](#36-practical-shape-of-the-data-pipeline)).

```python
# Cell 6 — tokenize to flat binary
from tqdm.auto import tqdm

split = int(0.995 * len(deduped))
# Keep these named: EVERY later stage must draw from train_docs, never val_docs.
train_docs, val_docs = deduped[:split], deduped[split:]
splits = {"train": train_docs, "val": val_docs}

CHUNK = 10_000
for name, docs_split in splits.items():
    path = f"{cfg.data_dir}/{name}.bin"
    ids_all, n_content = [], 0
    # Chunk the encoding so the progress bar tracks the slow part. Wrapping
    # tqdm around a finished encode_batch() would measure nothing.
    for s in tqdm(range(0, len(docs_split), CHUNK), desc=name):
        for enc in tok.encode_batch(docs_split[s:s+CHUNK]):
            ids_all.append(np.array(enc.ids + [EOT], dtype=np.uint16))
            n_content += len(enc.ids)          # content tokens, excluding EOT
    arr = np.concatenate(ids_all)
    arr.tofile(path)
    n_bytes = sum(len(d.encode("utf-8")) for d in docs_split)
    json.dump({"n_tokens": int(arr.size),      # includes one EOT per document
               "n_content_tokens": int(n_content),   # use THIS for BPB
               "n_bytes": n_bytes},
              open(f"{cfg.data_dir}/{name}_meta.json", "w"))
    print(f"{name}: {arr.size/1e6:.2f}M tokens -> {path}")

meta = json.load(open(f"{cfg.data_dir}/train_meta.json"))
epochs = TOKENS_PER_STEP * cfg.max_steps / meta["n_tokens"]
print(f"\nthis run will make {epochs:.2f} passes over the training data")
```

**Verify.** Decode the first 200 tokens of `train.bin` and confirm it reads as
text with `<|endoftext|>` at document boundaries. Check the epoch count: more
than ~4 passes gives diminishing returns ([Part 3.5](#35-mixing)) — get more data
or fewer steps.

**Breaks like this.** `uint16` overflow with a vocab above 65535 (the assert in
Stage 5 catches it); forgetting the separator token, so the model learns to run
documents together.

---

### Stage 7 — The dataloader

**Goal.** `get_batch(step)` returning `(x, y)` — deterministic and resumable.

**What's happening.** A batch is `B` random windows of length `T+1` into the token
array; inputs are all but the last token, targets are all but the first. Keying
the RNG on the global step makes the whole data order a pure function of
`(seed, step)`, which means resuming a crashed run needs only the step number
([Part 9.6](#96-fault-tolerance)).

```python
# Cell 7 — memmap dataloader
class Loader:
    def __init__(self, path, batch_size, seq_len, seed=0):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.B, self.T, self.seed = batch_size, seq_len, seed
        self.n_start = len(self.data) - seq_len - 1
        assert self.n_start > 0, "corpus shorter than one sequence"

    def get_batch(self, step, device):
        # RNG keyed on step => data order is a pure function of (seed, step)
        g = np.random.default_rng((self.seed, step))
        ix = g.integers(0, self.n_start, size=self.B)
        x = np.stack([self.data[i:i+self.T]       for i in ix]).astype(np.int64)
        y = np.stack([self.data[i+1:i+1+self.T]   for i in ix]).astype(np.int64)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        if device == "cuda":
            return (x.pin_memory().to(device, non_blocking=True),
                    y.pin_memory().to(device, non_blocking=True))
        return x.to(device), y.to(device)

train_loader = Loader(f"{cfg.data_dir}/train.bin", cfg.batch_size, cfg.seq_len, cfg.seed)
val_loader   = Loader(f"{cfg.data_dir}/val.bin",   cfg.batch_size, cfg.seq_len, cfg.seed + 1)

x, y = train_loader.get_batch(0, device)
print(x.shape, y.shape, x.dtype)
print("x:", tok.decode(x[0, :40].tolist()))
print("y:", tok.decode(y[0, :40].tolist()))   # must be x shifted by exactly one
assert torch.equal(x[0, 1:], y[0, :-1]), "off-by-one in the targets"

a, _ = train_loader.get_batch(5, device)
b, _ = train_loader.get_batch(5, device)
assert torch.equal(a, b), "loader is not deterministic -> not resumable"
```

**Verify.** The `y` decode must be `x` shifted by one token — the assert makes
this mechanical. This is the single most common silent bug in pretraining: an
off-by-one here still trains, just to a permanently worse loss, because the model
can still fit unigram statistics ([Part 10](#10-stability-when-training-breaks)).

**Breaks like this.** Sampling with replacement means occasional repeats — fine
at scale, worth knowing. Windows cross document boundaries; real runs often add
document masking, which is a good extension exercise.

---

### Stage 8 — Build the model

**Goal.** A modern decoder-only transformer: RMSNorm, RoPE, GQA, SwiGLU, QK-norm.

**What's happening.** This is a representative modern stack, not the 2017 one
([Part 5](#5-architecture)). Pre-norm keeps a clean residual path; RoPE encodes
relative position by rotation; GQA shares K/V heads to shrink the inference
cache; SwiGLU gates the MLP; QK-norm bounds attention logits for stability.

```python
# Cell 8 — the model
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__(); self.eps = eps; self.g = nn.Parameter(torch.ones(d))
    def forward(self, x):
        dt = x.dtype; x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.g.float()).to(dt)

def build_rope(head_dim, max_len, theta, device):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    freqs = torch.outer(torch.arange(max_len, device=device).float(), inv)
    return torch.cos(freqs), torch.sin(freqs)          # each (max_len, hd/2)

def apply_rope(x, cos, sin):
    # x: (B, n_head, T, hd); rotate-half convention (pairs i with i+hd/2)
    T = x.size(-2)
    cos, sin = cos[:T][None, None], sin[:T][None, None]
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat([x1*cos - x2*sin, x1*sin + x2*cos], -1).type_as(x)

class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        assert c.n_head % c.n_kv_head == 0
        self.nh, self.nkv = c.n_head, c.n_kv_head
        self.hd = c.d_model // c.n_head
        self.rep = self.nh // self.nkv
        self.wq = nn.Linear(c.d_model, self.nh  * self.hd, bias=False)
        self.wk = nn.Linear(c.d_model, self.nkv * self.hd, bias=False)
        self.wv = nn.Linear(c.d_model, self.nkv * self.hd, bias=False)
        self.wo = nn.Linear(self.nh * self.hd, c.d_model, bias=False)
        self.q_norm, self.k_norm = RMSNorm(self.hd), RMSNorm(self.hd)   # QK-norm

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.nh,  self.hd).transpose(1, 2)
        k = self.wk(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
        v = self.wv(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)
        # RoPE before the repeat: same result either way (it acts per-head with
        # shared cos/sin), but this rotates nkv heads instead of nh of them.
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        k = k.repeat_interleave(self.rep, dim=1)      # GQA: broadcast KV heads
        v = v.repeat_interleave(self.rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # FlashAttention
        return self.wo(y.transpose(1, 2).reshape(B, T, C))

class MLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        h = int(8 * c.d_model / 3)                      # SwiGLU param-matching
        h = c.ffn_multiple * math.ceil(h / c.ffn_multiple)
        self.w_gate = nn.Linear(c.d_model, h, bias=False)
        self.w_up   = nn.Linear(c.d_model, h, bias=False)
        self.w_down = nn.Linear(h, c.d_model, bias=False)
    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))

class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n1, self.attn = RMSNorm(c.d_model), Attention(c)
        self.n2, self.mlp  = RMSNorm(c.d_model), MLP(c)
    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin)       # pre-norm residual
        return x + self.mlp(self.n2(x))

class GPT(nn.Module):
    def __init__(self, c):
        super().__init__(); self.cfg = c
        self.embed  = nn.Embedding(c.vocab_size, c.d_model)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.n_layer)])
        self.norm_f = RMSNorm(c.d_model)
        self.head   = nn.Linear(c.d_model, c.vocab_size, bias=False)
        if c.tie_embeddings:
            self.head.weight = self.embed.weight
        cos, sin = build_rope(c.d_model // c.n_head, c.seq_len, c.rope_theta, "cpu")
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.apply(self._init)
        # scale residual-output projections by 1/sqrt(2*n_layer)  [Part 7.4]
        for n, p in self.named_parameters():
            if n.endswith(("wo.weight", "w_down.weight")):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * c.n_layer))

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if getattr(m, "bias", None) is not None: nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        x = self.embed(idx)
        for b in self.blocks:
            x = b(x, self.cos, self.sin)
        x = self.norm_f(x)
        logits = self.head(x)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(),
                               targets.reshape(-1))     # CE in fp32
        return logits, loss

    def n_params(self, non_embedding=True):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding: n -= self.embed.weight.numel()
        return n

model = GPT(cfg).to(device)
N = model.n_params()
print(f"total {sum(p.numel() for p in model.parameters())/1e6:.2f}M  |  "
      f"non-embedding {N/1e6:.2f}M")
print(f"Chinchilla-optimal tokens ≈ {20*N/1e6:.0f}M   "
      f"(this run: {TOKENS_PER_STEP*cfg.max_steps/1e6:.0f}M)")
```

**Verify.** Parameter count matches the preset's advertised size. A forward pass
on one batch returns `logits.shape == (B, T, vocab_size)`. Print the
Chinchilla comparison and understand which side of it you're on.

**Breaks like this.** `d_model` not divisible by `n_head`. Forgetting the fp32
cast in cross-entropy. Using `chunk`-style rotate-half `cos`/`sin` tables with
interleaved-pair RoPE code (or vice versa) — the two conventions are both
correct but not interchangeable, and mixing them trains to a worse loss without
ever erroring.

---

### Stage 9 — Pre-flight sanity checks

**Goal.** Prove the model can learn *before* spending an hour finding out it can't.

**What's happening.** Three cheap tests catch the overwhelming majority of
implementation bugs. This stage has the best time-saved-per-line ratio in the
entire notebook and is the one most often skipped.

```python
# Cell 9 — pre-flight checks
# 1. Initial loss must be ≈ ln(vocab_size): a fresh model is uniform.
x, y = train_loader.get_batch(0, device)
with torch.no_grad():
    _, loss0 = model(x, y)
expected = math.log(cfg.vocab_size)
print(f"init loss {loss0.item():.4f}  expected ≈ {expected:.4f}")
assert abs(loss0.item() - expected) < 0.7, "bad init or broken forward pass"

# 2. Causality: changing a FUTURE token must not change an EARLIER prediction.
xa = x[:1].clone(); xb = xa.clone(); xb[0, -1] = (xb[0, -1] + 1) % cfg.vocab_size
with torch.no_grad():
    la, _ = model(xa); lb, _ = model(xb)
drift = (la[0, :-1] - lb[0, :-1]).abs().max().item()
print(f"causality drift {drift:.2e}")
assert drift < 1e-4, "information is leaking from the future"

# 3. Overfit a single batch: a correct model drives one batch to ~0 loss.
probe = GPT(cfg).to(device)
opt = torch.optim.AdamW(probe.parameters(), lr=3e-4)
for i in range(200):
    _, l = probe(x, y); opt.zero_grad(); l.backward(); opt.step()
    if i % 50 == 0: print(f"  overfit step {i:3d}  loss {l.item():.4f}")
print(f"final overfit loss {l.item():.4f}")
assert l.item() < 0.5, "cannot memorize one batch -> real bug, do not proceed"
del probe, opt; torch.cuda.empty_cache() if device == "cuda" else None
```

**Verify.** All three asserts pass. Test 1 catches init and forward bugs, test 2
catches masking bugs, test 3 catches gradient-flow bugs. If test 3 fails, nothing
downstream matters — debug here.

**Breaks like this.** An init loss *well above* `ln V` means the logits are too
large at initialization — init std too big, or a missing final norm. Note what it
does **not** mean: at initialization the output is near-uniform regardless of
what the targets are, so misaligned labels cannot move the initial loss. Test 1
is blind to off-by-one bugs; that is what the Stage 7 assert is for. Overfit
stalling near `ln V` means no gradient is reaching the parameters at all.

> **Reference:** [Part 10](#10-stability-when-training-breaks) — Checkpoint 10
> asks you to induce these failures on purpose so you recognize their shapes.

---

### Stage 10 — Optimizer and schedule

**Goal.** AdamW with correct parameter groups, plus a WSD learning-rate schedule.

**What's happening.** Weight decay belongs on matrices but not on norms and
biases. Whether it belongs on *embeddings* is genuinely unsettled — the common
`dim >= 2` rule decays them, and nanoGPT does exactly that. Here they are
excluded explicitly, so the code and the comment agree; see
[Part 7.1](#71-the-optimizer). The WSD schedule warms up, holds flat, then
decays over the final stretch, which lets you stop at any point and makes the
decay phase a natural place to switch to higher-quality data
([Part 7.2](#72-learning-rate-schedule)).

```python
# Cell 10 — optimizer + schedule
def make_optimizer(model, c):
    raw = getattr(model, "_orig_mod", model)
    # Three groups, stated explicitly rather than inferred from p.dim():
    #   matrices -> decay | norms & biases -> no decay | embeddings -> choose
    # NOTE: embed.weight is 2-D, so a bare `p.dim() >= 2` rule WOULD decay it.
    emb_ids = {id(raw.embed.weight)}
    decay, nodecay, embeds = [], [], []
    for n, p in raw.named_parameters():
        if not p.requires_grad:     continue
        if id(p) in emb_ids:        embeds.append(p)
        elif p.dim() >= 2:          decay.append(p)
        else:                       nodecay.append(p)
    tot = lambda ps: sum(p.numel() for p in ps)
    print(f"decay   : {len(decay):3d} tensors, {tot(decay)/1e6:6.2f}M")
    print(f"no-decay: {len(nodecay):3d} tensors, {tot(nodecay)/1e3:6.1f}K")
    print(f"embed   : {len(embeds):3d} tensors, {tot(embeds)/1e6:6.2f}M  (wd=0 here)")
    return torch.optim.AdamW(
        [{"params": decay,   "weight_decay": c.weight_decay},
         {"params": nodecay, "weight_decay": 0.0},
         {"params": embeds,  "weight_decay": 0.0}],
        lr=c.lr, betas=(c.beta1, c.beta2), eps=1e-8,
        fused=(device == "cuda"))

def lr_at(step, c):
    """Warmup -> Stable -> Decay. Clamped: never returns a negative LR."""
    step = min(step, c.max_steps)                     # <- clamp past the end
    warm  = max(int(c.warmup_frac * c.max_steps), 1)
    decay = max(int(c.decay_frac  * c.max_steps), 1)
    stable_end = c.max_steps - decay
    lo = c.lr * c.min_lr_frac
    if step < warm:        return c.lr * (step + 1) / warm
    if step < stable_end:  return c.lr
    prog = min((step - stable_end) / decay, 1.0)
    return lo + (c.lr - lo) * (1 - prog)              # linear decay tail

optimizer = make_optimizer(model, cfg)
if cfg.compile and device == "cuda":
    model = torch.compile(model)

assert lr_at(cfg.max_steps + 5_000, cfg) >= 0, "schedule goes negative past the end"

import matplotlib.pyplot as plt
plt.plot([lr_at(s, cfg) for s in range(int(cfg.max_steps * 1.2))])
plt.axvline(cfg.max_steps, ls=":", c="k")
plt.xlabel("step"); plt.ylabel("lr"); plt.title("WSD schedule"); plt.show()
```

**Predict before you run.** How many tensors end up in the no-decay group? Count
the RMSNorm gains: each block has four (`n1`, `n2`, and the attention's `q_norm`
and `k_norm`), plus one final norm — so `4·n_layer + 1`, which is 33 for `small`.
Work it out from the Stage 8 code before running; being wrong means your mental
model of the module tree is off.

**Verify.** The printed groups match your prediction, and embeddings appear on
their own line rather than hidden in `decay`. The plot flattens at `min_lr` past
`max_steps` instead of continuing downward.

**Breaks like this.** A `p.dim() >= 2` rule under a comment claiming embeddings
are excluded — they are not, and with tied weights that silently decays your LM
head too. An unclamped schedule: the linear tail keeps falling past `max_steps`
and eventually returns negative learning rates to anything that trains a few
steps longer than planned.

---

### Stage 11 — The training loop

**Goal.** A trained model, plus the instrumentation that tells you whether the
run is healthy.

**What's happening.** The loop is: sample → forward → scaled backward ×
`grad_accum` → clip → step → log. What separates a real run from a toy one is
everything around it — grad norm, MFU, per-domain eval loss — because these are
what let you diagnose a run instead of guessing ([Part 9.5](#95-mfu--the-number-that-matters),
[Part 10](#10-stability-when-training-breaks)).

```python
# Cell 11 — training loop with instrumentation
autocast = (torch.autocast("cuda", dtype=amp_dtype) if device == "cuda"
            else torch.autocast("cpu", enabled=False))
# fp16 generally needs dynamic loss scaling to avoid gradient underflow; bf16
# normally does not. Enabled only on pre-Ampere GPUs.
scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda" and not use_bf16))

# Dense bf16/fp16 peak, no sparsity. Add your card if it's missing.
# Matching is by substring, so ORDER MATTERS: specific names before the names
# they contain ("L40S" before "L4", "H100 PCIe" before "H100", "A100" before "A10").
PEAK = [("H100 PCIe", 756e12), ("H100", 989e12), ("A100", 312e12),
        ("L40S", 362e12), ("L40", 181e12), ("L4", 121e12), ("A10", 125e12),
        ("T4", 65e12), ("V100", 125e12), ("4090", 165e12), ("3090", 71e12)]
PEAK_FLOPS = next((v for k, v in PEAK if k in gpu_name), 100e12)
print(f"assuming {PEAK_FLOPS/1e12:.0f} TFLOP/s peak for '{gpu_name}'")

def flops_per_token(c, n_nonemb):
    """6N undercounts at this scale: the LM head is ~36% of N for `small`,
    and attention is not a parameter matmul at all. [Part 9.5]"""
    head = c.vocab_size * c.d_model                      # tied head still costs FLOPs
    attn = 12 * c.n_layer * c.d_model * c.seq_len        # QK^T and AV, fwd+bwd
    return 6 * (n_nonemb + head) + attn

FLOPS_PER_TOKEN = flops_per_token(cfg, N)   # fuller estimate: head + attention
FLOPS_6N        = 6 * N                      # bare 6N: comparable to most papers

@torch.no_grad()
def estimate_loss(net, loader, n_batches=20):
    was_training = net.training
    net.eval(); tot = 0.0
    for i in range(n_batches):
        # Fixed, deterministic batches -- NOT disjoint: offsets are sampled with
        # replacement, so windows can overlap. Fine for a noisy estimate.
        xb, yb = loader.get_batch(10**6 + i, device)
        with autocast: _, l = net(xb, yb)
        tot += l.item()
    net.train(was_training); return tot / n_batches

hist = {"step": [], "loss": [], "val": [], "lr": [], "gnorm": [], "mfu": []}
start_step = 0

# --- resume, if a checkpoint is already on disk ---
resume_path = f"{cfg.out_dir}/last.pt"
if os.path.exists(resume_path):
    model, optimizer, start_step, cfg, hist = load_ckpt(resume_path, device)  # Stage 12
    print(f"resumed from step {start_step}")

STABLE_END = cfg.max_steps - int(cfg.decay_frac * cfg.max_steps)
model.train(); t0 = time.time(); last_log = start_step

for step in range(start_step, cfg.max_steps):
    lr = lr_at(step, cfg)
    for g in optimizer.param_groups: g["lr"] = lr

    loss_accum = torch.zeros((), device=device)         # stays on GPU: no sync
    for micro in range(cfg.grad_accum):
        xb, yb = train_loader.get_batch(step * cfg.grad_accum + micro, device)
        with autocast:
            _, loss = model(xb, yb)
        scaler.scale(loss / cfg.grad_accum).backward()
        loss_accum += loss.detach() / cfg.grad_accum    # .detach(), not .item()

    scaler.unscale_(optimizer)                          # unscale before clipping
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    scaler.step(optimizer); scaler.update()
    optimizer.zero_grad(set_to_none=True)

    if step % 10 == 0:                                  # the only per-step sync
        if device == "cuda": torch.cuda.synchronize()
        dt = time.time() - t0; t0 = time.time()
        n_steps = max(step - last_log, 1); last_log = step
        tok_s = TOKENS_PER_STEP * n_steps / dt
        mfu    = FLOPS_PER_TOKEN * tok_s / PEAK_FLOPS   # fuller estimate [Part 9.5]
        mfu_6n = FLOPS_6N        * tok_s / PEAK_FLOPS   # bare 6N, for comparisons
        l = loss_accum.item()
        hist["step"].append(step); hist["loss"].append(l)
        hist["lr"].append(lr);     hist["gnorm"].append(gnorm.item())
        hist["mfu"].append(mfu);   hist["val"].append(None)
        print(f"step {step:5d} | loss {l:.4f} | lr {lr:.2e} | "
              f"gnorm {gnorm.item():5.2f} | {tok_s/1e3:6.1f}k tok/s | "
              f"mfu {mfu:5.1%} (6N {mfu_6n:5.1%})")

    if step % cfg.eval_every == 0 and step > start_step:
        v = estimate_loss(model, val_loader)
        hist["val"][-1] = v
        print(f"  >>> step {step}: val loss {v:.4f}  ppl {math.exp(v):.2f}")

    # Checkpoints store the NEXT step to run (step + 1). This step's update is
    # already applied; storing `step` would replay its batch on resume.

    # --- periodic checkpoint, so a crash costs at most ckpt_every steps ---
    if (step + 1) % cfg.ckpt_every == 0:
        save_ckpt(model, optimizer, step + 1, cfg, hist, resume_path)   # Stage 12

    # --- snapshot the end of the STABLE phase: Stage 14 branches from here ---
    if step + 1 == STABLE_END:      # last stable step done; next one starts decay
        save_ckpt(model, optimizer, step + 1, cfg, hist, f"{cfg.out_dir}/stable_end.pt")
        print(f"  >>> saved stable-phase checkpoint (decay starts at step {step + 1})")

save_ckpt(model, optimizer, cfg.max_steps, cfg, hist, f"{cfg.out_dir}/final.pt")
```

> **Cell ordering note.** This cell calls `save_ckpt` / `load_ckpt`, which are
> defined in Stage 12. In the notebook, put the Stage 12 *function definitions*
> cell **above** this one; Stage 12's verification cell stays below.

**Predict before you run.** Before the first step prints: what loss do you expect
at step 10, and at step 100? (Step ~0 should be near `ln V`; within a hundred
steps a working model is at or below unigram entropy, roughly 5–6 nats for
English.) Also guess your MFU. Most people guess far too high on a small model —
at `d_model=512` you are launch-latency bound, and 15–25% is normal.

**Verify.** Watch these four, in this order of importance:

| Signal | Healthy | Meaning if not |
|---|---|---|
| Loss at step ~100 | below the unigram entropy (~5–6 nats) | data or label bug |
| Grad norm | settles to a stable band, spikes rare | instability brewing ([Part 10](#10-stability-when-training-breaks)) |
| MFU (fuller estimate) | 15–25% at this size; 35–55% on a large, well-shaped run | dataloader stall, no compile, bad shapes |
| Val − train loss | small and stable | too few tokens per parameter |

**Breaks like this.** Calling `.item()` on every micro-step forces a GPU sync and
quietly costs throughput — accumulate `loss.detach()` and sync only when logging.
Clipping *before* `scaler.unscale_()`, which clips the scaled gradients and so
applies an arbitrary threshold. Hardcoding an H100's peak FLOPs and then
reporting an MFU that is 15× too low on a T4.

> **On forgetting `/ cfg.grad_accum`:** it does *not* inflate your effective
> learning rate under Adam — see the note in [Part 7.3](#73-batch-size). It
> inflates the gradient norm, so clipping fires every step and you lose it as a
> diagnostic.

---

### Stage 12 — Checkpoint and resume

**Goal.** A run you can kill and restart without losing progress or repeating data.

**What's happening.** Real runs get interrupted. A checkpoint must carry model,
optimizer, config, *and the step number* — because the step is what reconstructs
the data order in Stage 7. Forgetting the data position silently re-trains on the
same tokens ([Part 9.6](#96-fault-tolerance)).

**This is two cells.** `12a` defines the functions and must run **before**
Stage 11, which calls them. `12b` verifies the round trip and runs after training.

```python
# Cell 12a — checkpoint helpers  (PLACE THIS ABOVE STAGE 11)
# Both functions use the global `scaler`, which Stage 11 creates before it first
# calls either one.
def save_ckpt(model, optimizer, step, cfg, hist, path):
    raw = getattr(model, "_orig_mod", model)        # unwrap torch.compile
    tmp = path + ".tmp"
    torch.save({"model": raw.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,              # NEXT step to run -> reconstructs data order
                "cfg": asdict(cfg),
                "hist": hist,
                "scaler": scaler.state_dict(),       # fp16 loss scale ({} if bf16)
                # Both RNGs: harmless today (no dropout; the loader is keyed on
                # step), required for exact resumes once you add dropout.
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": (torch.cuda.get_rng_state_all()
                             if torch.cuda.is_available() else None)}, tmp)
    os.replace(tmp, path)     # atomic: a crash mid-write can't corrupt the file
    print(f"saved {path} @ step {step}")

def load_ckpt(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    c = Config(**ck["cfg"])
    m = GPT(c).to(device); m.load_state_dict(ck["model"])
    o = make_optimizer(m, c); o.load_state_dict(ck["optimizer"])
    if ck.get("scaler"):             # else fp16 restarts at 65536 and re-backs-off
        scaler.load_state_dict(ck["scaler"])
    torch.set_rng_state(ck["torch_rng"].cpu())
    if ck.get("cuda_rng") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in ck["cuda_rng"]])
    if c.compile and device == "cuda":
        m = torch.compile(m)
    return m, o, ck["step"], c, ck["hist"]
```

```python
# Cell 12b — prove the round trip actually works
m2, o2, s2, c2, h2 = load_ckpt(f"{cfg.out_dir}/final.pt", device)
xb, yb = val_loader.get_batch(0, device)
with torch.no_grad():
    _, l1 = model(xb, yb)
    _, l2 = m2(xb, yb)
print(f"reloaded {l2.item():.6f} vs original {l1.item():.6f}  (step {s2})")
assert abs(l1.item() - l2.item()) < 1e-3, "checkpoint round-trip is broken"

# The optimizer state matters as much as the weights: confirm Adam's moments
# survived, or resuming will visibly bump the loss.
st = next(iter(o2.state.values()))
assert "exp_avg" in st and st["exp_avg"].abs().sum() > 0, "Adam moments lost"
print("optimizer moments restored OK")
del m2, o2
```

**Verify.** Both asserts pass. Then do the real test: **restart the kernel** and
resume *without* redoing the data pipeline. Stages 2–6 are 20–60 minutes of
download, filtering, dedup and tokenizer training — and their only outputs that
training needs are already on disk. Run cell 1, then this cell in place of 2–6:

```python
# Cell 12c — fast resume after a kernel restart (replaces cells 2-6)
from tokenizers import Tokenizer
need = ["tokenizer.json", "train.bin", "val.bin", "train_meta.json", "val_meta.json"]
missing = [f for f in need if not os.path.exists(f"{cfg.data_dir}/{f}")]
assert not missing, f"missing {missing}: run Stages 2-6 once first"
tok  = Tokenizer.from_file(f"{cfg.data_dir}/tokenizer.json")
EOT  = tok.token_to_id("<|endoftext|>")
meta = json.load(open(f"{cfg.data_dir}/train_meta.json"))
print(f"loaded tokenizer + shards; last.pt present: "
      f"{os.path.exists(f'{cfg.out_dir}/last.pt')}")
```

then cells 7, 8, 12a, 10 and 11. Stage 11 should print `resumed from step N` and
continue from that loss rather than jumping back up. Delete `last.pt` to start
fresh. (Stage 14 needs `train_docs`, `val_docs` and `quality_signals`, which
only exist in memory, so it still requires Stages 2–6 in the same session. Those
stages are deterministic —
`blake2b`, not `hash()` — so re-running them reproduces the same shards.)

**Breaks like this.** Saving `torch.compile`'s wrapper, which prefixes every key
with `_orig_mod.`. Saving the model but not the optimizer, which throws away the
Adam moments and causes a visible loss bump on resume. Storing the step you just
*finished* instead of the next one to run, so every resume replays one batch.
Forgetting the `GradScaler` state on fp16, so the loss scale restarts high and
the first steps after resume are skipped as overflows. Rebuilding the data
pipeline on resume with anything non-deterministic in it — the step-keyed
loader then reads the right offsets into the wrong tokens. Writing the checkpoint
non-atomically, so a crash during the save leaves a truncated file and you lose
the run. Saving only at the end — which is not checkpointing, it's exporting.

---

### Stage 13 — Evaluate

**Goal.** Held-out loss, bits-per-byte, and generated samples.

**What's happening.** Raw loss is only comparable between identical tokenizers;
BPB normalizes by the underlying bytes and is the honest cross-model metric
([Part 6](#6-the-objective-and-the-loss)). Generation is the qualitative check —
numbers can look fine while output is degenerate.

```python
# Cell 13 — evaluation
val_meta = json.load(open(f"{cfg.data_dir}/val_meta.json"))
val_loss = estimate_loss(model, val_loader, n_batches=50)
# Use CONTENT tokens, not n_tokens: the injected EOT separators have no
# corresponding bytes, so including them inflates BPB.
bpb = val_loss * val_meta["n_content_tokens"] / (math.log(2) * val_meta["n_bytes"])
print(f"val loss {val_loss:.4f} | ppl {math.exp(val_loss):.2f} | BPB {bpb:.4f}")

@torch.no_grad()
def generate(model, prompt, max_new=120, temp=0.8, top_k=50):
    model.eval()
    ids = torch.tensor([tok.encode(prompt).ids], device=device)
    for _ in range(max_new):
        window = ids[:, -cfg.seq_len:]
        with autocast: logits, _ = model(window)
        logits = logits[:, -1, :].float() / temp
        if top_k:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")
        nxt = torch.multinomial(F.softmax(logits, -1), 1)
        if nxt.item() == EOT: break
        ids = torch.cat([ids, nxt], dim=1)
    model.train()
    return tok.decode(ids[0].tolist())

for p in ["Once upon a time", "The little robot"]:
    print(f"\n--- {p!r} ---\n{generate(model, p)}")

fig, ax = plt.subplots(1, 3, figsize=(15, 3.5))
ax[0].plot(hist["step"], hist["loss"], lw=.8, label="train")
vs = [(s, v) for s, v in zip(hist["step"], hist["val"]) if v is not None]
ax[0].plot(*zip(*vs), "o-", label="val"); ax[0].set_yscale("log")
ax[0].set_title("loss"); ax[0].legend()
ax[1].plot(hist["step"], hist["gnorm"], lw=.8); ax[1].set_title("grad norm")
ax[2].plot(hist["step"], hist["mfu"],  lw=.8); ax[2].set_title("MFU")
for a in ax: a.set_xlabel("step")
plt.tight_layout(); plt.show()
```

**Verify.** On TinyStories at `small`, expect val loss ~1.3–1.8 and grammatical,
mostly-coherent stories. Degenerate repetition means undertrained or too low a
sampling temperature.

**Breaks like this.** Comparing raw loss across runs with different vocab sizes —
always use BPB. Evaluating without `model.eval()` if you later add dropout.

---

### Stage 14 — Anneal

**Goal.** A *controlled* measurement of whether annealing on higher-quality data
beats simply finishing the run normally.

**What's happening.** The last 10–20% of training, at decaying LR on a
higher-quality mix, moves downstream quality disproportionately
([Part 13](#13-after-pretraining)). The experimental design is the whole lesson
here. Two traps make the naive version worthless:

1. **You cannot anneal an already-annealed model.** Stage 11's WSD schedule
   already decayed to `min_lr` over the final 20%. Decaying *again* from there
   measures "more steps at a low LR", not "better data". So both arms must branch
   from the **same stable-phase checkpoint** — which is why Stage 11 saves
   `stable_end.pt`.
2. **You need a control arm.** Run the identical decay on the *original* mix.
   Without it, any improvement is unattributable.

```python
# Cell 14 — annealing, as a controlled A/B
# Quality proxy. NOTE: stopword *count* would just select the longest documents;
# the ratio is length-independent. On TinyStories this proxy is weak by design —
# the control arm is what makes the result interpretable.
def quality_score(d):
    s = quality_signals(d)
    return s["n_stopwords"] / max(s["n_words"], 1) + 0.5 * s["frac_terminal"]

DECAY_STEPS = int(cfg.decay_frac * cfg.max_steps)

# Size the anneal corpus to the decay phase. A fixed 20k docs (~4M tokens) against
# a ~59M-token decay would be ~15 epochs -- far past the ~4 in [Part 3.5] -- and
# the arm would lose to plain repetition, not to bad data.
MAX_ANNEAL_EPOCHS = 2
decay_tokens = DECAY_STEPS * TOKENS_PER_STEP
tok_per_doc  = meta["n_tokens"] / len(train_docs)
n_anneal = min(len(train_docs),
               math.ceil(decay_tokens / (MAX_ANNEAL_EPOCHS * tok_per_doc)))

# train_docs ONLY. Building this from `deduped` would include the validation
# split and contaminate every number below.
anneal_docs = sorted(train_docs, key=quality_score, reverse=True)[:n_anneal]
print(f"anneal corpus: top {n_anneal:,} of {len(train_docs):,} train docs "
      f"(~{decay_tokens / (n_anneal * tok_per_doc):.1f} epochs over the decay)")
# Compare CONTENT, not id(): an id() check is true by construction and can never
# fail. This one catches a val document that also appears in train by text.
assert not (set(anneal_docs) & set(val_docs)), "val leaked in"

np.concatenate([np.array(e.ids + [EOT], dtype=np.uint16)
                for e in tok.encode_batch(anneal_docs)]).tofile(
                f"{cfg.data_dir}/anneal.bin")
anneal_loader = Loader(f"{cfg.data_dir}/anneal.bin", cfg.batch_size, cfg.seq_len, 7)

def decay_from_stable(loader, tag):
    """Branch from the end of the stable phase and run the SAME decay on `loader`."""
    m, o, step0, c, _ = load_ckpt(f"{cfg.out_dir}/stable_end.pt", device)
    m.train()
    for s in range(DECAY_STEPS):
        lr = lr_at(step0 + s, c)                  # the real WSD decay tail
        for g in o.param_groups: g["lr"] = lr
        for micro in range(c.grad_accum):
            xb, yb = loader.get_batch(10**7 + s * c.grad_accum + micro, device)
            with autocast: _, loss = m(xb, yb)
            scaler.scale(loss / c.grad_accum).backward()
        scaler.unscale_(o)
        torch.nn.utils.clip_grad_norm_(m.parameters(), c.grad_clip)
        scaler.step(o); scaler.update(); o.zero_grad(set_to_none=True)
        if s % 100 == 0: print(f"  [{tag}] {s:4d}/{DECAY_STEPS} lr {lr:.2e}")
    v = estimate_loss(m, val_loader, 50)
    print(f"  [{tag}] val loss {v:.4f}")
    return m, v

print("control arm: decay on the ORIGINAL mix")
m_ctrl,   v_ctrl   = decay_from_stable(train_loader,  "control")
print("treatment arm: decay on the HIGH-QUALITY mix")
m_anneal, v_anneal = decay_from_stable(anneal_loader, "anneal")

print(f"\ncontrol {v_ctrl:.4f}   anneal {v_anneal:.4f}   "
      f"delta {v_ctrl - v_anneal:+.4f} nats")
print("\n--- control ---\n",  generate(m_ctrl,   "Once upon a time"))
print("\n--- anneal  ---\n",  generate(m_anneal, "Once upon a time"))
```

**Predict before you run.** Write down the delta you expect, with a sign. On
TinyStories — already clean and homogeneous — an honest prediction is "close to
zero, possibly negative", because there isn't much quality headroom to exploit
and the anneal corpus is a narrower slice of the same distribution, seen twice.
A near-zero result here is a *correct* experiment, not a failed one.

**Verify.** The two arms differ only in data. If the deltas are within run-to-run
noise, your proxy didn't separate quality — which is the common outcome on a
clean corpus and the reason this technique is demonstrated on real web data.
Re-run both arms with a different seed to estimate that noise before believing
any delta.

**Breaks like this.** Building the anneal corpus from `deduped` instead of
`train_docs`, which puts validation documents in training and makes every number
meaningless. Annealing on top of an already-decayed model, so you measure extra
low-LR steps rather than data quality. Ranking by a length-correlated statistic
(raw stopword *count*) and calling the longest documents the best ones. Using an
anneal corpus so small it is repeated many times over the decay, which measures
overfitting rather than quality. Reporting a single arm with no control.

---

### Stage 15 — Fit a scaling law

**Goal.** Predict the loss of a model you have not trained yet.

**What's happening.** This is the core professional skill in pretraining: you
de-risk expensive runs by forecasting them from cheap ones
([Part 8](#8-scaling-laws)). Doing it once at notebook scale makes every scaling
discussion afterwards concrete.

```python
# Cell 15 — mini scaling law
from scipy.optimize import curve_fit

# Each model trains to the SAME tokens-per-parameter ratio, so every point is at
# comparable convergence. Training all sizes for a fixed number of steps would
# give them equal tokens, leaving the largest the most under-trained -- which
# bends the curve and is the usual reason a notebook scaling fit "fails".
TOK_PER_PARAM = 20                     # Chinchilla-ish [Part 8.2]

SIZES = [(128, 4), (192, 6), (256, 6), (320, 8), (384, 8), (512, 8)]
train_tokens = json.load(open(f"{cfg.data_dir}/train_meta.json"))["n_tokens"]
results = []
for d, L in SIZES:
    n_head = max(4, d // 64)
    # GQA needs n_head % n_kv_head == 0. d=320 gives 5 heads, which 2 KV heads
    # cannot divide -- fall back to a single KV head (MQA) for odd head counts.
    n_kv = 2 if n_head % 2 == 0 else 1
    c = Config(**{**PRESETS[PRESET], "d_model": d, "n_layer": L,
                  "n_head": n_head, "n_kv_head": n_kv})
    m = GPT(c).to(device); n = m.n_params()
    tps = c.batch_size * c.grad_accum * c.seq_len
    c.max_steps = max(200, round(TOK_PER_PARAM * n / tps))   # compute-matched
    epochs = c.max_steps * tps / train_tokens
    if epochs > 4:     # [Part 3.5] -- repeated data, not model size, will set this loss
        print(f"  WARNING: d={d} needs {epochs:.1f} epochs; this point is "
              f"data-limited and will sit ABOVE the true curve")
    o = make_optimizer(m, c)
    if c.compile and device == "cuda": m = torch.compile(m)
    m.train()
    for step in range(c.max_steps):
        for g in o.param_groups: g["lr"] = lr_at(step, c)
        for micro in range(c.grad_accum):
            xb, yb = train_loader.get_batch(step * c.grad_accum + micro, device)
            with autocast: _, loss = m(xb, yb)
            scaler.scale(loss / c.grad_accum).backward()
        scaler.unscale_(o)
        torch.nn.utils.clip_grad_norm_(m.parameters(), c.grad_clip)
        scaler.step(o); scaler.update(); o.zero_grad(set_to_none=True)
    v = estimate_loss(m, val_loader, 30)
    results.append((n, v))
    print(f"N={n/1e6:6.2f}M  steps={c.max_steps:5d}  "
          f"tokens={c.max_steps*tps/1e6:7.1f}M  epochs={epochs:4.1f}  loss={v:.4f}")
    del m, o
    if device == "cuda": torch.cuda.empty_cache()

Ns = np.array([r[0] for r in results]); Ls = np.array([r[1] for r in results])
law = lambda N, E, A, alpha: E + A * N ** (-alpha)
# Fit on all but the largest, then predict it. 5 points / 3 parameters is still
# thin -- treat a good hit as encouraging, not as validation.
(E, A, alpha), _ = curve_fit(law, Ns[:-1], Ls[:-1], p0=[1.0, 100.0, 0.3], maxfev=20000)
pred = law(Ns[-1], E, A, alpha)
print(f"\nL(N) = {E:.3f} + {A:.1f}·N^-{alpha:.3f}   (fit on {len(Ns)-1} points)")
print(f"held-out largest: predicted {pred:.4f}  actual {Ls[-1]:.4f}  "
      f"({100*abs(pred-Ls[-1])/Ls[-1]:.1f}% error)")

plt.loglog(Ns, Ls, "o", label="measured")
grid = np.logspace(np.log10(Ns[0]), np.log10(Ns[-1]*4), 50)
plt.loglog(grid, law(grid, E, A, alpha), "--", label="fit + extrapolation")
plt.loglog([Ns[-1]], [pred], "x", ms=12, label="prediction")
plt.xlabel("non-embedding params"); plt.ylabel("val loss"); plt.legend(); plt.show()
```

**Predict before you run.** Write down your predicted loss for the largest model
*before* the fit prints it, using only the smaller points on a log-log plot by
eye. Then compare against the fit. You are calibrating your own extrapolation
instinct, which is the skill this stage exists to build.

**Verify.** The held-out prediction should land within a few percent. Check that
the printed token counts really do scale with `N` — if they are all equal, the
compute-matching broke and the fit is measuring convergence, not scale. Check the
`epochs` column too: at 20 tokens/param the 22M model needs ~450M tokens, which
is 5+ passes over the `small` corpus. If any size warns, raise `n_docs` toward
the full TinyStories train split (~2.1M stories) before running this stage —
otherwise the held-out point is exactly the one damaged by repetition.

**Budget.** Summed over all six sizes this is ~1B tokens — roughly 3–4× the main
`small` run. Plan on several hours on an A100 and most of a day on a T4.

**Breaks like this.** Training every size for the same number of steps, so larger
models are systematically under-trained and `alpha` comes out too small. Fitting
3 parameters to 5 points and treating the result as confirmed — with two degrees
of freedom, a good prediction is weak evidence. Letting the largest sizes run past
~4 epochs, so repeated data bends the curve upward at the one point you are
trying to predict. Reusing a learning rate tuned at
one width across all widths, which biases the largest models
([Part 7.5](#75-hyperparameter-transfer)).

---

### Stage 16 — Graduating from the notebook

You now have the whole pipeline. What a notebook structurally cannot teach:

| Next capability | Where it lives | Start with |
|---|---|---|
| Multi-GPU (FSDP/TP/PP) | a real cluster | `huggingface/nanotron`, `pytorch/torchtitan` |
| Trillion-token data | distributed processing | `huggingface/datatrove` |
| Fault tolerance | long runs on flaky hardware | `allenai/OLMo-core` |
| Kernel-level speed | Triton / CUDA | `linkedin/Liger-Kernel`, `karpathy/llm.c` |
| Current architecture tricks | the speedrun | `KellerJordan/modded-nanogpt` |

The natural port is: take this notebook's Stages 7–12, convert them to a script
with `torchrun` + FSDP2, swap TinyStories for a FineWeb slice, and run the `base`
preset for a day. That is a genuine, if small, pretraining run — and every
concept in Parts 1–15 will have a concrete referent by then.

---

## 17. Glossary

| Term | Meaning |
|---|---|
| **BPB** | Bits per byte. Tokenizer-independent loss measure. |
| **Chinchilla-optimal** | `D ≈ 20N`; minimizes loss for fixed training compute. |
| **Critical batch size** | Batch size past which larger batches stop reducing required steps proportionally. |
| **Decontamination** | Removing eval-set overlap from training data. |
| **FSDP / ZeRO-3** | Sharding parameters, gradients and optimizer states across data-parallel ranks. |
| **GQA** | Grouped-query attention; K/V heads shared across groups of Q heads. |
| **MFU** | Model FLOPs utilization; achieved ÷ theoretical peak FLOPs. |
| **MinHash LSH** | Scalable near-duplicate detection via hashed shingles. |
| **MoE** | Mixture of experts; sparse MLPs with a learned router. |
| **μP** | Maximal update parametrization; makes optimal LR width-invariant. |
| **Pre-norm** | Normalization before the sublayer; keeps the residual path clean. |
| **RoPE** | Rotary position embedding; encodes relative position via rotation. |
| **Teacher forcing** | Conditioning on ground-truth prefixes during training. |
| **Tokens/param** | `D/N`; the training-intensity ratio. Chinchilla ≈ 20, modern models 100–2000+. |
| **WSD** | Warmup–Stable–Decay LR schedule. |
| **z-loss** | Auxiliary penalty on the log-softmax normalizer; stabilizes logits. |

---

## A 12-week plan, if you want one

| Weeks | Focus | Deliverable |
|---|---|---|
| 1 | Prereqs, autograd | micrograd + Checkpoint 2 |
| 2 | Transformer from scratch | Rungs 2–3 |
| 3 | Tokenization + data | Checkpoints 3–4 |
| 4 | Full data pipeline | Rung 4 |
| 5–6 | First real pretraining run | Rung 5, fully instrumented |
| 7 | Optimization + stability | Checkpoints 7, 10 |
| 8 | Scaling laws | Rung 6 |
| 9–10 | Distributed training | Rung 7, profiled and measurably faster |
| 11 | Evaluation | Checkpoint 11 |
| 12 | Depth topic | Rung 8 |

The single most valuable output of this is **Rung 5 with real instrumentation**.
A small run you understand completely is worth more than a large run you merely
launched.
