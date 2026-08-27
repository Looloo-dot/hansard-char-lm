# A character-level language model for the House of Commons

A one-layer GRU written from scratch in PyTorch, trained on ~2.1M characters of Hansard —
every attributed speech from five sitting days of the House of Commons, 9–16 July 2026.

**273,939 parameters · validation loss 1.3749 · per-character perplexity 4.0**, against a
bigram baseline of 2.44 (perplexity 11.5).

Two constraints fixed the design space: **at most 500,000 parameters**, and **no attention of
any kind**. That makes the interesting question not how low the loss can go, but where a fixed
budget should be spent when context and capacity compete for it.

The headline number is not the point of this repository. The controls are.

---

## The result that mattered

The obvious experiment — train at `block_size=8`, train at `block_size=64`, report the
difference — gives a validation gap of **0.398** at an identical parameter count. That number
is mostly an artefact, for two reasons predicted in the hypothesis *before* the runs:

1. **More context also means more gradient signal.** At `block_size=64` each optimiser step
   sees 32 × 64 = 2,048 target characters against 32 × 8 = 256. Eight times the training
   signal per update, confounded with the window width.
   → *Control:* a token-matched arm at `block_size=8, batch_size=256`.

2. **The two losses average over different tasks.** A GRU's hidden state is zeroed at the
   start of every window, so at `block_size=8` no position ever conditions on more than eight
   characters. The two validation losses are not estimating the same quantity.
   → *Control:* score every model on the **same** 64-character windows, position by position.

With both controls in place:

| Comparison | Val loss gap |
|---|---|
| Raw headline (block 8 → 64) | **0.398** |
| After token-matching | 0.129 of the gap was training signal, not context |
| After position-wise evaluation on shared windows | **0.041** |

**A tenth of the headline figure.** The reported result is 0.041, not 0.398.

The per-position curve also inverts the story: the token-matched `block_size=8` model is
*better* over positions 1–8 (1.679 vs 1.732), and only from position 9 does the ordering
reverse and hold (1.368 vs 1.422). Both curves are flat by roughly position 32 — the same
fading-context limit that shows up in the generated text.

## Design: why a GRU, and why stop at 55% of the budget

In a fixed-window model, **context costs parameters** — a window of *W* characters over
64-dimensional embeddings adds 64*W* weights per hidden unit, so window and network compete
for the same 500,000. A recurrent state carries context instead, which makes `block_size` a
pipeline argument that costs *nothing*.

```
embedding  V·d          =  83·64                 =   5,312
GRU        3H(d+H)+6H   =  3·256·(64+256)+6·256  = 247,296
head       H·V+V        =  256·83+83             =  21,331
                                          total  = 273,939   (55% of the cap)
```

Almost the entire budget sits in the recurrent weights — the part that does the remembering.

Stopping at width 256 was a decision from **measured returns, not from the ceiling**:

| Hidden width | Params | Val loss | Gain vs previous | Loss per parameter |
|---|---|---|---|---|
| 64 | 35,667 | 1.7253 | — | — |
| 128 | 90,515 | 1.5420 | 0.183 | 3.3 × 10⁻⁶ |
| 256 | 273,939 | 1.4048 | 0.137 | 0.75 × 10⁻⁶ |

The second doubling bought 75% of what the first did in absolute terms, but the fall *per
parameter* dropped 4.5-fold. That is what justified leaving 226,000 parameters unspent rather
than filling the budget because it was there.

## What the model actually learned

It learned the **shape** of Hansard, not its content. Every sample reproduces the corpus
format unprompted — capitalised two-part name, colon, newline, then a turn:

```
Grevieve Reed:
I have increding it to the victims of nature
```

Prompted with `The Prime Minister:`, it answers in the register of Question Time and then
hands the floor to a fresh invented header. The most diagnostic failure in the whole project:
**a backbencher's question comes out of the Prime Minister's mouth.** The header format is
learned; the constraint that format places on the turn is not. Names are plausible letter
sequences rather than real MPs ("Scutelocker"), words drift into near-words ("afternoties",
"recognife"), and no sentence holds a single subject.

That failure is exactly what the per-position curve predicts: flat past position 32, so
dependencies spanning the speaker header and the later turn are not available to be exploited.

## Limitations

- **The 0.041 residual is quoted against the wrong noise scale.** The 0.006 spread in the
  notebook comes from repeated `estimate_loss` calls on fixed weights. The right yardstick for
  a residual is seed-to-seed variance, which is measured here for the raw gap but not for the
  residual itself.
- **One seed pair per arm.** Each experiment is replicated at seed 2024 on a quarter of the
  compute, which checks the direction and rough size of each effect but is not a variance
  estimate.
- **The block-8 model is scored on window lengths it never trained on**, so the position-wise
  comparison is a diagnostic rather than a clean decomposition.
- **The train/validation split is positional**, so validation is a later stretch of Hansard
  with different business in it. Part of the 0.176 train/val gap at width 256 is distribution
  shift rather than memorisation — the 6,889-parameter bigram shows a 0.040 gap on the same
  split — and the two cannot be cleanly separated here.

## Reproduce

Open [`notebook.ipynb`](notebook.ipynb) and run all cells (GPU recommended; a full
restart-and-run-all completes in under ~20 minutes on a Colab GPU). `hansard.txt` is committed
here, so nothing is downloaded at run time.

`checkpoint.pt` is the trained model — a dict with `state_dict`, `block_size`, `n_params` and
`val_loss`:

```python
import torch
b = torch.load('checkpoint.pt', map_location='cpu', weights_only=True)
print(b['n_params'], b['block_size'], b['val_loss'])
# 273939 64 1.374898076057434

model = MyLanguageModel(vocab_size=83)   # constructor defaults rebuild the trained shapes
model.load_state_dict(b['state_dict'])
```

The notebook's verify cell reloads the checkpoint from scratch, re-scores it, and asserts the
parameter budget and the absence of any attention-like module — so the file has to reproduce
the number it claims. Every seeded run uses `torch.manual_seed(1337)`, with a seed-2024
replication of both experiments at a quarter of the compute.

## Data

Hansard, sourced via TheyWorkForYou from the official record and published under the
[Open Parliament Licence v3.0](https://www.parliament.uk/site-information/copyright-parliament/open-parliament-licence/),
which permits copying, adaptation and derivative works. 2,091,809 characters, 83-character
vocabulary, split positionally into train and validation.

## Stack

PyTorch · character-level tokenisation · GRU · AdamW (lr 1e-3, 4,000 iterations) · matplotlib.
No attention, no pretrained weights, no external model code — the architecture, the training
loop instrumentation and the position-wise diagnostic are written directly against the
pipeline in this notebook.
