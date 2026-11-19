# Q-TRACE research plan

Written 13 Nov 2026 while you were offline. You could not answer questions, so I made five
calls on your behalf. They are listed first so you can overrule any of them.

## Decisions I made for you

| # | Decision | Why | Cost of reversing |
|---|---|---|---|
| D1 | Keep **Qwen3-0.6B** as the primary model, add Qwen3-1.7B and Qwen3-4B for scale. Do not switch to Qwen3.5-0.8B. | Two 2026 papers report Qwen3-0.6B numbers, so results are comparable. Qwen3.5 uses a new hybrid architecture and is multimodal; quantization tooling for it is younger. | One config line. |
| D2 | **Non-thinking mode**, with the reasoning written in the visible answer. | It keeps solutions short and gradable, and matches the concise-reasoning baseline setup. One public Qwen3-0.6B run reports thinking mode never closing its think block within 2,560 tokens in their harness. Thinking mode becomes a later ablation on the larger models. | One config flag, but all pools must be regenerated. |
| D3 | Add **3-bit** to the precision ladder and treat it as the main stress setting. | Prior work finds 8-bit and 4-bit weight quantization near lossless on these benchmarks. Without a setting that hurts, there is nothing to make robust. | None. |
| D4 | Accuracy experiments use our own **simulated RTN** quantizer; energy is measured separately on a **real low-bit runtime**. | Simulated quantization is exact for accuracy and runs anywhere, including a laptop. It says nothing about speed or energy. bitsandbytes is likely to cost energy at this model size. | None; both paths are in the plan. |
| D5 | The proposed score is **teacher-forced KL** between the full-precision model and a quantized copy, per solution. | It is cheap (two forward passes), needs no training, and rests on a published mechanism (Lotfi et al. 2026). | Swap one function in `scripts/02_score_pool.py`. |

## Candid assessment

The original framing would not survive review as written. Four reasons:

1. The field moved fast this year. At least five papers since May 2026 cover "quantization makes
   reasoning worse and longer" and "train it back". See `01_literature_and_gaps.md`.
2. INT8 and 4-bit on GSM8K are the mildest settings on the least sensitive benchmark. You may
   measure no degradation, which leaves the hypothesis untestable.
3. "Quantization-stable" was undefined. A reviewer's first question is whether it is just length,
   or just low entropy.
4. Energy with bitsandbytes on a 0.6B model will probably show quantization *increasing* energy.

What survives, and is open: **selecting which correct demonstration to fine-tune on, as a way to
make the model robust to stock post-training quantization.** No paper I found does this.

Is it top-conference material? With one 0.6B model on GSM8K: no, that is a workshop paper. With
three model sizes, two task families, two quantizers, three seeds, real energy numbers, and an
effect that clears the noise: plausible for a main track, and a solid Findings or TMLR paper if
the effect is modest. Your budget covers the larger version comfortably. The binding constraints
are rigor and speed, not compute.

## Research questions

**RQ1. Does concise-reasoning fine-tuning make a model more or less robust to PTQ?**
Compare `random`, `shortest`, `longest` across the precision ladder. Publishable either way, and
it needs nothing new: it is two existing ideas that nobody has crossed.

**RQ2. Can a per-solution stability score pick demonstrations that yield a more PTQ-robust model,
beyond what length and entropy explain?**
Compare `stable` and `qtrace` against `shortest`, `low_entropy`, and the 2x2 arms.

**RQ3. What is the effect in joules per correct answer on a real low-bit runtime?**
Tokens per answer from the accuracy runs, times measured joules per token.

### Hypotheses, written so they can fail

- H1: At 3-bit, `shortest` loses fewer accuracy points relative to its own BF16 score than `random`.
- H2: At 3-bit, `qtrace` beats `shortest` in accuracy by more than the paired-bootstrap noise,
  at equal or lower mean output tokens.
- H3: `short_stable` beats `short_unstable`, and `long_stable` beats `long_unstable`. This is the
  test that stability matters independently of length.
- H4: `stable` beats `low_entropy`. If not, the quantized probe is unnecessary and the story becomes
  "train on low-entropy solutions", which is simpler and still new, but a smaller claim.
- H5: Models trained on stable solutions show less token inflation under quantization.

## Metrics

- Accuracy, with a 95% bootstrap interval. GSM8K test has 1,319 problems, so a single run is
  uncertain by roughly plus or minus 2.5 points. Differences smaller than that need paired tests and seeds.
- **Quantization gap**: accuracy at precision p minus accuracy at BF16, per model. This is the
  robustness measure. Report it next to absolute accuracy, since a method can shrink the gap by
  lowering the BF16 score.
- Mean output tokens, token inflation relative to BF16, and unfinished rate.
- Joules per problem, per token, and per correct answer (real runtimes only).

## Preliminary studies: the first two weeks

These are go or no-go gates. Do them in order. Each has a pass condition.

### P0. Is there anything to fix? (1-2 days)
Evaluate the untouched base model at bf16, rtn8, rtn4, rtn3 on GSM8K test, and on MATH-500.
```
python scripts/05_evaluate.py --model Qwen/Qwen3-0.6B --tag base
```
**Pass:** some setting loses at least 5 accuracy points or inflates tokens by at least 15%.
**If it fails at 4-bit but passes at 3-bit:** 3-bit is the headline setting.
**If GSM8K barely moves at any setting:** make MATH the primary task. Prior work found harder
benchmarks degrade several times more than GSM8K.
**If RTN-3 collapses to near zero:** it is too destructive to be informative. Add a calibrated
quantizer (GPTQ or AWQ at 3-bit) before going further.

### P1. Does the pool have the variation the method needs? (2-3 days)
Run scripts 01 and 02 on the full training set. Then compute:
- Share of problems with at least 4 distinct correct solutions. Need more than half.
- Within-problem spread of length and of `kl_mean`. If solutions to the same problem all have
  nearly the same KL, there is nothing to select on.
- Within-problem rank correlation between length and `kl_mean`. **If its magnitude is above
  about 0.8, stability is a proxy for length** and the project reduces to RQ1.
- Rank correlation of `kl_mean` across probes (rtn3 vs rtn4, and vs GPTQ-3 once available). A
  score that changes ranking with the probe is not a property of the solution.

### P2. Does the score mean what we claim, before any training? (2 days)
Take each scored solution, keep its first half, and let the *quantized* model finish it with
greedy decoding. If low-KL solutions are completed correctly more often than high-KL ones for the
same problem, the score measures "a path the quantized model can follow". This is cheap, direct,
and makes a good figure. If there is no difference, rethink the score before spending on training.

### P3. One-seed pilot (2-3 days)
Train `random`, `shortest`, `stable`, `qtrace`; evaluate across the ladder. You are looking for
direction and effect size, not significance. This tells you how many seeds the real run needs.

### P4. Energy calibration (2 days, parallel to the above)
On the target GPU, measure joules per token for BF16 and for real 8-bit and 4-bit runtimes, at
batch size 1 and at a throughput batch size. Answer one question: **at 0.6B, does low precision
save energy at all on this hardware?** If not, say so in the paper and present the energy result
as "shorter outputs save energy; low-bit weights save memory", or move the energy study to a
setting where quantization does pay (the 4B model, or CPU and edge inference via GGUF).

## Main experiment

Arms (all implemented in `qtrace/select.py`): `random`, `shortest`, `stable`, `qtrace`,
`low_entropy`, `longest`, and the 2x2 `short_stable`, `short_unstable`, `long_stable`, `long_unstable`.

| Factor | Levels |
|---|---|
| Model | Qwen3-0.6B, then 1.7B, then 4B |
| Seeds | 3 per arm |
| Precision | bf16, 8, 4, 3 bits |
| Quantizer | RTN (ours), plus GPTQ or AWQ via an established library |
| Train task | GSM8K train; then MATH train |
| Test | GSM8K test; held-out word-problem sets (GSM-Plus, GSM-Symbolic, SVAMP); MATH-500 |

### Baselines a reviewer will expect
1. Base model, no fine-tuning.
2. `random` and `shortest` (Munkhbat et al.).
3. Lotfi et al.'s overthinking penalty applied at inference to every arm. It is orthogonal to
   data selection, so the interesting result is whether the two stack.
4. One method-side alternative at equal cost: LoRA fine-tuning on the `random` arm with
   simulated quantization noise in the forward pass. This answers "why select data instead of
   just training with noise?"
5. Mention, do not reproduce: OnQ, on-policy distillation, Silver Bullet. They train the quantized
   model with a teacher at far higher cost. State the cost difference in GPU-hours.

### Confounds and how each is handled

| Confound | Control |
|---|---|
| Stable solutions are also shorter | 2x2 arms; report mean length per arm |
| Stability is just low entropy | `low_entropy` arm |
| Short arms see fewer training tokens | Fixed step count for all arms; `train_meta.json` logs tokens seen; add one token-matched run |
| Pool only contains problems the model already solves | Same pool for every arm; later add a teacher-generated pool from a larger model |
| Probe quantizer differs from deployment quantizer | Score with RTN-3, evaluate under GPTQ or AWQ too |
| Score computed on the base model, but the fine-tuned model is what gets quantized | Re-score the pool with the fine-tuned model and report rank agreement; optionally run a second selection round |
| Seed noise | 3 seeds, paired bootstrap on shared test problems |
| GSM8K is saturated or memorised | Held-out sets and MATH |

## Energy protocol

1. **Never report energy from `rtnN` runs.** The evaluation script marks them `efficiency_is_real: false`.
2. Real runtimes: vLLM with 4-bit AWQ or GPTQ weights for the GPU story; llama.cpp GGUF
   (Q8, Q4, Q3 variants) for the CPU and edge story. Pick one as primary after P4.
3. Read the GPU's cumulative energy counter before and after (`qtrace/energy.py`). Run a warm-up
   batch first. Repeat each measurement 3 times. Fix batch size and report it.
4. Measure idle power once per machine and report both raw and idle-subtracted joules.
5. State the limits: GPU energy only. EC2 guests generally cannot read CPU energy counters.
6. Headline number: **joules per correct answer**.

## The optional weight-space extension

The MIT result is about repairing thinking loops in a full-precision Qwen3.5-0.8B using a frontier
model as the editor. It is a blog post, and by the authors' own account it removes a looping
defect and does not add ability. I would not build on it directly.

A version you can run yourselves, if the main result lands with time to spare: after PTQ, find the
small set of weight columns that, when restored to full precision, most reduce the teacher-forced
KL at high-entropy positions. Compare against Lotfi's logit penalty, which targets the same
failure from the output side. This is a sparse full-precision patch, in the spirit of
outlier-preserving quantizers. Keep it out of scope until RQ1 and RQ2 have answers.

## Budget

Compute is not the constraint. Rough estimates for Qwen3-0.6B on one 24 GB GPU instance, which
rents for roughly one to two dollars an hour (check current AWS pricing; I did not look it up):

| Step | GPU-hours (estimate) |
|---|---|
| Pool generation, 7.5k problems x 16 samples | 2-4 |
| Scoring | 1-2 |
| Training, 10 arms x 3 seeds | 8-12 |
| Evaluation, 31 models x 4 precisions x 3 test sets | 20-30 |
| Mistakes and reruns | double it |

That is well under 200 dollars for the 0.6B study. The 4B model costs several times more per step.
The whole three-model study, with reruns, should fit in the low thousands. These are estimates
from token counts, not measurements; P0 and P1 will give you real throughput.

Spend the surplus on what strengthens the paper: more seeds, the 1.7B and 4B replications, a
teacher-generated pool from a large hosted model, and the MATH task.

## Ten-week schedule for three people

| Weeks | Person A: data | Person B: quantization and energy | Person C: training and writing |
|---|---|---|---|
| 1-2 | P1 pool and score analysis, P2 | P0 headroom, P4 energy calibration, add GPTQ or AWQ | P3 pilot; reproduce the `shortest` baseline's published length reduction |
| 3-5 | Held-out test sets, MATH pool, teacher pool | Real-runtime energy pipeline | Full 0.6B run: 10 arms x 3 seeds; baselines 3 and 4 |
| 6-7 | Re-scoring and second-round selection | 1.7B and 4B evaluation | 1.7B and 4B training |
| 8 | Analysis: which tokens differ between arms? | Energy tables | Draft |
| 9-10 | Ablations reviewers will ask for (alpha, N, probe bits) | Reproducibility pass | Paper |

Post to arXiv as soon as RQ1 has a clean answer. Related papers are appearing monthly.

## Risks, in order of likelihood

1. **No headroom.** Nothing degrades at realistic bit-widths. Caught by P0 in two days.
2. **Effect smaller than noise.** Selection changes accuracy by a point or two, inside the
   confidence interval. Mitigation: seeds, paired tests, larger held-out sets, focus on the gap metric.
3. **Score collapses into length or entropy.** Caught by P1 and H3/H4. The paper becomes RQ1
   plus a negative result on stability, which is still a workshop or short paper.
4. **Scooped.** Mitigation: speed, and an early arXiv post.
5. **Energy story inverts.** Low-bit costs energy at 0.6B. Caught by P4; report it plainly.

## What to do first when you wake up

1. Read the five "must read" papers in `01_literature_and_gaps.md`.
2. Run `bash scripts/00_smoke_test.sh` on a laptop to see the pipeline work end to end.
3. Start one GPU instance and run P0. Everything else waits on that result.
