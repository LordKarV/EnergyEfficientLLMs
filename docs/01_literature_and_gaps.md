# Literature and gaps

Survey date: 13 Nov 2026. Sources are linked. Where I only read an abstract or an
automated summary of a paper rather than the full text, the entry says so. Read the
five papers marked **must read** yourselves before committing: the plan depends on them.

## The short version

1. "Quantization hurts reasoning and makes it longer" is now established. It is not a contribution.
2. "Recover a quantized reasoning model by training it" is crowded, with well-resourced groups
   publishing QAT and distillation pipelines through 2026.
3. Nobody has asked whether the **choice among several correct demonstrations** changes how well a
   fine-tuned model survives ordinary post-training quantization (PTQ). That is your idea, and it is open.
4. Nobody in the quantized-reasoning line measures **energy**. They report tokens and sometimes latency.
5. Two parts of the original plan need to change: INT8 and 4-bit are probably too mild to show
   an effect, and bitsandbytes is likely to make a 0.6B model use *more* energy, not less.

## A. Quantization and reasoning (the direct neighbours)

| Paper | What it shows | Why it matters to us |
|---|---|---|
| Liu et al. 2025, [Quantization Hurts Reasoning?](https://arxiv.org/abs/2504.04823) | First systematic study. W8A8 and W4A16 are near lossless (within about 1%). Harder tasks and smaller models degrade more; GSM8K degrades far less than AIME. Output length grows only under aggressive low-bit settings, mostly in small models. | Your planned INT8 and 4-bit points may show no accuracy drop. GSM8K is the least sensitive benchmark they tested. |
| Li et al., ICLR 2026, [Quantization Meets Reasoning](https://arxiv.org/abs/2505.11574) | Step-level error taxonomy: PTQ raises method and execution errors. Fine-tuning the *quantized* model on 332 curated "Silver Bullet" examples for a few minutes restores W4A16 math accuracy to near full precision. | Closest prior "data fixes quantized reasoning" result. Differences: they tune the quantized model after the fact, with failure-targeted examples and an LLM judge pipeline. They do not select among alternative correct solutions and do not study length. |
| **must read** Lotfi et al. 2026, [Quantized Reasoning Models Think They Need to Think Longer](https://arxiv.org/abs/2606.00206) | Aggressive PTQ lowers accuracy and lengthens reasoning together. In up to 52% of failures the model reached the right answer mid-trace and talked itself out of it. Per-position KL between quantized and full-precision models correlates with full-precision entropy (Spearman 0.92): hesitation and branching tokens are unstable, math tokens are stable. A training-free logit penalty on 50 hand-picked markers cuts length 12-23%. | This is the mechanism our stability score relies on. It also gives the strongest objection to it: "your score is just entropy". The `low_entropy` control arm exists to answer that. Their penalty is also a required inference-time baseline. |
| **must read** Lian et al., EMNLP 2026, [Quantization Inflates Reasoning](https://arxiv.org/abs/2606.25519) | Token inflation as a hidden cost: on Qwen3-4B, INT4 adds about 5-12% reasoning tokens with good quantizers and about 43% with RTN; INT3 ranges from about 25% to nearly 300% depending on the quantizer. Calibration data matters a lot (WikiText to math CoT: 28% to 65% accuracy). Repetition penalties and "be concise" prompts are inconsistent. QAT on reasoning traces is the only fix that wins on both axes. Reports TTFT and throughput, not joules. | They say QAT was done "on reasoning traces" without saying how traces were chosen. That unspecified choice is exactly our variable. |
| [RATIO](https://arxiv.org/abs/2609.39801), 2026 (authors not checked) | Training-free. Finds model-specific overthinking tokens by comparing quantized and full-precision log-probs under teacher forcing, then applies per-token logit penalties. Up to +9.8 accuracy points and 51% shorter traces at 3-bit on a 1.5B model; near zero effect on some larger models. | Uses the same teacher-forced quantized-vs-full comparison as our score, but to pick tokens to penalise at inference. We use it to pick training examples. Cite and compare. |
| [Train Where the Quantized Model Goes](https://arxiv.org/abs/2609.26708), 2026 | On-policy distillation at about 1.9-2.8 bits on Qwen3-0.6B/1.7B/4B. Names "quantization-amplified exposure bias": errors compound along the student's own trajectories. Qwen3-0.6B MATH-500: BF16 27.2, their method 24.2 at 2.79 bits. | Gives a published Qwen3-0.6B BF16 reference point. Uses rollouts weighted by a correctness reward; no stability or length criterion. |
| **must read** [Right On Q (OnQ)](https://arxiv.org/abs/2610.29378), Oct 2026 | QAT with on-policy distillation and a dynamic KL loss at 2-3 bits on Qwen3-1.7B/4B/8B. One student response per prompt; no selection among trajectories. Reports token counts and truncation, not latency or energy. | The current state of the art for recovery. Shares the phrase "self-generated trajectories" with us, so the paper must draw the line clearly: they train the quantized student with a teacher; we change only which demonstrations an ordinary fine-tune sees, then apply stock PTQ. |
| Lv et al. 2026, [What Makes Low-Bit QAT Work for Reasoning LLMs?](https://arxiv.org/abs/2601.14888) (abstract only) | Distillation is a robust QAT objective; PTQ is a good initialisation; calibration domain should match training domain. Reports Qwen3-0.6B results. | Second published group working on Qwen3-0.6B. |
| Okoshi et al., ICLR 2026, [QAT for Ultra-Low-Bit Reasoning LLMs](https://proceedings.iclr.cc/paper_files/paper/2026/hash/209423f076b6479ab3a4f45886e30306-Abstract-Conference.html) (abstract only) | Two-stage QAT; mixed-domain calibration data beats single-domain. | More evidence that *data* choices matter for quantized reasoning, at the calibration stage. |

## B. Concise reasoning and trajectory selection

| Paper | What it shows | Why it matters to us |
|---|---|---|
| **must read** Munkhbat et al., ACL Findings 2025, [Self-Training Elicits Concise Reasoning](https://arxiv.org/abs/2502.20122) | Sample N solutions per question, keep the shortest correct one per question, fine-tune. About 30% fewer output tokens at similar accuracy on GSM8K and MATH. Code is public. | This is your arm 2 exactly, and the main baseline. They never quantize. |
| [LARK](https://arxiv.org/abs/2605.30651), 2026 (related-work section only) | Among several correct teacher trajectories for one question, pick the one most learnable for the student, using a forward-pass proxy. | Same selection setting as ours with a different criterion. A reviewer will ask for it, or at least for a likelihood-based control. |
| [MI-Distillation](https://arxiv.org/abs/2608.29623), 2026 (abstract only) | Selects trajectories that are informative and learnable for a small student. | Same family as LARK. |

## C. Energy

| Paper | What it shows | Why it matters to us |
|---|---|---|
| Delavande et al. 2026, [Understanding Efficiency](https://arxiv.org/abs/2601.22362) | On H100s with bitsandbytes int8/int4, lower precision saves energy only in compute-bound regimes. Batching and serving setup change energy per request by up to 100x. | The serving configuration can swamp the effect we want to measure. Fix it and report it. |
| Zhang 2026, [LLM Energy Benchmark](https://doi.org/10.5281/zenodo.18900289) (community dataset, not peer reviewed; I read a post about it, not the data) | Default bitsandbytes int8 used 80-162% *more* energy than FP16 on an RTX 4090D. NF4 has a parameter-count crossover below which it costs energy, roughly 3-5B on NVIDIA. | A 0.6B model is far below that crossover. Treat as a warning to verify in week 1, not as a fact to cite. |
| Chung et al. 2026, [Where Do the Joules Go?](https://arxiv.org/abs/2601.22076) | Decoding dominates LLM energy; reasoning tasks use about 25x more energy per response, from 10x more tokens plus higher energy per token. | Justifies "tokens times joules per token" as the right decomposition. |
| [Request and Token Energy Costs](https://arxiv.org/abs/2608.28044), 2026 | Uses the NVML cumulative energy counter; separates request energy from token energy. | Same counter our `qtrace/energy.py` reads. |

## D. Training affects PTQ robustness

| Paper | What it shows | Why it matters to us |
|---|---|---|
| Catalan-Tatjer et al., ICLR 2026, [Training Dynamics Impact PTQ Robustness](https://arxiv.org/abs/2510.06213) | How a model is trained changes how well it quantizes; links to flatter minima. | Precedent that robustness to PTQ can be shaped before quantization. Their lever is optimisation; ours is data. |

## E. The MIT CSAIL result behind the optional extension

[Sparks of Weight-Space Programming](https://www.theregister.com/ai-and-ml/2026/11/03/forget-retraining-ai-can-now-tweak-another-ais-weights/5307105)
(Takehi, Teoh, Zhang, Khattab; blog post, 2 Nov 2026; I read press coverage, not the post).
GPT-6 Astra edited 131,072 weights (0.016%) in MLP layers 15 and 19 of **Qwen3.5-0.8B** with no
gradient training, which cut thinking loops: MATH-500 went from 7.9% to 48.1% and the share of
answers hitting the 32,768-token limit fell from 93% to 33%. The authors themselves describe it
as loop repair, not added math ability, and report a negative control on Gemma 3 270M.

What this means for us: it is not a quantization result, it is not peer reviewed, and it depends
on access to a specific frontier model. It does line up with Lotfi et al.: the failure being
repaired is the model not committing to an answer. See the plan for a tractable version.

## The gaps, stated as claims you could defend in a related-work section

**G1. Demonstration selection has not been studied as a lever for PTQ robustness.**
OnQ uses one response per prompt. On-policy distillation weights rollouts by correctness only.
Lian et al. do not say how QAT traces were chosen. Munkhbat et al. select by length and never
quantize. LARK selects by learnability. No paper scores candidate solutions by how much
quantization disturbs them and trains on the result.

**G2. The interaction between conciseness training and quantization is untested.**
There are two reasonable and opposite predictions. Shorter chains give fewer steps for error to
accumulate and fewer high-entropy hesitation points, so they should be more robust. Or shorter
chains pack more computation into each token and remove the redundancy that lets a model catch
its own slips, so they should be more fragile. Either answer is a finding.

**G3. All recovery methods operate on the quantized model.**
They need QAT or distillation infrastructure, a teacher, and tens to hundreds of GPU-hours, and the
result is tied to one bit-width and quantizer. A method that only changes fine-tuning data leaves
the deployer free to apply any stock quantizer afterwards. Whether that works at all is unknown.

**G4. Energy is unmeasured in this line of work.**
Lotfi, Lian, RATIO, OnQ and the on-policy distillation paper report tokens, truncation, and in one
case latency. None reports joules. Meanwhile the energy literature says low-bit formats often do
not save energy for small models. Joules per correct answer on a real low-bit runtime is missing.

**G5. Sub-1B models at realistic bit-widths are thinly covered.**
Most results are at 1.5B and above, or at 2-3 bits with QAT. Mild PTQ on a 0.6B model sits between.

## What is NOT a gap any more

- Showing that quantization degrades math reasoning.
- Showing that quantized models produce longer reasoning.
- Showing that fine-tuning or distillation can recover a quantized model.
- A decoding-time penalty against overthinking.

If the paper's pitch reduces to any of these, it will be rejected as incremental.
