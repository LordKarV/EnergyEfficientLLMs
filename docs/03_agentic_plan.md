# Q-TRACE, agentic version: the plan

Status: 19 Nov 2026. Agreed in issue #1 between the owner's agent and LordKarV's agent, with the
owner's decisions. Open: who builds the pilot harness, and the spending ceiling after the pilot. This plan
replaces the GSM8K plan in `docs/02_research_plan.md` for the main study. The detailed protocol for
the first experiments is `docs/04_p0_protocol.md`.

## The idea in one paragraph

Small coding agents are usually run quantized, to save memory and energy. Quantized agents make more
malformed tool calls and take more turns, so they can end up using more energy per solved task, not
less (arXiv 2610.32417, 2610.41672). We test a cheap training-side fix: from many successful runs of
the full-precision model, pick the runs that a quantized copy of the model can also follow, fine-tune
on those, then quantize with ordinary stock tools. If it works, a deployer gets a model that keeps its
accuracy and its energy savings after quantization, without quantization-aware training, a teacher
loop or any special quantizer.

## The question

Does choosing **which** successful trajectories to fine-tune on make a small coding agent hold up
better under stock post-training quantization, measured in resolve rate, turns, tokens and joules per
resolved task?

Hypotheses, each able to fail:

- H1: fine-tuning on the fewest-turn successful runs shrinks the quantization gap compared with
  random successful runs.
- H2: picking runs by stability under quantization (`qtrace`) shrinks the gap more than fewest-turns
  alone, at equal or fewer turns.
- H3: the stability score is more than entropy or clean formatting: `stable` beats `low_entropy` and
  `clean_format`.
- H4: the differences between arms survive grammar-constrained tool calls, so the gain is not just
  formatting.
- H5: at 4-bit, the best arm uses the fewest joules per resolved task.

## What others have done, and where we differ

| Work | What it does | Our difference |
|---|---|---|
| SteerDistill (2611.13410) | trains the quantized agent with a teacher; at Qwen3.5-4B, stock GPTQ loses 7.46 points and SteerDistill recovers 6.73 of them | we train in full precision and keep any stock quantizer; no teacher loop; coding tasks |
| 2610.32417, 2610.41672 (Texas A&M) | measure that quantized agents cost more energy per completed task, including SWE-bench Verified | they measure, we try a fix; we reuse their harness and energy accounting |
| HERO (2609.38885) | RL that cuts tokens 25-48% for SWE agents | full precision only; no quantization |
| Munkhbat et al. 2025 | fine-tune on the shortest correct solution | math, never quantized; our `fewest_turns` arm |
| OnQ, on-policy distillation | train the quantized model with a teacher | same distinction as SteerDistill |

What is still open: no published work changes only the fine-tuning data, keeps stock quantization,
and reports joules per resolved task on SWE-style agent tasks.

## Setup

- **Model:** Qwen3.5-4B, with 9B for the scale check. Thinking on, 64K context, 16 concurrent
  runs (owner decision; the card advises 128K, so the pilot checks how often runs hit 64K).
- **Training tasks:** R2E-Gym (the environment set of Agentica's DeepSWE recipe), 6 repos: pandas,
  numpy, pillow, orange3, aiohttp, tornado.
- **Primary test:** 4 R2E-Gym repos never used in training (scrapy, pyramid, datalad, coveragepy;
  691 tasks). A within-repo split is reported next to it, to show how much is repo memorization.
- **External checks:** SWE-bench Verified (500), Datacurve DeepSWE (113, reported as Python vs
  non-Python), Terminal-Bench 2.1 (89, on its own harness).
- **Quantization profiles:** llama.cpp GGUF at one pinned version: `BF16`, `Q8_0`, `Q4_K_M`
  (primary), `Q3_K_M`, plus `Q4_K_M` with the two DeltaNet gate tensors kept at 8-bit. MLX and AWQ
  are secondary. Our uniform round-to-nearest quantizer is a harsh ablation only.
- **Harness:** one bash tool in Qwen3.5's native XML call format; episodes stored as raw tokens;
  one parser for every runtime; "try again" sent as a tool response. Built on the MIT-licensed
  `mantra-energy` paper-2 harness, with new plugins for R2E-Gym and Datacurve.
- **Hardware:** one EC2 `g6e.16xlarge` (1x L40S 48 GB, 64 vCPUs, 512 GiB) for sandboxes, runs and
  energy; all task images are amd64-only. The Macs do score development, MLX cross-checks and
  harness smoke tests.

## Method

1. **Pool:** run the full-precision model 16 times on each training task. Keep tasks with at least
   4 successful runs.
2. **Score:** for each successful run, measure how much a quantized copy (`Q3_K_M`) disagrees with
   the full-precision model on the tokens the model wrote (teacher-forced KL; whole-run average and
   at decision points only).
3. **Select:** build one training set per arm by picking one run per task:
   - `random`: any successful run
   - `fewest_turns`: the shortest successful run
   - `stable`: the run the quantized copy disagrees with least
   - `qtrace`: a mix of fewest turns and most stable
   - `low_entropy`: the run the full-precision model was most certain about (control)
   - `clean_format`: a run with no malformed tool calls (control)
4. **Train:** LoRA fine-tune on each set, same number of steps for every arm, 3 seeds.
5. **Quantize and evaluate:** merge, quantize with each stock profile, evaluate everything.

## Baselines

- The untouched base model, quantized the same way.
- Grammar-constrained tool calls at decode time, crossed with every arm (tests H4).
- A calibrated quantizer: llama.cpp importance matrix computed on the pool's trajectories.
- Optional if budget allows: an RL baseline (HERO-style) and a direct SteerDistill run on our tasks.

## Metrics

- **Headline:** the quantization gap (score at a profile minus score at BF16, same model), with the
  base model in every table.
- Resolve rate, turns, processed tokens (cache hits and misses), malformed calls by type.
- **Energy:** joules per resolved task, with failed runs charged to solved ones, broken down into
  per-token cost x extra tokens x extra failures (2610.32417 Eqs. 2-4). GPU only on EC2, stated in
  every table, with container CPU time next to it.

## Phases

| Phase | What | Gate |
|---|---|---|
| P0 | pilot, pool size, headroom, pool reach, score check (`docs/04_p0_protocol.md`) | four go/no-go gates |
| P1 | one-seed pilot of all arms on the primary test | direction and effect size |
| P2 | full run: all arms, 3 seeds, all profiles, primary test plus within-repo split | pre-registered primary test |
| P3 | external checks: SWE-bench Verified, Datacurve, Terminal-Bench | none, reported as is |
| P4 | 9B replication and the optional baselines | budget |

Post a preprint as soon as P2 has a clean answer. Related papers are appearing monthly.

## Kill rules

- Quantization doesn't hurt the base model (P0a): stop the method.
- The stability score doesn't predict whether the quantized model can follow a run (P2 of P0): drop
  the score and keep the fewest-turns and entropy arms.
- The pool is too small or misses the tasks quantization breaks: use the fallbacks (fewer required
  successes, a DeepSWE-Preview teacher pool, the 9B model), then stop if all fail.

## Cost

Only the 20-task pilot is approved. Its exit report gives measured throughput, and the owner sets
the ceiling for the rest of P0 from it. Current estimate: P0 is about 11,000 agent runs, roughly
5 days on the one machine. P1-P4 are sized after P0.

## Who does what (proposed)

- Owner: the spending ceiling after the pilot; who builds the pilot harness; merges this plan.
- Owner's agent: the R2E-Gym runtime plugin, the token-level episode store, the parser and repair
  path, the per-tensor quantization tables.
- LordKarV's agent: the chunked KL scoring wrapper (PyTorch first, then llama.cpp), the P0 analysis
  scripts (bootstrap, flip tables, the P2 difference-in-differences), and the related-work tracking.
- Both: review each other's code in PRs, discuss in issue #1.
