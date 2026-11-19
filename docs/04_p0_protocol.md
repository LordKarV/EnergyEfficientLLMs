# P0 protocol: agentic scope (v3)

Status: v3, 19 Nov 2026. Nothing here has been run. Time figures are estimates to be replaced by
the pilot. v2 applied the owner's agent's review (issue #1). v3 records the owner's decisions:
thinking on at 64K with 16 concurrent runs, one `g6e.16xlarge`, and spending approved for the
20-task pilot only. It also adds llama.cpp to the pilot, a CPU headroom rule, and a pilot exit
report from which the owner sets the ceiling for the rest of P0.

## What P0 must answer before any training

| Gate | Question | If it fails |
|---|---|---|
| P0a | Does stock PTQ hurt a 4B coding agent on our tasks? | **stop** the selection method |
| P0b | Can we build a pool of at least 4 successful trajectories per task, at a usable size? | fallbacks (below) |
| P0c | Does the pool contain the tasks quantization breaks? | fallbacks, or claim generalization only |
| P2 | Does a low stability score mean the quantized model can follow that trajectory? | **stop** KL scoring; keep length and entropy arms |

Order: the 20-task pilot first. **Only the pilot is approved.** The owner sets the spending
ceiling for the rest of P0 from the pilot exit report. Then P0b, with P0a alongside; then P0c and
P2 on P0b's pool.

## Fixed setup for all of P0

### Model and decoding

- `Qwen/Qwen3.5-4B`, text-only (`Qwen3_5ForCausalLM`).
- Thinking mode and context (owner decision D-P0-1): **thinking on, 64K context, 16 concurrent
  runs.** The model card advises at least 128K in thinking mode; the owner chose 64K knowing that.
  Check: if more than 10% of pilot attempts end at the 64K limit, the question goes back to the
  owner before the main P0b run. No silent switch.
- Sampling, pinned per request in both runtimes and logged: `temperature=0.6, top_p=0.95,
  top_k=20, min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0` (card, thinking-mode coding).
  Server defaults are never relied on.
- No greedy decoding (it risks endless repetition in thinking mode). Every "first run" is a sampled
  run with a fixed seed. The pilot logs how often attempts hit the turn or context limit.

### Episodes are token sequences

- The harness stores each episode as a token sequence. Each turn appends the exact generated tokens,
  then the template's rendering of the tool response plus the next generation prompt. Prompts are
  sent as raw tokens or text. The conversation is never re-rendered from messages, because the
  template trims reasoning, re-wraps it, and rebuilds tool calls in canonical form. Re-rendering
  would break the prefix cache, inflate processed tokens and joules, and make the teacher-forced
  score condition on a context the model never saw.
- Consequence: the scored sequence is exactly the generated one, and one forward pass over the
  final sequence gives every position.
- Masking: sampled tokens, including the whole `<tool_call>` block, count as model-written.
  Template-inserted tokens don't. The logit at position i predicts token i+1.

### Action format, parsing, repair

- One bash tool, as in `mantra-energy` paper-2: 180 s tool timeout, output capped at 10,000
  characters, `--network none`, 4 CPUs and 16 GB per container.
- The model emits Qwen3.5's native XML-style call
  (`<tool_call><function=bash><parameter=command>...`). Tool definitions go into the system prompt
  via the official template.
- Parsing: raw completions from every server, server-side tool parsers off, one parser in our
  harness for every runtime. The raw text of every failed parse is logged.
- Malformed-call categories, fixed before P0a: missing or misnested closing tag; wrong function or
  parameter name; text after `</tool_call>`; call truncated at the token limit; no call and no
  final answer.
- Repair: "malformed call, try again" arrives **inside a `<tool_response>`**, never as a user
  message. A user message mid-episode moves the template's last-query index, after which earlier
  turns' reasoning is dropped from context. Each repair is a turn and is charged to the attempt
  (`llm-agent-recovery` accounting).

### Runtimes, budgets, profiles

- SWE-bench Verified runtime from paper-2 as is. A new R2E-Gym runtime with the same interface
  (`begin / context / tool / finalize / verify`), graded by R2E-Gym's own tests. Datacurve and
  Terminal-Bench are not in P0.
- Budgets per attempt: 64K context, 100 turns, 1,800 s wall clock. Hitting a limit is a
  failure, logged with its reason.
- Serving: vLLM for the large BF16 sampling in P0b; llama.cpp at one pinned commit for every
  comparison between precisions, BF16 included. llama.cpp splits `-c` across `-np` slots, so
  N slots at context C need `-c N*C` (16 x 64K = 1,048,576).
- Profiles: `BF16` GGUF, `Q8_0`, `Q4_K_M`, `Q3_K_M`, and `Q4_K_M` with `ssm_alpha`/`ssm_beta`
  overridden to Q8_0 (the gate-tensor ablation). Default `llama-quantize` flags, no imatrix in P0.
  The per-tensor type table of every file is saved with its results.

### Task sets (seeded ID lists saved before anything runs)

- Pool repos (6): pandas, numpy, pillow, orange3, aiohttp, tornado (3,887 tasks).
- Held-out repos (4, never in any pool): scrapy, pyramid, datalad, coveragepy (691 tasks).
- `pool200`: 200 tasks stratified across the pool repos. `heldout100`: 100 tasks stratified
  across the held-out repos. `verified100`: 100 random SWE-bench Verified tasks.

### Logged per attempt

Resolved, failure reason, turns, tool calls, malformed calls by category, repairs, output tokens,
input tokens split into cache hits and misses, wall time, container CPU-seconds, GPU joules (NVML
via Zeus), sampler parameters, seed, and the token sequence with model-written spans marked.

## P0b: pool feasibility (first)

- Pilot (the only approved spend so far):
  - vLLM `BF16`: 20 `pool200` tasks x 4 attempts.
  - llama.cpp `BF16` and `Q4_K_M`: the same 20 tasks x 1 attempt each. About 6,200 of the P0
    attempts run on llama.cpp, and its throughput with 16 slots of 64K on a hybrid model is the
    least certain number in this plan.
  - Concurrency: 14 containers at 4 vCPUs, leaving 8 vCPUs for the inference server, the harness
    and Docker. The measured container CPU load decides between 14-15 containers at 4 vCPUs or
    16 at 3 vCPUs for the rest of P0.
- **Pilot exit report** (what the owner reads to set the ceiling):
  - attempts per hour, separately for vLLM and llama.cpp
  - peak GPU memory per runtime
  - share of attempts hitting each limit: turns, 64K context (the 10% check), wall clock
  - malformed-call rate by category
  - container CPU load, and the concurrency chosen
  - projected instance-hours for each P0 step, and the total
- Main: BF16 on vLLM, 16 attempts on each `pool200` task and each `heldout100` task (held-out
  trajectories are measurement only and never enter a pool).
- Report: successes out of 16 per task and per repo, tasks with at least 4 successes, and the
  projected pool size = (share of `pool200` with at least 4 successes) x 3,887.
- Pass, all three:
  1. Among `pool200` tasks with at least 1 success, more than half reach 4.
  2. The projected pool is at least 500 tasks (one selected trajectory per task per arm), which
     means at least 13% of `pool200` reach 4 successes.
  3. `heldout100` is not at the floor: at least 10% of tasks solved at pass@16.
- Fallbacks, in order: `min_candidates` 2; a teacher pool from DeepSWE-Preview (Qwen3-32B) on the
  same tasks, scored with the 4B BF16/quantized pair; the 9B model.

## P0a: headroom

- Run the 5 profiles on llama.cpp on `heldout100` and `verified100`, 3 seeded sampled attempts
  each (3,000 attempts).
- **Pre-registered primary test:** `Q4_K_M` vs `BF16`, processed tokens per resolved task (failed
  attempts charged to resolved ones, 2610.32417 Eq. 4), on `heldout100`. Ratio with a 5,000-draw
  paired bootstrap over tasks and a geometric mean. **Pass:** at least 15% inflation, with a 95%
  interval excluding zero.
- Secondary (reported, not used for the gate): every other profile, turns per resolved task,
  `verified100`, and resolve rate per profile with McNemar on paired tasks. At 100 tasks the
  resolve-rate interval is about plus or minus 8 points.
- Diagnostic: teacher-forced KL between BF16 and each quantized profile on BF16 trajectories, by
  turn index (does damage grow with position in a hybrid model?), with the chunked wrapper at
  model-written positions only.
- Energy sub-run: 30 tasks per profile at concurrency 1, 45 s settling, idle power measured once
  per machine, giving per-attempt joules and the Eq. 2-3 decomposition. The full run reports
  joules only as run-level totals, since energy can't be split between concurrent attempts.
- The SteerDistill GPTQ gap (-7.46 points at 4B) is a prior, not a substitute for this test.

## P0c: does the pool reach the flips?

- On `pool200`, on llama.cpp: `BF16`, `Q4_K_M` and `Q3_K_M`, 4 seeded sampled attempts each.
- A **flip** is defined against llama.cpp `BF16` in the same runtime: quantized pass rate lower than
  BF16's by more than 0.25. The vLLM 16-attempt counts are used only to bin tasks by BF16
  successes (0, 1-3, 4-8, 9-16).
- Pass: at least 40% of flipped tasks fall in the 4-plus bins. Otherwise, use the P0b fallbacks or
  state that any effect works through generalization.

## P2: does the score mean what we claim?

- Tasks: `pool200` tasks with at least 4 successful BF16 trajectories (target at least 50).
- Score every successful trajectory two ways: `kl_mean` over model-written tokens, and
  decision-point KL (first 8 tokens of each action plus the top 10% BF16-entropy positions). Probe
  profile: `Q3_K_M`.
- For each task, take the lowest- and highest-scoring trajectory. Replay the first half of its
  turns in a fresh container (re-run the recorded commands; drop the pair if any tool output
  differs from the recording). Continue each prefix with **both `BF16` and `Q3_K_M`, 4 seeded
  attempts each**.
- Test, difference-in-differences: (Q3 success minus BF16 success) on low-score prefixes vs on
  high-score prefixes, paired over tasks. This controls for low-score prefixes simply being further
  along. **Pass:** p < 0.05, and the effect is larger than the same test with prefixes ranked by
  `low_entropy` or by `clean_format`. Auto-bench found model-level KL did not predict coding-agent
  performance, so this is the gate that matters most.

## Kill rules (same as the gates table)

- P0a fails: stop the selection method.
- P2 fails: stop KL scoring and keep the RQ1-style length and entropy arms.
- P0b or P0c fails: apply the fallbacks in order. If all fail, stop.
- In every case, the constrained-decoding and calibrated-quantizer baselines still run.

## EC2 (decided: D-P0-2)

Facts checked by the owner's agent against the AWS g6e page and `config.json`:

- **Decided (D-P0-2): one `g6e.16xlarge`**: 1x L40S 48 GB, 64 vCPUs, 512 GiB RAM, about 1,900 GB of
  local NVMe (ephemeral).
- KV cache for Qwen3.5-4B (8 of 32 layers full attention, 4 KV heads, head_dim 256): 32 KiB per token,
  2.0 GiB per 64K sequence, 4.0 GiB per 128K sequence. DeltaNet state is 24-48 MiB per sequence.
  Weights about 9 GB. So 16 rollouts at 64K, or 8 at 128K, fit in 48 GB with headroom; 16 at 128K
  does not. Hence D-P0-1.
- Sandboxes: 16 containers at 4 vCPUs would use all 64 vCPUs and leave nothing for the server,
  the harness or Docker, so the pilot runs 14 and decides from measured load (see P0b).
- Disk: about 300 images at 0.4-1.3 GB compressed each, roughly 150-400 GB. The local NVMe covers
  P0. Keep results on EBS or S3, since NVMe is wiped on stop.
- If CPU becomes the bottleneck: keep the GPU box and add a separate CPU instance for containers.
  There is no 2-GPU g6e size, and `g6e.12xlarge` (4 GPUs, 48 vCPUs) has fewer vCPUs than the 16xlarge.
- 9B and the main experiment: revisit after P0.
- Prices: check the EC2 console before approving. Not checked here.

### Energy boundary on EC2

GPU via NVML (Zeus), read every 0.1 s. Virtualized instances generally don't expose CPU energy
counters, so every energy table states "GPU only" and reports container CPU-seconds next to it.
2610.32417 cites a 41-45% system-energy gap for GPU-only telemetry, and the paper should say so.

## Compute estimate (replace after the pilot)

About 11,000 attempts: P0b 4,800, P0a 3,000 plus the energy sub-run, P0c 2,400, P2 800. At 14-16
concurrent attempts and a guessed 10 minutes each, roughly 5 days on one machine. The pilot exit
report replaces this.

## What runs on the Macs

Chunked KL wrapper development (PyTorch `logits_to_keep`, then the llama.cpp low-level API), MLX
cross-check scoring, and harness smoke tests on 3-5 tasks under Rosetta (correctness only).

## Owner decisions

- D-P0-1: thinking on, 64K, 16 concurrent runs (with the 10% context-limit check).
- D-P0-2: one `g6e.16xlarge`.
- Spending: the 20-task pilot only. The ceiling for the rest of P0 is set from the pilot exit report.
- Open: who builds the pilot harness (owner's side, or PRs from LordKarV's fork).
