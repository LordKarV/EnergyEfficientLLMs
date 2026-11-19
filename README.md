# Q-TRACE: quantization-robust training for energy-efficient reasoning

Does it matter *which* correct solution you fine-tune on, if the model will be quantized afterwards?

Start with the two documents in `docs/`:

- `docs/01_literature_and_gaps.md`: what already exists and what is open.
- `docs/02_research_plan.md`: decisions made, research questions, go/no-go studies, schedule, risks.

## The pipeline

Six scripts, run in order. Each reads `configs/base.yaml` and writes into `outputs/`.

| Step | Script | What it does |
|---|---|---|
| 1 | `scripts/01_generate_pool.py` | Sample 16 solutions per training problem from the full-precision model and grade them. |
| 2 | `scripts/02_score_pool.py` | For each correct solution, measure how much a quantized copy of the model disagrees with the full-precision model along that solution (teacher-forced KL). |
| 3 | `scripts/03_select.py` | Build one training set per arm: random, shortest, stable, qtrace (short and stable), plus controls. |
| 4 | `scripts/04_train_lora.py` | LoRA fine-tune on one arm, merge, save. Hand-written training loop. |
| 5 | `scripts/05_evaluate.py` | Evaluate a model at several precisions: accuracy, tokens, time, memory, GPU energy. |
| 6 | `scripts/06_summarize.py` | One table with confidence intervals; paired comparison between two models. |

The library code is in `qtrace/`, one small file per job:

| File | Job |
|---|---|
| `common.py` | config, JSONL, seeding |
| `data.py` | load problems, build the prompt |
| `answers.py` | extract and grade the final number |
| `models.py` | load a model at a named precision |
| `quantize.py` | our own round-to-nearest quantizer, about 20 lines of logic |
| `generate.py` | batched generation |
| `select.py` | the selection rules for every arm |
| `energy.py` | GPU energy counter |

## Try it in one minute, no GPU

```
pip install -r requirements.txt
python -m pytest tests -q          # 10 unit tests
bash scripts/00_smoke_test.sh      # full pipeline on a tiny random model
```

The smoke test builds a 300k-parameter random model, so accuracy is zero. It only proves the
code path works.

## Run it for real (Linux with an NVIDIA GPU)

```
python scripts/05_evaluate.py --model Qwen/Qwen3-0.6B --tag base     # study P0: is there headroom?
python scripts/01_generate_pool.py
python scripts/02_score_pool.py
python scripts/03_select.py
for arm in random shortest stable qtrace; do
  python scripts/04_train_lora.py --arm $arm --seed 0
  python scripts/05_evaluate.py --model outputs/runs/${arm}_seed0/merged --tag ${arm}_seed0
done
python scripts/06_summarize.py
python scripts/06_summarize.py --compare qtrace_seed0 shortest_seed0 --precision rtn3
```

Set `data.train_limit` and `data.test_limit` in the config to something small for a first pass.

## What has and has not been tested

Tested: all unit tests, and the full six-step pipeline end to end on CPU with a tiny random
Qwen3-architecture model, including LoRA on the real Qwen3 module names, merge, save, reload,
and simulated quantization. The stability score was checked to be exactly zero against an
unquantized copy and to grow as bits drop.

**Not tested**, because the machine this was written on had no GPU and no access to the
Hugging Face hub:

- downloading and running the real `Qwen/Qwen3-0.6B` and the GSM8K dataset
- any CUDA path, including bfloat16 autocast in training
- `bnb8` / `bnb4` loading through bitsandbytes
- the NVML energy counter in `qtrace/energy.py`
- throughput: every GPU-hour figure in the plan is an estimate

Expect to fix small things the first time you run on a GPU. Do that with `train_limit: 50`.

## Known limits of this starter code

- `rtnN` precisions simulate low-bit weights. Accuracy is faithful. Speed, memory and energy are
  not, and the evaluation output says so. Real low-bit runtimes are a week 1-2 task (plan, P4).
- Only round-to-nearest is implemented. GPTQ and AWQ should come from an established library.
- Evaluation uses greedy decoding and one prompt format.
- Scoring runs one solution at a time for readability. Batch it if it becomes the bottleneck.
- Baselines 3 and 4 in the plan (overthinking penalty, noise-injected fine-tuning) are not written yet.
