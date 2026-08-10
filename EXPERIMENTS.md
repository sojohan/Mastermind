# Mastermind On-Policy Distillation — Experiment Log

This document summarizes training and evaluation runs for the Tinker-based Mastermind project in this repository.

**Acknowledgments:** This research was supported by a Grant from Thinking Machines Lab.

## Overview

**Goal:** Train a student model (`Qwen/Qwen3.5-9B-Base`) to play Mastermind via on-policy distillation from a teacher (`Qwen/Qwen3.5-9B`), optionally mixed with verifiable game rewards (black / white / solve).

**Scripts:**

| Script | Purpose |
|--------|---------|
| `onpolicy_mastermind.py` | OPD training (Tinker `train_on_policy`), single-turn episodes |
| `rlvr_mastermind.py` | RLVR training (`tinker_cookbook.rl.train`), full multi-turn games |
| `knuth_solver.py` | Knuth minimax teacher + trajectory dump for distillation |
| `sft_mastermind.py` | SFT on Knuth JSONL (`tinker_cookbook.supervised.train`) |
| `play_student.py` | Play full games against a hidden secret |

**Environment:** Python venv at `onpolicy/`, `TINKER_API_KEY` required.

**Default game:** 4 pegs, colors `{R, G, B, Y, O, P}`, repeats allowed, JSON output `{"guess": ["R", "G", "B", "Y"]}`.

**Training paradigms:** Runs 1–5 (OPD) use single-turn episodes — one guess from a synthetic mid-game prompt. Runs 6–11 (RLVR) play full multi-turn games with verifiable rewards and no teacher. **Run 12** distills Knuth trajectories via SFT. **Run 13** tried RLVR-for-speed on top of run 12 and regressed badly.

---

## Models and pricing (reference)

| Role | Model |
|------|-------|
| Student (OPD, runs 1–5) | `Qwen/Qwen3.5-9B-Base` + LoRA rank 64 |
| Teacher (OPD) | `Qwen/Qwen3.5-9B` |
| Student (RLVR / Knuth SFT, runs 6–13) | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` + LoRA rank 64 |

Tinker pricing used for estimates (per 1M tokens): Qwen3.5-9B prefill $0.44, sample/train $1.33. Nemotron-3-Nano prefill $0.13, sample $0.33, train $0.40 (limited-time 50% discount).

---

## Training runs

### Run 1 — Baseline OPD (truncated completions)

| Field | Value |
|-------|-------|
| **Log dir** | `runs/mastermind-opd-Qwen-Qwen3.5-9B-Base-64rank-0.0001lr-64batch-2026-07-06-09-16` |
| **Steps** | 100 |
| **Renderer** | `role_colon` (default for Base; thinking-style outputs in practice) |
| **max_tokens** | 256 |
| **Prompt** | Reasoning + JSON at end |
| **Env reward** | 0 (pure teacher KL) |
| **kl_penalty_coef** | 1.0 |
| **Cost** | ~$28 (actual) |
| **Final checkpoint** | `tinker://9e78e639-d31e-5a49-99bb-ca1e57091494:train:0/sampler_weights/final` |

**Play eval:** 0/3 games solved. Most turns failed to parse — responses hit `max_tokens=256` mid-reasoning (`stop_reason=length`). Model rambled with chain-of-thought; JSON rarely appeared.

**Conclusion:** Truncated rollouts taught non-terminating behavior; distillation could not learn to finish with JSON.

---

### Run 2 — OPD with longer completions (retrain)

| Field | Value |
|-------|-------|
| **Log dir** | `runs/mastermind-opd-Qwen-Qwen3.5-9B-Base-64rank-0.0001lr-64batch-2026-07-06-19-14` |
| **Steps** | 100 |
| **Renderer** | `role_colon` |
| **max_tokens** | 1024 |
| **Prompt** | Updated (reasoning then JSON) |
| **Env reward** | 0 |
| **Cost** | (not recorded; higher than Run 1 due to 4× completion cap) |
| **Final checkpoint** | `tinker://185201cd-070c-584a-9298-16e14277c258:train:0/sampler_weights/final` |

**Play eval:** 0/3 games solved. Still almost every failure at `1024 tokens, stop_reason=length`. Raising the cap alone did not fix behavior learned from Run 1 / teacher rambling.

**Conclusion:** Problem is not just truncation at inference — teacher and student both fail to terminate with JSON.

---

### Run 3 — Probe: JSON-only prompt + disable thinking

| Field | Value |
|-------|-------|
| **Log dir** | `runs/probe-json-no-thinking` |
| **Steps** | 10 |
| **Renderer** | `qwen3_5_disable_thinking` |
| **max_tokens** | 1024 |
| **Prompt** | JSON-only (no reasoning section) |
| **Env reward** | 0 |
| **Final checkpoint** | See `runs/probe-json-no-thinking/checkpoints.jsonl` |

**Training signal:** `env/all/ac_tokens_per_turn` ≈ **17.7** (short JSON completions vs ~256–1024 before).

**Play eval (10-step checkpoint):** 0/3 solved, but **100% parse rate** on valid turns.

**Conclusion:** Disabling thinking + JSON-only prompt fixes format. Ready for full training.

---

### Run 4 — Full OPD: JSON + no thinking (pure KL)

| Field | Value |
|-------|-------|
| **Log dir** | `runs/mastermind-opd-json-no-thinking` |
| **Steps** | 100 |
| **Renderer** | `qwen3_5_disable_thinking` |
| **max_tokens** | 1024 |
| **Prompt** | JSON-only |
| **Env reward** | 0 |
| **kl_penalty_coef** | 1.0 |
| **Est. cost** | ~$12–15 |
| **Final checkpoint** | `tinker://4cd3d112-d453-59de-bb9a-886de2fef027:train:0/sampler_weights/final` |

**Play eval (`num_games=5`, `seed=1`):** 0/5 solved. **100% parse rate.** Model uses feedback somewhat (e.g. 3 black in one game) but repeats guesses and does not close games.

**Conclusion:** Pipeline works; distillation ceiling ≈ teacher quality, not winning.

---

### Run 5 — OPD + game reward (KL + verifiable reward)

| Field | Value |
|-------|-------|
| **Log dir** | `runs/probe-reward` (probe 10 steps, then resumed to 100) |
| **Steps** | 10 (probe) → resumed to **100** |
| **Renderer** | `qwen3_5_disable_thinking` |
| **Prompt** | JSON-only |
| **kl_penalty_coef** | 1.0 |
| **Game rewards** | `black_reward_coef=0.25`, `white_reward_coef=0.1`, `solve_reward=1.0`, `format_coef=0.1`, `repeat_guess_penalty=-0.25` |
| **Probe checkpoint** | `tinker://5fa68663-4399-59db-8bb7-3201e25f8b7e:train:0/sampler_weights/final` (batch 10) |
| **Final checkpoint** | `tinker://6d2f1f4d-ccef-54d0-833c-cf78eef459c3:train:0/sampler_weights/final` |

**Resume command used:**

```powershell
python onpolicy_mastermind.py log_path=runs/probe-reward max_steps=100 behavior_if_log_dir_exists=resume
```

**Play eval (`num_games=10`, `seed=1`):** 0/10 solved. **99/100** turns parsed (one `stop_reason=length` relapse in Game 10). More O/P exploration and higher peg scores on some turns; still repeat loops (e.g. `R,G,B,G` × 6) and weak play on repeated-color secrets.

**Conclusion:** Game reward improves guess diversity and training `env/all/reward/total` trend; does not yield solves within 10 turns under single-turn training + strong KL.

---

### Run 6 — RLVR probe (Nemotron, multi-turn)

| Field | Value |
|-------|-------|
| **Log dir** | `runs/probe-rlvr-nemotron` |
| **Script** | `rlvr_mastermind.py` |
| **Steps** | 10 |
| **Model** | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`, LoRA 64 |
| **Renderer** | `nemotron3_disable_thinking` |
| **Episodes** | Full games, `max_turns=10`, feedback-only follow-up turns |
| **kl_penalty_coef** | 0.0 (no teacher) |
| **Rewards** | `black=0.25`, `white=0.1`, `solve=1.0`, `format_penalty=0.1`, `repeat_penalty=-0.25` |

**Step 9 metrics:** `format=0.999`, `black=0.74`, `white=1.36`, `reward/total=3.21`, `correct=0.0008` (~2 solves/2551 turns), `entropy=0.217`, `ob_tokens_per_turn≈377`, `frac_mixed=1.0`.

**Conclusion:** Pipeline healthy; feedback-only design confirmed cheap (~$0.3–0.5/step). Proceeded to full run.

---

### Run 7 — RLVR full run (Nemotron, 100 steps)

| Field | Value |
|-------|-------|
| **Log dir** | `runs/mastermind-rlvr` |
| **Script** | `rlvr_mastermind.py` |
| **Steps** | 100 |
| **Config** | Same as Run 6 (`learning_rate=1e-4`, `temperature=1.0`, `group_size=4`, `groups_per_batch=64`) |
| **Final checkpoint** | `tinker://bcfd34e7-faf6-5f5b-bf6c-c5ae7d2315bb:train:0/sampler_weights/final` |

**Training progression (step 9 → 99):**

| Metric | Step 9 | Step 99 |
|--------|--------|---------|
| `mastermind/black` | 0.74 | 0.97 |
| `reward/total` | 3.21 | 3.79 |
| `correct` (per turn) | 0.0008 | 0.0024 (~2.3% of games solved in training) |
| `optim/entropy` | 0.217 | **0.036** (collapsed) |
| `turns_per_episode` | 9.96 | 9.95 |

First run ever to produce solves (OPD runs and teacher: 0). But entropy collapsed 6× — policy became nearly deterministic and plateaued.

**Play eval (`num_games=3`):** 0/3 solved. Format perfect. **Key finding: the policy never guesses repeated colors** — all 30 guesses across 3 games used 4 distinct colors, while all 3 secrets contained duplicates (`Y,B,Y,B`, `R,O,P,O`, `Y,P,O,Y`). With repeats allowed, ~72% of secrets contain a duplicate (all-distinct probability = 360/1296 ≈ 28%), so this policy caps out near the ~2% training solve rate. Also still occasionally repeats an exact previous guess (Game 3, turns 2 and 7) and opens every game with `R,G,B,Y` (deterministic).

**Conclusion:** RLVR + multi-turn works (real reward improvement, first solves), but `lr=1e-4` drove premature entropy collapse into a distinct-colors-only strategy. Next: lower LR / higher temperature, possibly warm-start from this checkpoint; consider a reasoning model (GPT-OSS-20B) if peg-counting logic stays weak.

---

### Run 8 — RLVR round 2 (warm-start, lower LR, repeated-colors prompt)

| Field | Value |
|-------|-------|
| **Log dir** | `runs/mastermind-rlvr-2` |
| **Script** | `rlvr_mastermind.py` |
| **Steps** | 100 |
| **Warm-start** | Run 7 `weights/final` (note: must use `weights/`, not `sampler_weights/` — sampler checkpoints cannot be loaded for training) |
| **Config** | `learning_rate=3e-5`, `temperature=1.2`, `max_turns=10` |
| **Prompt** | Added: secrets often repeat colors + guesses may reuse a color |
| **Final checkpoint** | `tinker://af9a65db-23d6-5de3-b671-50af4e944a1f:train:0/sampler_weights/final` |

**Training (step 99):** `black=1.42`, `reward/total=4.59`, `correct=0.028` (~27% of training games solved), `entropy=0.026` (still collapsed), `turns_per_episode=9.4`.

**Play evals (all `seed=1`):**

| Eval setup | Result |
|-----------|--------|
| `prompt_mode=full`, temp 0.7, 10 turns, 10 games | **0/10** — train/eval conversation mismatch |
| `prompt_mode=rlvr`, temp 1.2, 10 turns, 10 games | **1/10** |
| `prompt_mode=rlvr`, temp 1.2, **15 turns**, 20 games | **10/20** — first real wins |

Key findings: the eval had to match the training conversation format (`play_student.py` now supports `prompt_mode=rlvr`, full rules on turn 1 then feedback-only user turns — this is the default). Several wins landed on turns 11–15, so the 10-turn cap was hiding real skill. The repeated-colors prompt fix worked: the model now guesses duplicates and solved duplicate secrets (`R,P,O,R`, `Y,P,G,Y`, `Y,Y,O,G`, `Y,O,Y,O`). Remaining failures concentrated on triple-color secrets (`R,Y,Y,Y`, `R,B,R,R`, `P,B,P,P`).

**Conclusion:** Best checkpoint of the project: **50% solve rate** with aligned eval. Two separate issues had masked progress: checkpoint-type confusion (`weights/` vs `sampler_weights/`) and eval prompt mismatch.

---

### Run 9 — RLVR round 3 (50-step continuation, max_turns=15)

| Field | Value |
|-------|-------|
| **Log dir** | `runs/mastermind-rlvr-3` |
| **Script** | `rlvr_mastermind.py` |
| **Steps** | 50 |
| **Warm-start** | Run 8 `weights/final` |
| **Config** | `max_turns=15` (now matching eval), `learning_rate=1e-5`, `temperature=1.4` |
| **Final checkpoint** | `tinker://77d1efc2-5d95-57d9-ad13-05afe28a760f:train:0/sampler_weights/final` |

**Note:** First attempt crashed at step 7 with `UnicodeEncodeError` — Windows `cp1252` default encoding when the cookbook wrote logtree HTML. Fixed by forcing UTF-8 in `logtree._write_trace` (patched in site-packages and monkeypatched in `rlvr_mastermind.py`); also lowered default `save_every` to 10 and exposed `num_groups_to_log`.

**Training (step 49):** `black=1.62`, `reward/total=6.76`, `correct=0.040` (~50% of training games solved), `entropy=0.13` (stayed healthy — no collapse), `turns_per_episode=12.4`.

**Play eval (`prompt_mode=rlvr`, temp 1.2, 15 turns, 20 games, `seed=1`):** **8/20** — statistically equivalent to Run 8's 10/20. Solves are faster (avg ~9.5 turns vs ~10.7; five solves in ≤9 turns), but no more games converted.

**Key finding — the triple-color blind spot:** In Game 2 (secret `R,Y,Y,Y`) the model reached black=3 on turn 5 and spent 10 turns enumerating every variant *except* `R,Y,Y,Y`. Same pattern on `Y,R,Y,Y`, `R,B,R,R`, `P,B,P,P`, `Y,O,Y,O`. The policy has a hard bias against guessing 3–4 of the same color; such secrets are rare in training (~6%), so reward pressure never fixes it.

**Conclusion:** Recipe has converged at **~40–50% play solve rate**. Aggressive duplicate bias (run 10) overcorrected to 2/20; remaining options are milder bias/curriculum, solver-based reward shaping, or a reasoning model.

---

### Run 10 — RLVR round 4 (duplicate-biased training secrets) — negative result

| Field | Value |
|-------|-------|
| **Log dir** | `runs/mastermind-rlvr-4` |
| **Script** | `rlvr_mastermind.py` |
| **Steps** | 50 |
| **Warm-start** | Run 8 `weights/final` (`tinker://af9a65db-.../weights/final`) |
| **Config** | `max_turns=15`, `learning_rate=1e-5`, `temperature=1.4` |
| **Train secrets** | Duplicate-biased mixture: 50% uniform / 25% triple / 15% double-pair / 10% quad (test secrets stay uniform) |
| **Final checkpoint** | `tinker://78409305-ec77-5b06-8570-1943667b89ca:train:0/sampler_weights/final` |

**Training (step 49):** Harder distribution shows up clearly — `correct=0.0087` (~3% of training games solved vs ~50% on uniform in run 9), `black=1.34`, `reward/total=6.14`, `turns_per_episode=14.4`, `entropy=0.089`. Low train solve rate is expected when ~40% of secrets are triple/quad/double-pair.

**Play eval (`prompt_mode=rlvr`, temp 1.2, 15 turns, 20 games, `seed=1`):** **2/20** — regression vs run 8 (10/20) and run 9 (8/20).

| Outcome | Detail |
|---------|--------|
| Solved | Game 10 `P,R,O,G` (11 turns); Game 19 `Y,O,Y,O` (9 turns) — double-pair win matches the bias |
| Triple blind spot | Still fails: `R,Y,Y,Y`, `Y,R,Y,Y`, `R,B,R,R`, `P,B,P,P` — never commits to a third copy of a color |
| Skill erosion | Previously solved secrets now lost: `G,O,R,B`, `B,P,G,O`, `R,Y,G,P`, `O,R,P,Y`, `R,P,O,R`, `Y,P,G,Y` |

**Conclusion:** Strong duplicate bias **overcorrected**. The policy partially learned double-pairs (Game 19) but lost the general play that made run 8 strong, and still does not close triples. Keep **run 8** as the best checkpoint. If retrying bias, use a milder mixture (e.g. `frac_uniform=0.8`) or a curriculum (biased then back to uniform) — not this 50/25/15/10 split as a 50-step fine-tune.

---

### Run 11 — RLVR round 5 (turn-efficiency group rewards)

| Field | Value |
|-------|-------|
| **Log dir** | `runs/mastermind-rlvr-5` |
| **Script** | `rlvr_mastermind.py` |
| **Steps** | 50 |
| **Warm-start** | Run 8 `weights/final` (`tinker://af9a65db-.../weights/final`) |
| **Config** | `max_turns=15`, `learning_rate=1e-5`, `temperature=1.4`, **uniform secrets**, `turn_efficiency_coef=0.5` |
| **Group reward** | Pairwise within each group of 4: solvers beat non-solvers; among solvers, fewer turns wins (zero-sum; mean `turn_efficiency` ≈ 0 by construction) |
| **Final checkpoint** | `tinker://732d0e16-4f09-5f8f-8ba6-fd72a6411cb6:train:0/sampler_weights/final` |

**Training (step 49):** `black=1.45`, `reward/total=6.74`, `correct=0.015`, `group_solved=0.215`, `turns_per_episode=14.0`, `entropy=0.18` (healthy), `turn_efficiency=0.0` (expected for zero-sum pairwise scores).

**Play eval (`prompt_mode=rlvr`, temp 1.2, 15 turns, 20 games, `seed=1`):** **10/20** — tied with run 8; better than run 9 (8/20) and run 10 (2/20).

Notable wins include fast solves (Game 6 in 6, Game 20 in 7, Game 10 in 8) and duplicate secrets (`R,P,O,R`, `Y,P,G,Y`, `Y,O,Y,O`, `O,Y,O,P`). Triple/quad blind spot unchanged: `R,Y,Y,Y`, `Y,R,Y,Y`, `R,B,R,R`, `P,B,P,P` still fail (black=3 stalls without guessing the repeated color).

**Conclusion:** Turn-efficiency is **safe but not a breakthrough** — matches run 8 without the skill erosion of run 10. Speeds up some wins; does not teach closing hard duplicates (no group signal when nobody solves). Run 8 and run 11 are both best-so-far; prefer run 8 as the simple baseline.

---

### Run 12 — Knuth SFT (algorithm distillation) — breakthrough

| Field | Value |
|-------|-------|
| **Log dir** | `runs/mastermind-knuth-sft-1` |
| **Scripts** | `knuth_solver.py` (data) → `sft_mastermind.py` (train) |
| **Data** | 5000 Knuth full games (`data/knuth_trajectories.jsonl`) + deductive \|S\|=1 slices (`data/knuth_deductive.jsonl`); expanded to one SFT example per assistant turn |
| **Steps** | 1611 (2 epochs, batch 32, `test_size=64`) |
| **Warm-start** | Run 8 `weights/final` (`tinker://af9a65db-.../weights/final`) |
| **Config** | `learning_rate=1e-4`, `lora_rank=64`, `max_length=2048`, `train_on_what=LAST_ASSISTANT_MESSAGE`, `expand_per_assistant_turn=True`, `nemotron3_disable_thinking` |
| **Final checkpoint** | `tinker://0e95f869-1abb-5baa-baea-f456a8ecf82e:train:0/sampler_weights/final` |

**Training note:** First attempt used `ALL_ASSISTANT_MESSAGES` on full multi-turn chats and hit Tinker's sequence-extension warning (`nemotron3_disable_thinking` strips thinking from history). Fixed by expanding each game into per-turn prefixes and supervising only the last assistant message. Final `train_mean_nll≈0.0002`.

**Play eval (`prompt_mode=rlvr`, temp 1.2, 15 turns, `seed=1`):**
- **20/20** on the original suite (avg ~4.7 turns).
- **99/100** on a larger eval (mean **4.55** turns among solves; min 3, max 9; 1 fail: Game 79 `P,O,Y,O`). Most games finish in 4–5 turns.
- Temp probe **1.0**: **97/100**, mean 4.46 among solves — slightly faster but worse reliability; keep **temp=1.2**.

Former blind spots all closed on the 20-game suite: Game 2 `R,Y,Y,Y` (4), Game 4 `Y,R,Y,Y` (4), Game 7 `R,B,R,R` (4), Game 16 `P,B,P,P` (4).

**Conclusion:** Distilling Knuth beats sparse RLVR reward shaping for the closing/triple failure mode. **Best checkpoint** — prefer this over any later RLVR-from-SFT attempt (see run 13).

---

### Run 13 — RLVR-for-speed from Knuth SFT — negative result

| Field | Value |
|-------|-------|
| **Log dir** | `runs/mastermind-rlvr-knuth-speed-1` |
| **Script** | `rlvr_mastermind.py` |
| **Steps** | 50 |
| **Warm-start** | Run 12 `weights/final` (`tinker://0e95f869-.../weights/final`) |
| **Config** | `max_turns=15`, `learning_rate=1e-5`, `temperature=1.4`, `turn_efficiency_coef=0.5` |
| **Final checkpoint** | `tinker://c8bd5f41-9d87-5eb3-8f3d-a7adb64fda9d:train:0/sampler_weights/final` |

**Training (step 49):** Warning signs — `group_solved=0.19`, `turns_per_episode=13.3`, `black=1.07`, `reward/total=5.1`, **`entropy=2.05`** (was ~0.2 under healthy RLVR / sharp SFT), format 0.97. Turn-efficiency mean 0 by construction.

**Play eval (`prompt_mode=rlvr`, temp 1.2, 15 turns, 100 games, `seed=1`):** **21/100** — catastrophic regression vs run 12 (99/100). Failures show inconsistent search, invalid colors, parse errors, and repeat loops — not faster Knuth play.

**Conclusion:** Naive RLVR-for-speed on a strong Knuth SFT **destroys** the distilled policy (high train entropy, low train/play solve rate). Keep **run 12**. Do not retry the same LR/temp/efficiency recipe; if revisiting speed, use a better-average teacher + SFT, or much gentler RL with early stop on solve-rate.

---

## Teacher benchmark (ceiling)

**Command:**

```powershell
python play_student.py model_name=Qwen/Qwen3.5-9B num_games=5 seed=1
```

| Setting | Result |
|---------|--------|
| Same JSON prompt + `qwen3_5_disable_thinking` | **0/5 solved** |
| Parse failures | Rare (1 ramble / length in 5 games with thinking renderer earlier; mostly OK with disable thinking) |
| Behavior | Long reasoning chains; same peg-scoring weakness as student |

**Conclusion:** Student cannot surpass teacher via pure distillation. Teacher is not a strong Mastermind player on this task format.

---

## Evaluation summary

| Run | Checkpoint (sampler) | Parse rate | Solved | Notes |
|-----|----------------------|------------|--------|-------|
| 1 Baseline | `9e78e639-.../final` | ~50% | 0/3 | Truncation at 256 |
| 2 Retrain 1024 | `185201cd-.../final` | ~50% | 0/3 | Still truncates at 1024 |
| 3 Probe JSON | `probe-json-no-thinking` | 100% | 0/3 | Format fixed |
| 4 Full JSON/KL | `4cd3d112-.../final` | 100% | 0/5 | Best pure-OPD student |
| 5 KL + reward | `6d2f1f4d-.../final` | ~99% | 0/10 | More exploration, still no wins |
| 7 RLVR Nemotron | `bcfd34e7-.../final` | 100% | 0/3 | Never guesses repeated colors; ~2.3% train solve rate |
| 8 RLVR round 2 | `af9a65db-.../final` | 100% | **10/20** | Best baseline; aligned eval |
| 9 RLVR round 3 | `77d1efc2-.../final` | 100% | **8/20** | Same eval; faster solves, no net gain |
| 10 Dup-biased | `78409305-.../final` | 100% | **2/20** | Overcorrected; eroded easy wins |
| 11 Turn-efficiency | `732d0e16-.../final` | 100% | **10/20** | Tied with run 8; triples still fail |
| 12 Knuth SFT | `0e95f869-.../final` | 100% | **20/20** / **99/100** | Best; mean ~4.55 turns @100 |
| 13 RLVR speed on SFT | `c8bd5f41-.../final` | degraded | **21/100** | Destroyed Knuth policy |
| Teacher | (no checkpoint) | ~99% | 0/5 | Ceiling reference |

**Eval settings:** Runs 1–7: `max_turns=10`, `temperature=0.7`, full prompt each turn. Runs 8–13: `prompt_mode=rlvr`, `temperature=1.2`, `max_turns=15`, `seed=1` (100-game evals for runs 12–13).

---

## Key metrics (training)

| Metric | Meaning | Healthy signal |
|--------|---------|----------------|
| `teacher_kl` | Reverse KL student vs teacher | Down over steps |
| `env/all/ac_tokens_per_turn` | Avg completion length | ~15–25 after JSON fix (was 256–1024) |
| `env/all/reward/total` | Mean env reward per rollout | Noisy upward trend (Run 5); was ~0 in pure OPD |
| `optim/entropy` | Sampling entropy | Decreases as policy sharpens |

Rewards are **centered within each group of 4 samples** per prompt before combining with KL advantages.

---

## Lessons learned

1. **Format vs strategy are different problems.** JSON + `qwen3_5_disable_thinking` solved output format; winning requires optimizing game outcome, not imitating a weak teacher.

2. **Truncation poisoned early runs.** Training with `max_tokens=256` taught endless reasoning; fixing inference cap alone is insufficient without retraining on a terminating prompt.

3. **Teacher benchmark is essential.** 0/5 teacher solves explains 0/5–0/10 student solves under pure KL.

4. **Single-turn training ≠ multi-turn play.** Training scores one guess from a random synthetic history; `play_student.py` runs up to 10 sequential guesses with growing real history.

5. **Game reward helps but competes with KL.** Mixed signal improved peg scores and exploration but did not produce solves with `kl_penalty_coef=1.0`.

6. **RLVR multi-turn beats OPD, but watch entropy.** Run 7 produced the first solves of the project, then plateaued as entropy collapsed (0.22 → 0.04 over 100 steps at `lr=1e-4`). A near-deterministic policy that never guesses repeated colors can solve at most ~28% of secrets in principle and ~2% in practice.

7. **Evaluate in the training conversation format.** Run 8 scored 0/10 with the old full-prompt eval but 10/20 once `play_student.py` replicated the RLVR format (full rules turn 1, feedback-only after) with matching temperature and `max_turns=15`. Half the apparent failure was measurement error.

8. **Checkpoint types matter.** `weights/...` (state_path) is for training warm-starts; `sampler_weights/...` (sampler_path) is for play/sampling only. Loading a sampler checkpoint for training fails with a Tinker error.

9. **Pure RLVR converged at ~40–50% solves with a specific blind spot.** Runs 8/9/11 failed on triple-color secrets (`R,Y,Y,Y`-style): black=3 then never commit to the triple. More identical RLVR steps alone did not fix this — **Run 12 (Knuth SFT) did.**

10. **Windows: force UTF-8 for HTML log writes.** The cookbook's logtree HTML writer crashed mid-run with `UnicodeEncodeError` under the default `cp1252` codec. Patched to `encoding="utf-8"`; keep `save_every` small enough that a crash doesn't lose all checkpoints.

11. **Aggressive duplicate bias overcorrects.** Run 10 (50% uniform / 25% triple / 15% double-pair / 10% quad) regressed to 2/20: one double-pair win, triples still unsolved, and many previously solved uniform secrets lost. Exposure to hard secrets ≠ learning the closing move; cutting uniform practice hurts the skills that already worked. Prefer mild bias, curriculum, or reward shaping over a hard 50% non-uniform mix.

12. **Turn-efficiency group rewards are safe, not transformative.** Run 11 (`turn_efficiency_coef=0.5`, uniform secrets) tied run 8 at 10/20 with some faster wins, but the triple-color blind spot remained. Pairwise bonuses only reshape ranking among rollouts that already solve; they cannot invent a closing strategy nobody in the group attempts.

13. **Closing-move dense rewards are too sparse.** Run 6 (`close_move_coef=0.5`) barely fired in training (`close_move≈0.0016`) and play regressed to 9/20 — the policy never sampled the triple, so the bonus could not teach it.

14. **Knuth SFT fixes the blind spot.** Run 12 (5000 Knuth games + deductive slices, per-turn `LAST_ASSISTANT_MESSAGE` SFT from run 8) scored **20/20** then **99/100** on the same protocol, including prior triple/quad failures. Dense correct demos beat hoping GRPO invents elimination. For Nemotron disable-thinking, expand multi-turn chats into per-assistant-turn examples — do not use `ALL_ASSISTANT_MESSAGES` (sequence-extension mismatch).

15. **RLVR-for-speed on Knuth SFT can erase the skill.** Run 13 (50 steps, `lr=1e-5`, train temp 1.4, `turn_efficiency_coef=0.5` from run 12) collapsed to **21/100** with entropy exploding (~2.0). Efficiency rewards need solvers in-group *and* must not overpower a sharp cloned policy; this recipe did not.

---

## Reproducing runs

**Activate environment:**

```powershell
.\onpolicy\Scripts\Activate.ps1
$env:TINKER_API_KEY = "your-api-key-here"
```

**Train (current defaults: JSON prompt, disable thinking, game rewards, 100 steps):**

```powershell
python onpolicy_mastermind.py log_path=runs/my-run
```

**Play a checkpoint:**

```powershell
python play_student.py model_name=Qwen/Qwen3.5-9B-Base log_dir=runs/my-run num_games=5 seed=1
```

**Play teacher baseline:**

```powershell
python play_student.py model_name=Qwen/Qwen3.5-9B num_games=5 seed=1
```

**RLVR train (Nemotron defaults; warm-starts from run 7 unless overridden):**

```powershell
python rlvr_mastermind.py log_path=runs/my-rlvr-run
```

**Play an RLVR / Knuth-SFT checkpoint (aligned eval used for runs 8–12):**

```powershell
python play_student.py `
  model_name=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 `
  renderer_name=nemotron3_disable_thinking `
  log_dir=runs/my-rlvr-run `
  num_games=20 seed=1 temperature=1.2 max_turns=15
```

**Dump Knuth distillation data:**

```powershell
python knuth_solver.py --num-games 5000 --seed 42 `
  --out data/knuth_trajectories.jsonl `
  --deductive-out data/knuth_deductive.jsonl
```

**Knuth SFT (run 12 recipe):**

```powershell
python sft_mastermind.py log_path=runs/mastermind-knuth-sft-1 `
  extra_file_path=data/knuth_deductive.jsonl
```


---

## Suggested next experiments (not yet run)

1. **Better-average teacher + SFT** — regenerate trajectories with expected-remaining / entropy-minimizing guess selection (often slightly better avg than classic Knuth minimax), then SFT from run 8 or fresh; aim to cut mean turns below ~4.55 without RL.
2. **Upweight fast traces in SFT** — filter or reweight Knuth demos toward ≤4-turn games; keep run 12 as baseline.
3. **Ultra-gentle RL only if needed** — tiny LR, few steps, low `turn_efficiency_coef`, early stop on play solve-rate; do **not** repeat run 13.
4. **Deductive-only probe** — \|S\|=1 histories (MastermindEval paradigm) on run 12.

---

## File layout

```
Tinker_MasterMind/
├── onpolicy_mastermind.py   # OPD training
├── rlvr_mastermind.py       # RLVR training (multi-turn)
├── knuth_solver.py          # Knuth teacher + JSONL dump
├── sft_mastermind.py        # Knuth SFT
├── play_student.py          # Evaluation / play
├── data/
│   ├── knuth_trajectories.jsonl
│   └── knuth_deductive.jsonl
├── onpolicy/                # Python venv
└── runs/
    ├── mastermind-opd-Qwen-Qwen3.5-9B-Base-...-09-16/   # Run 1
    ├── mastermind-opd-Qwen-Qwen3.5-9B-Base-...-19-14/   # Run 2
    ├── probe-json-no-thinking/                          # Run 3
    ├── mastermind-opd-json-no-thinking/                 # Run 4
    ├── probe-reward/                                    # Run 5
    ├── probe-rlvr-nemotron/                             # Run 6
    ├── mastermind-rlvr/                                 # Run 7
    ├── mastermind-rlvr-2/                               # Run 8
    ├── mastermind-rlvr-3/                               # Run 9
    ├── mastermind-rlvr-4/                               # Run 10 (dup-biased; 2/20)
    ├── mastermind-rlvr-5/                               # Run 11 (turn-efficiency; 10/20)
    ├── mastermind-rlvr-6/                               # close-move probe (9/20; not a win)
    ├── mastermind-knuth-sft-1/                          # Run 12 (Knuth SFT; 20/20, 99/100)
    └── mastermind-rlvr-knuth-speed-1/                   # Run 13 (speed RLVR; 21/100 — fail)
```

Each run directory contains `config.json`, `checkpoints.jsonl`, `metrics.jsonl`, and `logs.log`.
