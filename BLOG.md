# Teaching an LLM to Crack Mastermind (and Explain Itself)

*A short write-up of experiments on the classic codebreaking game — from Knuth’s five-guess algorithm to a LoRA-fine-tuned Nemotron that plays in a browser UI.*

**Acknowledgments:** This research was supported by a Grant from Thinking Machines Lab.

---

## A Christmas classic with teeth

If you grew up in the late 1970s or early 1980s, you may remember **Mastermind** under the tree: a plastic board, coloured pegs, and a cardboard shield to hide the secret. Invented by Mordecai Meirowitz and released by Invicta Plastics in 1972, it became a huge Christmas present in the mid-to-late seventies — equal parts family game and miniature logic puzzle.

The rules are simple enough for a child and deep enough for a computer scientist.

**How you play (standard version we used):**

- One player (the *codemaker*) chooses a secret code of **4 pegs**.
- Each peg is one of **6 colours** (we use `R, G, B, Y, O, P`).
- **Repeats are allowed** — so the universe has \(6^4 = 1{,}296\) possible codes.
- The other player (the *codebreaker*) proposes a guess each turn.
- Feedback comes as **black** and **white** key pegs:
  - **Black** — right colour in the **right** position.
  - **White** — right colour in the **wrong** position (no double-counting).
- The goal is to find the secret in as few guesses as possible.

Under that friendly plastic board sits a real deduction problem: maintain the set of codes still consistent with every clue, and choose the next probe carefully.

---

## Harder than it looks

Mastermind is not just nostalgia. In a 2005 paper (*Mastermind is NP-Complete*, Stuckman & Zhang; arXiv:cs/0512049, December 2005), researchers showed that the related **Mastermind Satisfiability Problem** — given a list of guesses and their black/white scores, does *any* secret still fit? — is **NP-complete**. So the “is this history consistent?” question is already computationally hard in the general case.

Separately, the famous computer scientist **Donald Knuth** (1977) gave a practical algorithm for the classic game: keep the set \(S\) of codes consistent with the feedback so far, and at each step pick a guess that **minimises the worst-case size of \(S\)** after the next reply (with a sensible opener such as two pairs of colours). For the standard 4-peg / 6-colour game, Knuth’s strategy guarantees a solution in **at most five guesses**.

That gap — NP-hard satisfiability in theory, a five-guess algorithm in practice for the small classic board — is exactly what makes Mastermind a good stress test for language models.

---

## Why bother with LLMs?

In 2025, **MastermindEval** (Golde, Haller, Barth & Akbik; ICLR 2025 Workshop on Reasoning and Planning for LLMs) argued that Mastermind is a simple but scalable benchmark for **deductive reasoning**. Their finding, bluntly: **current LLMs struggle**, even on relatively easy instances. Performance drops as the model must combine more hints; integrating several black/white statements into one forced conclusion is where many models fall apart.

That lined up with what I saw in my own runs.

### What I tried first (and what failed)

Using [Tinker](https://thinkingmachines.ai/) / the Tinker cookbook stack, I first tried:

1. **On-policy distillation** from a larger “teacher” LLM — the teacher itself could not reliably solve Mastermind, so the student learned little strategy.
2. **RL with verifiable rewards** (black / white / solve) on full multi-turn games — this got the first real wins (~50% solve rate) but **plateaued**. A stubborn blind spot remained: codes with **triple or quadruple repeats** (e.g. `R,Y,Y,Y`). The policy would often reach *black = 3* and then never commit to the repeated colour.

Sparse rewards never taught a closing move the model never sampled.

---

## Distilling Knuth instead of hoping the model invents him

The breakthrough was **algorithm distillation**:

1. Implement Knuth’s minimax codebreaker.
2. Generate **5,000 full solved games** (plus some “deductive” slices where only one code remains).
3. Expand each game into **one supervised example per turn** (so the model learns: *given this history, emit the next JSON guess*).
4. **LoRA fine-tune** (`rank = 64`) **NVIDIA-Nemotron-3-Nano-30B-A3B-BF16**, warm-started from the best RL checkpoint so format and multi-turn chat were already in place.

This is **not** on-policy RL for the final policy. It is supervised fine-tuning on expert trajectories — dense, correct demonstrations every turn.

**Result (same eval protocol, temperature 1.2, up to 15 turns):**

- **20/20** on the original suite  
- **99/100** on a larger eval (mean ~**4.55** turns among solves)  
- Former triple/quad blind spots closed  

Trying to “speed up” that policy with more RL afterwards **destroyed** it (solve rate collapsed). The lesson I took away: once you have cloned a strong algorithm, treat further RL carefully — or prefer a better teacher and more SFT.

---

## Putting it in an app

I built a small **React** Mastermind board backed by a FastAPI server that samples from the fine-tuned checkpoint via Tinker:

- You set the secret (four coloured pegs).
- The **fine-tuned Nemotron** plays as codebreaker — guessing in as few turns as it can.
- When the game is solved, a **second LLM call** (same fine-tuned model, separate prompt) writes a short table: **Turn → Strategy → What it proves** — a post-hoc explainer of the deduction, not part of the guessing policy itself.

That separation matters. MastermindEval and my early runs both suggest that *playing* and *narrating* are different skills. Keeping guesses as tight JSON (`{"guess":["R","Y","G","P"]}`) and explaining *after* the solve keeps the codebreaker sharp while still giving a human-readable story.

---

## How this fits MastermindEval (ICLR 2025)

My experiments sit next to MastermindEval’s message:

| Theme | MastermindEval | This project |
|--------|----------------|--------------|
| LLMs find Mastermind hard | Even easy instances hurt | Pure OPD / weak teachers → 0 solves; RLVR plateaus ~50% |
| Multi-hint deduction is brittle | Solve rate drops as hints to combine grow | Triple-colour “almost solved” states needed expert demos |
| Agentic play is a real test | Model must act turn by turn | Full multi-turn eval + interactive UI |
| Scaling the challenge | Longer codes / more colours | Classic 4×6 board, but algorithm distillation as a remedy |

Where I diverge is the remedy: instead of only measuring the failure, I **injected Knuth’s algorithm** into the model via LoRA SFT. The UI is the demo that the resulting policy can both **play** and (separately) **explain**.

---

## Takeaways

1. Mastermind is culturally light and computationally serious — Christmas toy, Knuth’s five-guess algorithm, and NP-complete satisfiability in the same story.
2. LLMs really do struggle with the multi-step elimination MastermindEval highlights.
3. **Distilling a known optimal-ish algorithm** (5k Knuth games + LoRA on Nemotron) beat hoping sparse RL would invent the closing move.
4. A small app — user sets the code, the fine-tuned model breaks it, then narrates the deduction — makes the research tangible.

---

Please see experiments.md for detailed information of my experiments. 

## Acknowledgments

**Acknowledgments:** This research was supported by a Grant from Thinking Machines Lab. Training and sampling used the Tinker stack; the student policy is a LoRA adaptation of `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`.

---

## References (short)

- Donald E. Knuth (1977). *The computer as Master Mind.* Journal of Recreational Mathematics.
- Jeff Stuckman & Guo-Qiang Zhang (2005). *Mastermind is NP-Complete.* arXiv:cs/0512049.
- Jonas Golde, Patrick Haller, Fabio Barth & Alan Akbik (2025). *MastermindEval: A Simple But Scalable Reasoning Benchmark.* ICLR 2025 Workshop on Reasoning and Planning for LLMs. [OpenReview](https://openreview.net/forum?id=H4donosutm) · [arXiv:2503.05891](https://arxiv.org/abs/2503.05891)

---

*Draft notes for editing: add screenshots of the UI (secret picker + board + “Why those guesses” table); link to the demo if public; optionally mention that the NP-completeness result is for the satisfiability decision problem, not “playing optimally in five moves,” which Knuth already settled for the classic board.*
