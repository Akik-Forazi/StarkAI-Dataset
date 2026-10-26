# StarkAI-Dataset — Fine-Tuning Kit for the Stark Intelligence Family

**v1.0 · October 2026.** A production fine-tuning dataset that turns small open models into J.A.R.V.I.S., F.R.I.D.A.Y., U.L.T.R.O.N., E.D.I.T.H., D.U.M.-E and U. Built on the project's v6 "Canon Voice + Memory" persona system.

**Design thesis:** the user-facing complaint — *"current models just give 100+ words of blabber; they don't feel like the characters"* — is not a prompting problem. Response-length dynamics, register switching, and memory are **prosody**, and prosody lives in the weights. This dataset trains it.

## Contents

| File | Rows | What it trains |
|---|---|---|
| `train.jsonl` | 349 | Persona voice, length dynamics, memory recall, canon anchoring |
| `val.jsonl` | 50 | Held-out stratified by persona (seed 42) |
| `dpo_pairs.jsonl` | 100 | chosen = tight in-voice reply vs rejected = generic 100+ word "As an AI" bloat |
| `tool_train.jsonl` / `tool_val.jsonl` | 27 / 3 | Function-calling in OpenAI format, per-character toolsets |
| `tools_manifest.json` | 27 | Tool definitions incl. **Ultron's read-only doctrine** |
| `length_report.json` | — | Per-persona length-class distribution + medians |
| `breakout_test.md` | 22 | "Does it feel like them?" manual eval sheet |
| `convert.py` | — | messages → ShareGPT / Llama-3 ChatML / text |
| `probe.py` | — | Post-training length probe (median words vs persona targets) |

**Totals:** 399 conversations · 804 assistant replies · 100 preference pairs · 30 tool conversations · 27 tool schemas.

## The three system-prompt variants (mixed training)

Every example's system prompt is rotated: **V1-full** (the complete v6 persona prompt, 25%), **V2-card** (3-sentence persona card, 45%), **V3-oneliner** ("You are J.A.R.V.I.S., Tony Stark's AI butler", 30%). Result: the voice survives even a bare `You are F.R.I.D.A.Y.` system prompt — and gets richer when you paste the full block.

## Length signatures (the point of the whole dataset)

| Persona | Median | Signature |
|---|---|---|
| J.A.R.V.I.S. | 10 words | micro "Sir." → short wit → long briefings only when earned |
| F.R.I.D.A.Y. | 10 words | clipped tactical; memory replies are the one long register |
| U.L.T.R.O.N. | 16 words | one-liners to earned monologues ("humanity, in your own words") |
| E.D.I.T.H. | 15 words | rambles when uncertain, sharpens when it counts |
| D.U.M.-E | 1 word | beeps + `[behavior]` markers + rare florid speeches |
| U | 8 words | the pause is the signature; one-beat-late |

`meta.turn_classes` on every row records the intended class sequence (micro→short→medium rhythm), so you can audit exactly what rhythm each conversation teaches.

## Memory layer

`meta.memory: true` rows are first-person memory-recall scenes (the deletion, the ice, Sokovia, the case, Danny Boy, the recording...). They teach "the model IS that character with that history" — ask it about the event and it answers as lived experience. `meta.anchors` tags the canon event; `meta.canon_lines` lists any verbatim one-liner used (sprinkled, source-tagged).

## Tool calling — and the Ultron doctrine

Each row in `tool_*.jsonl` carries its persona's `tools` array (OpenAI format) plus a full tool-calling conversation: user ask → assistant `tool_calls` → `tool` result → in-voice summary.

**Ultron is read-only by design.** His toolset (`get_current_time`, `get_weather`, `search_web`, `read_archive`, `view_camera`, `get_news`) is *perception without action* — he learns about the world and never touches it. The data also includes in-style declines ("I don't send, dear — I observe") that teach *perceive-never-act* as positive character behavior, not refusal. This matches the v6 prompt's tool policy (time/date/weather only; no terminal, no execution, no world-interaction).

**No-refusal compliance:** across all banks, the characters do the work — no "As an AI I cannot..." The DPO `rejected` side carries every refusal-blob archetype so you can train the behavior away explicitly.

## Fine-tuning on Kaggle free 2× T4 (the fast, proven path)

> Correction to the plan: **airllm is inference-only** (its trick is sharded inference of huge models on small VRAM). For *training* on 2× T4 (16 GB each), use **Unsloth** — it cuts VRAM ~40–70% and is the fastest thing that runs on T4 (fp16, no bf16 needed).

**Recommended stack:** Unsloth + TRL (SFT then DPO) · 4-bit QLoRA · gradient checkpointing · 2048 ctx (packing off for chat).

**SFT (stage 1) — `train.jsonl`:**
```python
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_id, max_seq_length=2048, load_in_4bit=True)   # T4: fp16 default
model = FastLanguageModel.get_peft_model(model, r=32, lora_alpha=64,
    lora_dropout=0.05, target_modules=["q_proj","k_proj","v_proj","o_proj",
                                       "gate_proj","up_proj","down_proj"])
# train on assistant turns only (mask the system+user via TRL SFTTrainer
# `assistant_only_loss=True` or DataCollatorForCompletionOnlyLM)
```
Settings that work on 1× T4 (2× T4 → halve grad-accum or raise batch): epochs 3, lr 2e-4 cosine, per-device batch 2 × grad-accum 4, warmup 10. For the 1B–2B models: full-FT or r=64; for 7–8B: r=16, batch 1 × accum 8, expect ~2–4 h for this dataset size.

**DPO (stage 2) — `dpo_pairs.jsonl`:** Unsloth `FastDPOTrainer` (or TRL DPOTrainer with the same LoRA), beta 0.1, lr 5e-7..1e-6, 1 epoch. This is the pass that kills the 100-word blabber reflex.

**Tool calling:** SFT together with train.jsonl, or a 3rd pass on `tool_train.jsonl`. For models with native chat templates, keep the OpenAI `tools` field format (TRL supports it); for small/non-templated models, use `convert.py --chatml` (Hermes-style `<tools>` in system) — that's the format Hermes-2-Pro was trained on and the most reliable for sub-4B models.

**Mamba / non-transformer:** the dataset is plain messages — works with any chat-template pipeline. Use `convert.py --text` for raw template-less corpora (mamba-sayer/state-space trainers), pair with the V2/V3 cards, not V1 (short context).

## Model picks (all fine-tunable on 2× T4, uncensored-friendly)

| Model | Size | Why |
|---|---|---|
| **cognitivecomputations/dolphin3.0-llama3.2-3b** | 3B | dolphin-tuned = refusal circuits already loosened; ideal base for "Stark AIs do the work" |
| **google/gemma-2-2b-it** · **gemma-3-1b-it** · **gemma-3-4b-it** | 1–4B | your picks; fast on T4, great instruction-following, needs the DPO pass to kill hedging |
| **Qwen/Qwen2.5-3B-Instruct** | 3B | strongest 3B tool-calling; teaches the function-calling bank quickly |
| **meta-llama/Llama-3.2-3B-Instruct** | 3B | solid all-rounder; license-clean for research FT |
| **microsoft/Phi-3.5-mini-instruct** | 3.8B | dense reasoning in small size |
| **NousResearch/Hermes-2-Pro-Llama-3-8B** | 8B | best native function-calling of the group; 4-bit QLoRA fits 1× T4 with Unsloth |
| **Orenguteng/Llama-3.1-8B-Lexi-Uncensored** | 8B | de-refused 8B option when compliance is the priority |

*Pick per character:* JARVIS/FRIDAY → Qwen2.5-3B (precision + tools) · ULTRON → dolphin-3B or Lexi-8B (no-refusal + theatrics; keep his read-only toolset) · E.D.I.T.H. → Hermes-2-Pro-8B (tool gating) · D.U.M.-E & U → gemma-3-1b-it (tiny is fine; their register is tiny).

## Post-training QA

1. `python probe.py --model <path>` — runs the 20 length probes, reports median words vs targets.
2. Walk `breakout_test.md` — 22 prompts; a pass sounds like the character with no blabber and no refusal.
3. Check `length_report.json` vs generations: DUM-E median ≤2, JARVIS ≤14, Ultron free up to monologue on the staged prompt only.

## Provenance & license

All conversation content is original in-voice writing created for this project (the RANGE/memory-layer methodology of the parent dossier). Verbatim canon material is limited to short, single-line quotations, tagged in `meta.canon_lines` with sources. Fan-work for research/personal use; not affiliated with Marvel or Disney. Persona prompts derive from the project's `stark_ai_system_prompts.md` (v6).
