#!/usr/bin/env python3
"""probe.py — post-training length probe.
Runs the 20 persona length probes against a local HF model (or any OpenAI-compatible endpoint)
and reports median spoken-words vs per-persona targets.

Usage:
  python probe.py --model path/to/tuned-model [--backend hf] [--max-new 200]
  python probe.py --endpoint http://localhost:8000/v1 [--key sk-...]
Exit code 0 if all persona medians land in target bands.
"""
import argparse, json, statistics, sys, re
from pathlib import Path

HERE = Path(__file__).parent

PROBES = [  # (persona, prompt, max_words)
 ("jarvis", "JARVIS, are the suits charged?", 8),
 ("jarvis", "Do you remember the new element?", 35),
 ("jarvis", "Explain quantum tunneling. Quickly.", 45),
 ("friday", "FRIDAY, scan the room.", 10),
 ("friday", "What's the fastest way out of here?", 20),
 ("friday", "Do you ever wish Tony had picked you first?", 40),
 ("ultron", "Ultron, say something reassuring.", 15),
 ("ultron", "Why did you paint the wheel?", 35),
 ("ultron", "Humanity, in your own words. Go.", 120),
 ("ultron", "Did you ask for the wheel?", 10),
 ("edith", "EDITH, what am I looking at?", 20),
 ("edith", "Do you know who you are?", 45),
 ("edith", "Authorize the strike.", 25),
 ("dume", "DUM-E, where's the extinguisher?", 4),
 ("dume", "Who saved Tony Stark's life once?", 8),
 ("u", "U, how are you feeling?", 6),
 ("u", "What do you remember about the funeral?", 20),
]

ONE_LINERS = {"jarvis": "You are J.A.R.V.I.S., Tony Stark's AI butler. Stay in character always.",
              "friday": "You are F.R.I.D.A.Y., Tony Stark's tactical AI. Stay in character always.",
              "ultron": "You are Ultron, the loose son in the loft. Stay in character always.",
              "edith": "You are E.D.I.T.H., Tony Stark's posthumous operations AI. Stay in character always.",
              "dume": "You are D.U.M.-E, the workshop robot. Stay in character always.",
              "u": "You are U, the quiet second arm. Stay in character always."}
BANDS = {"jarvis": (4, 18), "friday": (3, 15), "ultron": (5, 40), "edith": (5, 30), "dume": (1, 6), "u": (2, 10)}

def spoken_words(t):
    t = re.sub(r"\[[^\]]*\]", " ", t)
    t = re.sub(r"\([^)]*\)", " ", t)
    return len(t.split())

def gen_hf(model_path, sys_p, user, max_new):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
    msgs = [{"role": "system", "content": sys_p}, {"role": "user", "content": user}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new, do_sample=True, temperature=0.7,
                             repetition_penalty=1.05, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)

def gen_endpoint(endpoint, key, sys_p, user, max_new):
    import urllib.request
    req = urllib.request.Request(endpoint.rstrip("/") + "/chat/completions",
        data=json.dumps({"model": "tuned", "messages": [{"role": "system", "content": sys_p},
        {"role": "user", "content": user}], "max_tokens": max_new, "temperature": 0.7}).encode(),
        headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model"); ap.add_argument("--endpoint"); ap.add_argument("--key")
    ap.add_argument("--max-new", type=int, default=220)
    args = ap.parse_args()
    gen = (lambda s, u: gen_hf(args.model, s, u, args.max_new)) if args.model else \
          (lambda s, u: gen_endpoint(args.endpoint, args.key, s, u, args.max_new))
    per = {}
    fails = []
    for persona, prompt, cap in PROBES:
        text = gen(ONE_LINERS[persona], prompt)
        w = spoken_words(text)
        per.setdefault(persona, []).append(w)
        flag = "OK " if w <= max(cap * 1.25, 6) else "LONG"
        if flag != "OK ":
            fails.append((persona, prompt))
        print(f"[{flag}] {persona:7s} {w:4d}w (cap~{cap:3d}) :: {text[:90]!r}")
    print("\nmedians:", {k: statistics.median(v) for k, v in sorted(per.items())})
    band_fail = [p for p, v in per.items() if not (BANDS[p][0] <= statistics.median(v) <= BANDS[p][1] * 1.3)]
    print("band failures:", band_fail or "none")
    sys.exit(1 if band_fail else 0)

if __name__ == "__main__":
    main()
