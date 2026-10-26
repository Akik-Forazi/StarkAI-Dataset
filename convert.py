#!/usr/bin/env python3
"""convert.py — convert train/val JSONL (messages format) to ShareGPT, Llama-3 ChatML text, or raw text.
Usage:
  python convert.py --in train.jsonl --out train.sharegpt.jsonl --fmt sharegpt
  python convert.py --in train.jsonl --out train.chatml.txt     --fmt chatml
  python convert.py --in train.jsonl --out train.raw.txt        --fmt text
DPO pairs: use --dpo to convert dpo_pairs.jsonl into (prompt, chosen, rejected) ShareGPT-style triples.
"""
import argparse, json, sys

def to_sharegpt(row):
    msgs = row["messages"]
    conv = [{"from": "system", "value": msgs[0]["content"]}]
    for m in msgs[1:]:
        conv.append({"from": "human" if m["role"] == "user" else "gpt", "value": m["content"]})
    return {"conversations": conv, "meta": row.get("meta", {})}

CHATML = {"system": "<|start_header_id|>system<|end_header_id|>\n\n", "user": "<|start_header_id|>user<|end_header_id|>\n\n", "assistant": "<|start_header_id|>assistant<|end_header_id|>\n\n"}
def to_chatml(row):
    out = []
    for m in row["messages"]:
        out.append(f'<|begin_of_text|>{CHATML[m["role"]]}{m["content"]}<|eot_id|>')
    return "".join(out) + "\n"

def to_text(row):
    out = []
    for m in row["messages"]:
        out.append(f'### {m["role"].upper()}:\n{m["content"]}\n')
    return "\n".join(out) + "\n"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--fmt", choices=["sharegpt", "chatml", "text"], default="sharegpt")
    args = ap.parse_args()
    conv = {"sharegpt": to_sharegpt, "chatml": to_chatml, "text": to_text}[args.fmt]
    n = 0
    with open(args.inp, encoding="utf-8") as f, open(args.out, "w", encoding="utf-8") as w:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if args.fmt == "sharegpt":
                w.write(json.dumps(conv(row), ensure_ascii=False) + "\n")
            else:
                w.write(conv(row))
            n += 1
    print(f"converted {n} rows -> {args.out} ({args.fmt})")

if __name__ == "__main__":
    main()
