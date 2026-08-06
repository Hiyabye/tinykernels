#!/usr/bin/env python3
"""Verify the C byte-level BPE tokenizer against a reference.

Primary reference is llama.cpp's `llama-tokenize`. If it is not installed, a
small set of hardcoded expected id sequences (captured from llama.cpp) is used
as a fallback so the check is runnable anywhere the GGUF is present.

Usage: python3 scripts/verify_tokenizer.py [gguf_path]

Exits 0 on full match, 1 on any mismatch.
"""
import re
import subprocess
import sys

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen3-0.6B-Q8_0.gguf"
BIN = "./tinykernels"

# text -> expected token ids, captured from llama.cpp (authoritative Qwen3-0.6B)
FALLBACK = {
    "Hello, world! This is a Qwen3 test.": [9707, 11, 1879, 0, 1096, 374, 264, 1207, 16948, 18, 1273, 13],
    "你好，世界！": [108386, 3837, 99489, 6313],
    "Let's think and respond.": [10061, 594, 1744, 323, 5889, 13],
    "café": [924, 58858],
    "<|im_start|>system\nYou are helpful.<|im_end|>": [151644, 8948, 198, 2610, 525, 10950, 13, 151645],
    "Hi<tool_call>bye</tool_call>": [13048, 151657, 28374, 151658],
}

CORPUS = [
    "Hello, world! This is a Qwen3 test.",
    "你好，世界！",
    "Let's think and respond.",
    "<|im_start|>system\nYou are helpful.<|im_end|>",
    "Hi<tool_call>bye</tool_call>",
    "café",
    "",
    "The quick brown fox jumps over the lazy dog. 1234567890!",
    "Don't stop believing, 'cause it's the final countdown.",
    "Mixed 中文 and English text \U0001F389 with emoji and numbers 42.5%!",
    "naïve coöperate résumé über straße 日本 漢字",
    "a very long " + "word" * 40,
    "tab\tseparated\tvalues",
    "<|endoftext|><|im_start|>assistant<|im_end|>",
    "I'm going. It's cold! You're warm. He'd go. We'll see. They've left.",
    "Qwen3-0.6B architecture: RoPE, GQA, SwiGLU, RMSNorm.",
    "camelCase and snake_case and kebab-case 1e-6 0x1F 3.14159",
    "\n\n  Leading and trailing spaces  \n\n",
    "日本語のテキストとEnglishの混合文です。",
    "Καλημέρα κόσμε! Привет мир. مرحبا بالعالم",
]


def mine(text):
    out = subprocess.run([BIN, "tokenize", text], capture_output=True, text=True).stdout
    return [int(x) for x in out.split()] if out.strip() else []


try:
    llama = subprocess.run(["llama-tokenize", "--version"], capture_output=True)
    have_llama = llama.returncode == 0
except FileNotFoundError:
    have_llama = False


def ref(text):
    if have_llama:
        out = subprocess.run(["llama-tokenize", "-m", MODEL, "-p", text], capture_output=True)
        txt = out.stdout.decode("utf-8", "replace")
        return [int(x) for x in re.findall(r"^\s*(\d+)\s*->", txt, re.M)]
    return FALLBACK[text]


def main():
    source = "llama-tokenize" if have_llama else "hardcoded fallback"
    print(f"reference: {source}")
    fails = 0
    for text in CORPUS:
        if text not in FALLBACK and not have_llama:
            continue  # only fallback cases checkable without llama.cpp
        a, b = ref(text), mine(text)
        if a != b:
            fails += 1
            print(f"MISMATCH {text!r}\n  ref : {a}\n  mine: {b}")
    if fails:
        print(f"{fails} mismatches")
        return 1
    checked = sum(1 for t in CORPUS if t in FALLBACK or have_llama)
    print(f"OK: {checked} cases matched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
