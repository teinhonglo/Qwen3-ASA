#!/usr/bin/env python
"""
VibeVoice ASR Batch Inference Demo Script

This script supports batch inference for ASR model and compares results
between batch processing and single-sample processing.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import argparse
import time
import json
import re
from typing import List, Dict, Any, Optional
from functools import wraps
from whisper.normalizers import EnglishTextNormalizer
import soundfile as sf
from qwen_asr import Qwen3ASRModel

FORCED_ALIGNER_PATH = "Qwen/Qwen3-ForcedAligner-0.6B"

def file2dict(fname):
    uttid_list = []
    file2uttid = {}
    data_dict = {}
    with open(fname, "r") as fn:
        for line in fn.readlines():
            info = line.split()
            uttid, text = info[0], " ".join(info[1:])
            data_dict[uttid] = text
            uttid_list.append(uttid)
            file2uttid[text] = uttid
    
    return data_dict, uttid_list, file2uttid

import re
import unicodedata
from typing import Tuple, List

# Match sequences like "U S B" / "a b c" (single-letter tokens) and merge into "USB" / "abc"
_SPELLED_LETTERS_RE = re.compile(r"(?:\b[A-Za-z]\b(?:\s+\b[A-Za-z]\b)+)")

def _is_cjk(ch: str) -> bool:
    """Return True if ch is a CJK ideograph (common Han ranges)."""
    cp = ord(ch)
    return (
        0x4E00 <= cp <= 0x9FFF   # CJK Unified Ideographs
        or 0x3400 <= cp <= 0x4DBF  # Extension A
        or 0x20000 <= cp <= 0x2A6DF # Extension B
        or 0x2A700 <= cp <= 0x2B73F # Extension C
        or 0x2B740 <= cp <= 0x2B81F # Extension D
        or 0x2B820 <= cp <= 0x2CEAF # Extension E/F
        or 0xF900 <= cp <= 0xFAFF   # Compatibility Ideographs
        or 0x2F800 <= cp <= 0x2FA1F # Compatibility Ideographs Supplement
    )

def _remove_punctuation(s: str) -> str:
    # remove Unicode punctuation categories: Pc, Pd, Pe, Pf, Pi, Po, Ps
    return "".join(ch for ch in s if not unicodedata.category(ch).startswith("P"))

def _merge_spelled_letters(s: str) -> str:
    # Turn "U S B" -> "USB" but keep normal words like "I am" unchanged
    return _SPELLED_LETTERS_RE.sub(lambda m: re.sub(r"\s+", "", m.group(0)), s)

def _tokenize_cjk_and_ascii_words(s: str) -> List[str]:
    """
    Tokens:
      - each CJK ideograph as a single token
      - contiguous ASCII [A-Za-z0-9]+ as one token
      - other non-space chars as single tokens (rare; punctuation already removed)
    """
    tokens: List[str] = []
    i, n = 0, len(s)
    while i < n:
        ch = s[i]
        if ch.isspace():
            i += 1
            continue

        if _is_cjk(ch):
            tokens.append(ch)
            i += 1
            continue

        if ch.isascii() and (ch.isalpha() or ch.isdigit()):
            j = i + 1
            while j < n and s[j].isascii() and (s[j].isalpha() or s[j].isdigit()):
                j += 1
            tokens.append(s[i:j])
            i = j
            continue

        tokens.append(ch)
        i += 1

    return tokens

def normalizer_zh(text: str) -> Tuple[str, str]:
    """
    Input: one-line text
    Return:
      v1: remove punctuation + de-break (merge "U S B"->"USB"), then tokenize as:
          CJK chars separated by spaces, English kept as whole word
      v2: based on v1, further split English word tokens into letters (USB -> U S B)
    """
    s = unicodedata.normalize("NFKC", text.strip())
    s = _remove_punctuation(s)
    s = re.sub(r"\s+", " ", s).strip()
    s = _merge_spelled_letters(s)

    tokens = _tokenize_cjk_and_ascii_words(s)
    v1 = " ".join(tokens)

    v2_tokens: List[str] = []
    for t in tokens:
        if re.fullmatch(r"[A-Za-z]+", t):
            v2_tokens.extend(list(t))
        else:
            v2_tokens.append(t)
    v2 = " ".join(v2_tokens)

    return v1, v2


def main():
    parser = argparse.ArgumentParser(description="VibeVoice ASR Batch Inference Demo")
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="Qwen/Qwen3-ASR-1.7B",
        help="Path to the model checkpoint"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing audio files for batch transcription"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory containing output ref and hyp"
    )
    parser.add_argument(
        "--max_duration",
        type=float,
        default=3600.0,
        help="Maximum duration in seconds for concatenated dataset audio (default: 3600 = 1 hour)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for processing multiple files"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda" if torch.cuda.is_available() else ("xpu" if torch.backends.xpu.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu") ),
        choices=["cuda", "cpu", "mps","xpu", "auto"],
        help="Device to run inference on"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for sampling (0 = greedy decoding)"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Top-p for nucleus sampling"
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help="Number of beams for beam search. Use 1 for greedy/sampling"
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="auto",
        choices=["flash_attention_2", "sdpa", "eager", "auto"],
        help="Attention implementation to use. 'auto' will select the best available for your device (flash_attention_2 for CUDA, sdpa for MPS/CPU/XPU)"
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
    )
    
    args = parser.parse_args()
    language = args.language
    
    # Auto-detect best attention implementation based on device
    if args.attn_implementation == "auto":
        if args.device == "cuda" and torch.cuda.is_available():
            try:
                import flash_attn
                args.attn_implementation = "flash_attention_2"
            except ImportError:
                print("flash_attn not installed, falling back to sdpa")
                args.attn_implementation = "sdpa"
        else:
            # MPS/XPU/CPU don't support flash_attention_2
            args.attn_implementation = "sdpa"
        print(f"Auto-detected attention implementation: {args.attn_implementation}")
    
    # Initialize model
    # Handle MPS device and dtype
    if args.device == "mps":
        model_dtype = torch.float32  # MPS works better with float32
    elif args.device == "xpu":
        model_dtype = torch.float32
    elif args.device == "cpu":
        model_dtype = torch.float32
    else:
        model_dtype = torch.bfloat16
    
    
    # If temperature is 0, use greedy decoding (no sampling)
    do_sample = args.temperature > 0
    
    # Combine all audio inputs
    data_dir = args.data_dir
    wav_dict, uttid_list, file2uttid_dict = file2dict(os.path.join(data_dir, "wav.scp"))
    utt2spk_dict, _, spk2uttid_dict = file2dict(os.path.join(data_dir, "utt2spk"))
    # ref
    utt2text_dict, _, _ = file2dict(os.path.join(data_dir, "text"))
    all_audio_inputs = list(wav_dict.values()) #[:10]
    
    print("\n" + "="*80)
    print(f"Processing {len(all_audio_inputs)} audio(s)")
    print("="*80)
    
    asr = Qwen3ASRModel.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
        max_inference_batch_size=32,
        max_new_tokens=args.max_new_tokens,
    )

    normalizer_en = EnglishTextNormalizer()
    hyp_dict = {} 
    # Print results
    for file_path in all_audio_inputs:
        results = asr.transcribe(
            audio=file_path,
            context="",
            language=language,
            return_time_stamps=False,
        )
        uttid = file2uttid_dict[file_path]
        text = results[0].text
        hyp_dict[uttid] = text
    
    # exp/stt/whisper-large-v2/all_16k/{ref,hyp}
    output_dir = args.output_dir
    recog_dir = os.path.join(data_dir, args.model_path.replace("/", "-").lower())
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(recog_dir, exist_ok=True)
    
    ref_fn = open(os.path.join(output_dir, "ref"), "w")
    hyp_fn = open(os.path.join(output_dir, "hyp"), "w")
    # normalize
    ref_norm_fn = open(os.path.join(output_dir, "ref_norm"), "w")
    hyp_norm_fn = open(os.path.join(output_dir, "hyp_norm"), "w")
    # normalize + characterized
    ref_normc_fn = open(os.path.join(output_dir, "ref_normc"), "w")
    hyp_normc_fn = open(os.path.join(output_dir, "hyp_normc"), "w")
    recog_fn = open(os.path.join(recog_dir, "text"), "w")

    for uttid in uttid_list:
        if uttid not in hyp_dict: continue
        spkid = utt2spk_dict[uttid]

        ref_text = utt2text_dict[uttid]
        hyp_text = hyp_dict[uttid]
        
        if language == "English":
            ref_norm_text = normalizer_en(ref_text)
            hyp_norm_text = normalizer_en(hyp_text)
            ref_normc_text = list("".join(ref_text.split()))
            hyp_normc_text = list("".join(hyp_text.split()))
        elif language == "Chinese":
            ref_norm_text, ref_normc_text = normalizer_zh(ref_text)
            hyp_norm_text, hyp_normc_text = normalizer_zh(hyp_text)
        else:
            print("Language {language} haven't supported normalize yet")

        ref_fn.write(f"{ref_text} ({spkid}-{uttid})\n")
        hyp_fn.write(f"{hyp_text} ({spkid}-{uttid})\n")

        ref_norm_fn.write(f"{ref_norm_text} ({spkid}-{uttid})\n")
        hyp_norm_fn.write(f"{hyp_norm_text} ({spkid}-{uttid})\n")

        ref_normc_fn.write(f"{ref_normc_text} ({spkid}-{uttid})\n")
        hyp_normc_fn.write(f"{hyp_normc_text} ({spkid}-{uttid})\n")
        recog_fn.write(f"{uttid} {hyp_text}\n")
    
    ref_fn.close()
    hyp_fn.close()
    recog_fn.close()
        
if __name__ == "__main__":
    main()
