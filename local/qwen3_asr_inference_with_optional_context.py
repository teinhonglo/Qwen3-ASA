#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import re
import argparse
from typing import Dict, List, Tuple

import torch
from qwen_asr import Qwen3ASRModel


def read_kaldi_map(path: str) -> Dict[str, str]:
    d = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            k = parts[0]
            v = " ".join(parts[1:]) if len(parts) > 1 else ""
            d[k] = v
    return d


def load_contexts(context_jsonl: str) -> Dict[str, str]:
    m = {}
    with open(context_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            m[obj["uttid"]] = obj.get("context", "")
    return m


def extract_asr_text(res) -> str:
    if res is None:
        return ""
    if isinstance(res, list) and len(res) > 0:
        x = res[0]
        if hasattr(x, "text"):
            return x.text or ""
        if isinstance(x, dict) and "text" in x:
            return x["text"] or ""
    if isinstance(res, dict) and "text" in res:
        return res["text"] or ""
    return ""


# -------------------------
# WER normalization + compute
# -------------------------
_PUNCT_RE = re.compile(r"[^A-Za-z0-9\s]+")

def normalize_for_wer(s: str, remove_punct: bool, uppercase: bool, collapse_ws: bool) -> str:
    if s is None:
        s = ""
    if remove_punct:
        s = _PUNCT_RE.sub(" ", s)
    if collapse_ws:
        s = re.sub(r"\s+", " ", s).strip()
    if uppercase:
        s = s.upper()
    return s


def edit_counts_words(ref_words: List[str], hyp_words: List[str]) -> Tuple[int, int, int]:
    """
    Return (S, I, D) via DP. Deterministic tie-break by (cost, S, I, D).
    """
    n = len(ref_words)
    m = len(hyp_words)

    prev = [(j, 0, j, 0) for j in range(m + 1)]  # (cost,S,I,D)
    for i in range(1, n + 1):
        cur = [(i, 0, 0, i)] + [(0, 0, 0, 0)] * m
        for j in range(1, m + 1):
            sub_cost, sub_S, sub_I, sub_D = prev[j - 1]
            if ref_words[i - 1] != hyp_words[j - 1]:
                cand_sub = (sub_cost + 1, sub_S + 1, sub_I, sub_D)
            else:
                cand_sub = (sub_cost, sub_S, sub_I, sub_D)

            ins_cost, ins_S, ins_I, ins_D = cur[j - 1]
            cand_ins = (ins_cost + 1, ins_S, ins_I + 1, ins_D)

            del_cost, del_S, del_I, del_D = prev[j]
            cand_del = (del_cost + 1, del_S, del_I, del_D + 1)

            cur[j] = min(cand_sub, cand_ins, cand_del, key=lambda x: (x[0], x[1], x[2], x[3]))
        prev = cur

    _, S, I, D = prev[m]
    return S, I, D


def compute_wer(ref_map: Dict[str, str], hyp_map: Dict[str, str], uttids: List[str],
                remove_punct: bool, uppercase: bool, collapse_ws: bool) -> Dict:
    total_ref_words = 0
    S_tot = I_tot = D_tot = 0

    for u in uttids:
        r = normalize_for_wer(ref_map.get(u, ""), remove_punct, uppercase, collapse_ws)
        h = normalize_for_wer(hyp_map.get(u, ""), remove_punct, uppercase, collapse_ws)

        r_w = r.split() if r else []
        h_w = h.split() if h else []

        total_ref_words += len(r_w)
        S, I, D = edit_counts_words(r_w, h_w)
        S_tot += S
        I_tot += I
        D_tot += D

    wer = (S_tot + I_tot + D_tot) / total_ref_words if total_ref_words > 0 else 0.0
    return {
        "num_utts": len(uttids),
        "num_ref_words": total_ref_words,
        "substitutions": S_tot,
        "insertions": I_tot,
        "deletions": D_tot,
        "wer": wer,
    }


def main():
    parser = argparse.ArgumentParser("Qwen3-ASR inference with optional per-utt context + auto WER")

    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--data_dir", type=str, required=True, help="dir containing wav.scp and (optional) text")
    parser.add_argument("--output_dir", type=str, required=True)

    # optional context
    parser.add_argument("--use_context", action="store_true", default=False)
    parser.add_argument("--context_jsonl", type=str, default="")
    parser.add_argument("--fallback_empty_context", action="store_true", default=True)

    # decoding / device
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        choices=["cuda", "cpu", "mps", "xpu", "auto"])
    parser.add_argument("--attn_implementation", type=str, default="auto",
                        choices=["flash_attention_2", "sdpa", "eager", "auto"])
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_inference_batch_size", type=int, default=32)

    # WER
    parser.add_argument("--compute_wer", action="store_true", default=True)
    parser.add_argument("--wer_remove_punct", action="store_true", default=True)
    parser.add_argument("--wer_uppercase", action="store_true", default=True)
    parser.add_argument("--wer_collapse_ws", action="store_true", default=True)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    wav = read_kaldi_map(os.path.join(args.data_dir, "wav.scp"))
    text_path = os.path.join(args.data_dir, "text")
    has_ref = os.path.exists(text_path)
    ref = read_kaldi_map(text_path) if has_ref else {}

    if args.compute_wer and not has_ref:
        raise RuntimeError("compute_wer enabled but data_dir/text is missing.")

    uttids = [u for u in wav.keys() if (not has_ref) or (u in ref)]

    ctx_map = {}
    if args.use_context:
        if not args.context_jsonl:
            raise RuntimeError("--use_context requires --context_jsonl")
        ctx_map = load_contexts(args.context_jsonl)

    # attention impl auto
    attn_impl = args.attn_implementation
    if attn_impl == "auto":
        if device == "cuda" and torch.cuda.is_available():
            try:
                import flash_attn  # noqa: F401
                attn_impl = "flash_attention_2"
            except Exception:
                attn_impl = "sdpa"
        else:
            attn_impl = "sdpa"

    if device in ["cpu", "mps", "xpu"]:
        dtype = torch.float32
        device_map = None
    else:
        dtype = torch.bfloat16
        device_map = "cuda"

    asr = Qwen3ASRModel.from_pretrained(
        args.model_path,
        dtype=dtype,
        device_map=device_map,
        attn_implementation=attn_impl,
        max_inference_batch_size=args.max_inference_batch_size,
        max_new_tokens=args.max_new_tokens,
    )

    hyp_raw: Dict[str, str] = {}

    hyp_path = os.path.join(args.output_dir, "hyp")
    ref_out_path = os.path.join(args.output_dir, "ref")
    debug_path = os.path.join(args.output_dir, "debug.jsonl")
    metrics_path = os.path.join(args.output_dir, "metrics.json")

    # ---- FIX: do NOT use conditional context manager ----
    ref_f = open(ref_out_path, "w", encoding="utf-8") if has_ref else None
    hyp_f = open(hyp_path, "w", encoding="utf-8")
    dbg_f = open(debug_path, "w", encoding="utf-8")

    try:
        for u in uttids:
            audio_path = wav[u]
            if args.use_context:
                if u in ctx_map:
                    ctx = ctx_map[u]
                else:
                    if args.fallback_empty_context:
                        ctx = ""
                    else:
                        raise RuntimeError(f"uttid {u} not found in context_jsonl")
            else:
                ctx = ""

            res = asr.transcribe(
                audio=audio_path,
                context=ctx,
                language=None,
                return_time_stamps=False,
            )
            hyp = extract_asr_text(res)
            hyp_raw[u] = hyp

            hyp_n = normalize_for_wer(hyp, args.wer_remove_punct, args.wer_uppercase, args.wer_collapse_ws)
            hyp_f.write(f"{hyp_n} ({u})\n")

            if ref_f is not None:
                ref_n = normalize_for_wer(ref[u], args.wer_remove_punct, args.wer_uppercase, args.wer_collapse_ws)
                ref_f.write(f"{ref_n} ({u})\n")

            dbg_f.write(json.dumps({
                "uttid": u,
                "audio": audio_path,
                "used_context": bool(ctx),
                "context_preview": ctx[:200],
                "hyp_raw": hyp,
                "hyp_norm": hyp_n,
                "ref_raw": ref.get(u, ""),
                "ref_norm": normalize_for_wer(ref.get(u, ""), args.wer_remove_punct, args.wer_uppercase, args.wer_collapse_ws) if has_ref else "",
            }, ensure_ascii=False) + "\n")
    finally:
        hyp_f.close()
        dbg_f.close()
        if ref_f is not None:
            ref_f.close()

    if args.compute_wer and has_ref:
        metrics = compute_wer(
            ref_map=ref,
            hyp_map=hyp_raw,
            uttids=uttids,
            remove_punct=args.wer_remove_punct,
            uppercase=args.wer_uppercase,
            collapse_ws=args.wer_collapse_ws,
        )
        metrics.update({
            "data_dir": args.data_dir,
            "use_context": bool(args.use_context),
            "context_jsonl": args.context_jsonl if args.use_context else "",
            "wer_normalization": {
                "remove_punct": args.wer_remove_punct,
                "uppercase": args.wer_uppercase,
                "collapse_ws": args.wer_collapse_ws,
            },
            "model_path": args.model_path,
            "attn_implementation": attn_impl,
            "device": device,
            "max_new_tokens": args.max_new_tokens,
            "max_inference_batch_size": args.max_inference_batch_size,
        })
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        print(f"[OK] WER = {metrics['wer']:.6f}  (S={metrics['substitutions']}, I={metrics['insertions']}, D={metrics['deletions']}, N={metrics['num_ref_words']})")
        print(f"[OK] wrote: {metrics_path}")

    print(f"[OK] wrote: {hyp_path}")
    if has_ref:
        print(f"[OK] wrote: {ref_out_path}")
    print(f"[OK] wrote: {debug_path}")


if __name__ == "__main__":
    main()
