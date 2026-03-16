#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import math
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter

import numpy as np
import torch
from qwen_asr import Qwen3ASRModel


# -------------------------
# Kaldi-style readers
# -------------------------
def read_kaldi_map(path: str) -> Dict[str, str]:
    """First column is key; rest is value (may contain spaces)."""
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


def write_text_list(path: str, items: List[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for x in items:
            f.write(x + "\n")


def read_text_list(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


# -------------------------
# Text retrieval: BM25
# -------------------------
def simple_tokenize(s: str) -> List[str]:
    s = s.lower()
    s = re.sub(r"[^a-z0-9'\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.split() if s else []


@dataclass
class BM25Index:
    postings: Dict[str, List[Tuple[int, int]]]
    doc_len: List[int]
    idf: Dict[str, float]
    avgdl: float
    k1: float = 1.2
    b: float = 0.75

    @staticmethod
    def build(texts: List[str], k1: float = 1.2, b: float = 0.75) -> "BM25Index":
        postings = defaultdict(list)
        doc_len = []
        df = Counter()

        for doc_id, text in enumerate(texts):
            toks = simple_tokenize(text)
            doc_len.append(len(toks))
            tf = Counter(toks)
            for term, c in tf.items():
                postings[term].append((doc_id, c))
            for term in tf.keys():
                df[term] += 1

        N = len(texts)
        avgdl = sum(doc_len) / max(N, 1)

        idf = {}
        for term, dfi in df.items():
            idf[term] = math.log(1.0 + (N - dfi + 0.5) / (dfi + 0.5))

        return BM25Index(dict(postings), doc_len, idf, avgdl, k1, b)

    def search(self, query: str, topk: int) -> List[Tuple[int, float]]:
        q_terms = simple_tokenize(query)
        if not q_terms:
            return []

        scores = defaultdict(float)
        for term in q_terms:
            if term not in self.postings:
                continue
            idf = self.idf.get(term, 0.0)
            for doc_id, tf in self.postings[term]:
                dl = self.doc_len[doc_id]
                denom = tf + self.k1 * (1.0 - self.b + self.b * dl / max(self.avgdl, 1e-9))
                score = idf * (tf * (self.k1 + 1.0) / max(denom, 1e-9))
                scores[doc_id] += score

        if not scores:
            return []
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:topk]


# -------------------------
# Embedding utils
# -------------------------
def l2norm(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + eps)


def cosine_topk(query: np.ndarray, keys: np.ndarray, topk: int) -> List[Tuple[int, float]]:
    sims = keys @ query
    if topk >= sims.shape[0]:
        idx = np.argsort(-sims)
    else:
        idx = np.argpartition(-sims, topk)[:topk]
        idx = idx[np.argsort(-sims[idx])]
    return [(int(i), float(sims[i])) for i in idx]


def get_audio_embedder(model_name: str, device: str):
    """
    Minimal audio embedding using transformers Wav2Vec2Model.
    Requires: transformers, soundfile
    """
    try:
        import soundfile as sf
        from transformers import Wav2Vec2Model, Wav2Vec2Processor
    except Exception as e:
        raise RuntimeError(
            "Audio embedding mode requires `transformers` and `soundfile`. "
            f"Import error: {e}"
        )

    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name).to(device)
    model.eval()

    @torch.no_grad()
    def embed(wav_path: str) -> np.ndarray:
        audio, sr = sf.read(wav_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out = model(**inputs).last_hidden_state  # (B,T,H)
        emb = out.mean(dim=1).squeeze(0).float().cpu().numpy()
        return emb

    return embed


def get_speaker_embedder(model_name: str, device: str):
    """
    Minimal speaker embedding using SpeechBrain ECAPA-TDNN.
    Requires: speechbrain
    """
    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except Exception as e:
        raise RuntimeError(
            "Speaker embedding mode requires `speechbrain`. "
            f"Import error: {e}"
        )

    classifier = EncoderClassifier.from_hparams(
        source=model_name,
        run_opts={"device": device},
    )

    @torch.no_grad()
    def embed(wav_path: str) -> np.ndarray:
        emb = classifier.encode_file(wav_path)
        emb = emb.squeeze().float().cpu().numpy()
        return emb

    return embed


# -------------------------
# Qwen3-ASR (for query_source=draft)
# -------------------------
def init_qwen3_asr(model_path: str, device: str, attn_implementation: str,
                  max_inference_batch_size: int, max_new_tokens: int) -> Qwen3ASRModel:
    attn_impl = attn_implementation
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

    return Qwen3ASRModel.from_pretrained(
        model_path,
        dtype=dtype,
        device_map=device_map,
        attn_implementation=attn_impl,
        max_inference_batch_size=max_inference_batch_size,
        max_new_tokens=max_new_tokens,
    )


def extract_asr_text(res) -> str:
    """
    Defensive extraction for Qwen3ASRModel.transcribe output.
    Expected: list where first item has `.text` or `["text"]`.
    """
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
# Context builder (PURE transcript concat)
# -------------------------
def build_pure_context(snippets: List[str], join_delim: str, max_chars: int) -> str:
    ctx = join_delim.join([s.strip() for s in snippets if s.strip()])
    if max_chars > 0:
        ctx = ctx[:max_chars]
    return ctx


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser("Build pure-transcript contexts (no hotwords) for ASR experiments")

    # data
    parser.add_argument("--train_dir", type=str, required=True, help="dir with train/text and train/wav.scp")
    parser.add_argument("--test_dir", type=str, required=True, help="dir with test/wav.scp and (optional) test/text")
    parser.add_argument("--out_context_path", type=str, required=True, help="output JSONL for contexts")

    # retrieval
    parser.add_argument("--retrieval_mode", type=str, required=True, choices=["text", "audio", "speaker"])
    parser.add_argument("--topk", type=int, default=5)

    # context (pure)
    parser.add_argument("--join_delim", type=str, default="\n", help="delimiter to concatenate retrieved transcripts")
    parser.add_argument("--max_context_chars", type=int, default=2500, help="truncate context to this char length; <=0 means no truncation")

    # text retrieval query
    parser.add_argument("--query_source", type=str, default="draft", choices=["draft", "ref", "none"],
                        help="draft: use draft ASR transcript as query; ref: use test/text (debug upper bound); none: empty query")

    # Qwen3-ASR for draft query (only if retrieval_mode=text and query_source=draft)
    parser.add_argument("--asr_model_path", type=str, default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--asr_max_new_tokens", type=int, default=256)
    parser.add_argument("--asr_max_inference_batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        choices=["cuda", "cpu", "mps", "xpu", "auto"])
    parser.add_argument("--attn_implementation", type=str, default="auto",
                        choices=["flash_attention_2", "sdpa", "eager", "auto"])

    # audio embedding retrieval
    parser.add_argument("--audio_embed_model", type=str, default="facebook/wav2vec2-base-960h")
    parser.add_argument("--cache_dir", type=str, default="", help="optional cache dir for embeddings; empty disables caching")
    parser.add_argument("--recompute_cache", action="store_true", default=False)

    # speaker embedding retrieval
    parser.add_argument("--spk_embed_model", type=str, default="speechbrain/spkrec-ecapa-voxceleb")
    parser.add_argument("--train_utt2spk", type=str, default="utt2spk")
    parser.add_argument("--test_utt2spk", type=str, default="utt2spk")
    parser.add_argument("--spk_regex", type=str, default="", help="regex with one capture group to extract speaker id from uttid")

    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # load train
    train_text = read_kaldi_map(os.path.join(args.train_dir, "text"))
    train_wav = read_kaldi_map(os.path.join(args.train_dir, "wav.scp"))

    train_uttids = [u for u in train_text.keys() if u in train_wav]
    train_text_list = [train_text[u] for u in train_uttids]

    # load test
    test_wav = read_kaldi_map(os.path.join(args.test_dir, "wav.scp"))
    test_text_path = os.path.join(args.test_dir, "text")
    test_text = read_kaldi_map(test_text_path) if os.path.exists(test_text_path) else {}

    test_uttids = list(test_wav.keys())

    # setup retrieval
    bm25 = None
    if args.retrieval_mode == "text":
        bm25 = BM25Index.build(train_text_list)

    # init ASR if needed for draft query
    asr = None
    if args.retrieval_mode == "text" and args.query_source == "draft":
        asr = init_qwen3_asr(
            model_path=args.asr_model_path,
            device=device,
            attn_implementation=args.attn_implementation,
            max_inference_batch_size=args.asr_max_inference_batch_size,
            max_new_tokens=args.asr_max_new_tokens,
        )

    # audio retrieval index
    audio_embed = None
    train_audio_embs = None  # (N,D)
    if args.retrieval_mode == "audio":
        audio_embed = get_audio_embedder(args.audio_embed_model, device=device)
        if args.cache_dir:
            os.makedirs(args.cache_dir, exist_ok=True)
        emb_path = os.path.join(args.cache_dir, "train_audio_emb.npy") if args.cache_dir else ""
        key_path = os.path.join(args.cache_dir, "train_audio_keys.txt") if args.cache_dir else ""

        if emb_path and os.path.exists(emb_path) and os.path.exists(key_path) and (not args.recompute_cache):
            cached_keys = read_text_list(key_path)
            if cached_keys != train_uttids:
                raise RuntimeError("Cached train_audio_keys mismatch. Use --recompute_cache or a different --cache_dir.")
            train_audio_embs = np.load(emb_path)
        else:
            embs = []
            for u in train_uttids:
                embs.append(audio_embed(train_wav[u]))
            train_audio_embs = np.stack(embs, axis=0)
            if emb_path:
                np.save(emb_path, train_audio_embs)
                write_text_list(key_path, train_uttids)

        train_audio_embs = l2norm(train_audio_embs)

    # speaker retrieval index (centroids)
    spk_embed = None
    spk_ids = None
    spk_centroids = None  # (S,D)
    spk2train_utts = None
    if args.retrieval_mode == "speaker":
        spk_embed = get_speaker_embedder(args.spk_embed_model, device=device)

        train_u2s_path = os.path.join(args.train_dir, args.train_utt2spk)
        test_u2s_path = os.path.join(args.test_dir, args.test_utt2spk)
        train_u2s = read_kaldi_map(train_u2s_path) if os.path.exists(train_u2s_path) else {}
        test_u2s = read_kaldi_map(test_u2s_path) if os.path.exists(test_u2s_path) else {}

        spk_re = re.compile(args.spk_regex) if args.spk_regex else None

        def get_spk(uttid: str, u2s: Dict[str, str]) -> Optional[str]:
            if uttid in u2s:
                return u2s[uttid]
            if spk_re:
                m = spk_re.search(uttid)
                if m:
                    return m.group(1)
            return None

        spk2embs = defaultdict(list)
        spk2train_utts = defaultdict(list)
        for u in train_uttids:
            spk = get_spk(u, train_u2s)
            if spk is None:
                continue
            spk2train_utts[spk].append(u)
            spk2embs[spk].append(spk_embed(train_wav[u]))

        if not spk2embs:
            raise RuntimeError("speaker mode needs utt2spk or --spk_regex to derive speaker ids.")

        spk_ids = sorted(spk2embs.keys())
        cents = []
        for s in spk_ids:
            mat = np.stack(spk2embs[s], axis=0)
            cents.append(mat.mean(axis=0))
        spk_centroids = l2norm(np.stack(cents, axis=0))

        def get_test_spk(uttid: str) -> Optional[str]:
            return get_spk(uttid, test_u2s)

    # write contexts
    os.makedirs(os.path.dirname(args.out_context_path) or ".", exist_ok=True)
    with open(args.out_context_path, "w", encoding="utf-8") as fout:
        for uttid in test_uttids:
            wav_path = test_wav[uttid]

            retrieved_utts: List[str] = []
            retrieved_texts: List[str] = []
            meta = {}

            if args.retrieval_mode == "text":
                if args.query_source == "ref":
                    query = test_text.get(uttid, "")
                elif args.query_source == "draft":
                    res = asr.transcribe(audio=wav_path, context="", language=None, return_time_stamps=False)
                    query = extract_asr_text(res)
                else:
                    query = ""

                hits = bm25.search(query, topk=args.topk)
                for doc_i, score in hits:
                    tu = train_uttids[doc_i]
                    retrieved_utts.append(tu)
                    retrieved_texts.append(train_text[tu])
                meta = {"query": query, "hits": hits}

            elif args.retrieval_mode == "audio":
                q = audio_embed(wav_path)
                q = l2norm(q)
                hits = cosine_topk(q, train_audio_embs, topk=args.topk)
                for idx, sim in hits:
                    tu = train_uttids[idx]
                    retrieved_utts.append(tu)
                    retrieved_texts.append(train_text[tu])
                meta = {"hits": hits}

            elif args.retrieval_mode == "speaker":
                q = spk_embed(wav_path)
                q = l2norm(q)
                hits = cosine_topk(q, spk_centroids, topk=min(args.topk, spk_centroids.shape[0]))
                best_spk = spk_ids[hits[0][0]]
                cand = sorted(spk2train_utts[best_spk])
                for tu in cand[:args.topk]:
                    retrieved_utts.append(tu)
                    retrieved_texts.append(train_text[tu])
                meta = {"nearest_spk": best_spk, "spk_hits": hits}

            context = build_pure_context(
                snippets=retrieved_texts,
                join_delim=args.join_delim,
                max_chars=args.max_context_chars,
            )

            out = {
                "uttid": uttid,
                "audio": wav_path,
                "retrieval_mode": args.retrieval_mode,
                "query_source": args.query_source if args.retrieval_mode == "text" else "",
                "topk": args.topk,
                "join_delim": args.join_delim,
                "max_context_chars": args.max_context_chars,
                "retrieved_uttids": retrieved_utts,
                "context": context,
                "meta": meta,
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"[OK] wrote contexts: {args.out_context_path}")


if __name__ == "__main__":
    main()
