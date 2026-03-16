#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import shutil
import tarfile
from pathlib import Path
from typing import Dict, List

from huggingface_hub import hf_hub_download


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download MAC-SLU and prepare Kaldi-format data dirs."
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="Gatsby1984/MAC_SLU",
        help="Hugging Face dataset repo id.",
    )
    parser.add_argument(
        "--download-dir",
        type=str,
        required=True,
        help="Directory to store downloaded original files.",
    )
    parser.add_argument(
        "--extract-root",
        type=str,
        required=True,
        help="Directory to extract audio archives.",
    )
    parser.add_argument(
        "--kaldi-root",
        type=str,
        required=True,
        help="Directory to write Kaldi-format data dirs.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "dev", "test"],
        choices=["train", "dev", "test"],
        help="Splits to prepare.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="macslu",
        help="Prefix used in generated uttids.",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def ensure_local_file(
    repo_id: str,
    filename: str,
    local_path: Path,
) -> Path:
    """
    If local_path already exists, skip download.
    Otherwise download from HF Hub and copy to local_path.
    """
    local_path.parent.mkdir(parents=True, exist_ok=True)

    if local_path.exists():
        print(f"[INFO] Skip download, file already exists: {local_path}")
        return local_path

    print(f"[INFO] Downloading {filename} ...")
    cached_path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=filename,
    )
    shutil.copy2(cached_path, local_path)
    print(f"[INFO] Saved to: {local_path}")
    return local_path


def download_split_files(repo_id: str, split: str, download_dir: Path):
    label_relpath = f"label/{split}_set.jsonl"
    audio_relpath = f"audio_{split}.tar.gz"

    label_local_path = download_dir / "label" / f"{split}_set.jsonl"
    audio_local_path = download_dir / f"audio_{split}.tar.gz"

    label_path = ensure_local_file(repo_id, label_relpath, label_local_path)
    audio_tar_path = ensure_local_file(repo_id, audio_relpath, audio_local_path)

    return label_path, audio_tar_path


def safe_extract_tar(tar_path: Path, out_dir: Path) -> None:
    marker = out_dir / ".extract_done"
    if marker.exists():
        print(f"[INFO] Skip extraction, already extracted: {out_dir}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(tar_path, "r:gz") as tar:
        abs_out = out_dir.resolve()
        for member in tar.getmembers():
            target = (out_dir / member.name).resolve()
            if not str(target).startswith(str(abs_out) + os.sep) and target != abs_out:
                raise RuntimeError(f"Unsafe path detected in tar file: {member.name}")
        tar.extractall(out_dir)

    marker.touch()
    print(f"[INFO] Extracted to: {out_dir}")


def load_jsonl(path: Path) -> List[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse JSONL at {path}:{line_no}") from e
    return records


def build_wav_index(audio_root: Path) -> Dict[str, List[Path]]:
    index: Dict[str, List[Path]] = {}
    for wav_path in audio_root.rglob("*.wav"):
        stem = wav_path.stem
        index.setdefault(stem, []).append(wav_path.resolve())
    return index


def resolve_wav(raw_id: str, wav_index: Dict[str, List[Path]]) -> Path:
    raw_id = str(raw_id)

    if raw_id in wav_index:
        matches = wav_index[raw_id]
        if len(matches) == 1:
            return matches[0]
        raise RuntimeError(f"Multiple wavs found for id={raw_id}: {matches}")

    suffix_matches = []
    for stem, paths in wav_index.items():
        if stem.endswith(raw_id):
            suffix_matches.extend(paths)

    if len(suffix_matches) == 1:
        return suffix_matches[0]
    if len(suffix_matches) > 1:
        raise RuntimeError(
            f"Multiple suffix-matched wavs found for id={raw_id}: {suffix_matches}"
        )

    raise FileNotFoundError(f"Cannot find wav for id={raw_id}")


def sort_key(record: dict):
    rid = str(record.get("id", ""))
    if rid.isdigit():
        return (0, int(rid))
    return (1, rid)


def write_kaldi_files(
    records: List[dict],
    split: str,
    prefix: str,
    wav_index: Dict[str, List[Path]],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    wav_scp_path = out_dir / "wav.scp"
    text_path = out_dir / "text"
    utt2spk_path = out_dir / "utt2spk"

    missing = []

    with wav_scp_path.open("w", encoding="utf-8") as fwav, \
         text_path.open("w", encoding="utf-8") as ftext, \
         utt2spk_path.open("w", encoding="utf-8") as futt2spk:

        for record in sorted(records, key=sort_key):
            if "id" not in record or "query" not in record:
                raise KeyError(
                    f"Each JSONL record must contain 'id' and 'query'. Bad record: {record}"
                )

            raw_id = str(record["id"])
            uttid = f"{prefix}_{split}_{raw_id}"
            transcription = normalize_text(record["query"])

            try:
                wav_path = resolve_wav(raw_id, wav_index)
            except FileNotFoundError:
                missing.append(raw_id)
                continue

            spkid = uttid

            fwav.write(f"{uttid} {wav_path}\n")
            ftext.write(f"{uttid} {transcription}\n")
            futt2spk.write(f"{uttid} {spkid}\n")

    if missing:
        missing_path = out_dir / "missing_wavs.txt"
        with missing_path.open("w", encoding="utf-8") as f:
            for raw_id in missing:
                f.write(f"{raw_id}\n")
        print(
            f"[WARN] {split}: {len(missing)} utterances were skipped because wavs were not found. "
            f"See {missing_path}"
        )
    else:
        print(f"[INFO] {split}: all utterances matched with wav files.")


def main():
    args = parse_args()

    download_dir = Path(args.download_dir).resolve()
    extract_root = Path(args.extract_root).resolve()
    kaldi_root = Path(args.kaldi_root).resolve()

    for split in args.splits:
        print(f"[INFO] Processing split: {split}")

        label_path, audio_tar_path = download_split_files(
            args.repo_id,
            split,
            download_dir,
        )

        split_audio_dir = extract_root / split
        safe_extract_tar(audio_tar_path, split_audio_dir)

        records = load_jsonl(label_path)
        wav_index = build_wav_index(split_audio_dir)

        if not wav_index:
            raise RuntimeError(f"No wav files found under {split_audio_dir}")

        out_dir = kaldi_root / split
        write_kaldi_files(records, split, args.prefix, wav_index, out_dir)

        print(f"[INFO] Wrote Kaldi files to: {out_dir}")

    print("[INFO] Done.")


if __name__ == "__main__":
    main()