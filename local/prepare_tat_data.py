#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from collections import defaultdict


def sort_key(x: str):
    return (0, int(x)) if x.isdigit() else (1, x)


def load_transcripts(answer_path: Path):
    """
    Read answer file with format:
        uttid token1 token2 token3 ...
    Return:
        dict[uttid] = transcript
    """
    transcripts = {}
    malformed = []

    with answer_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            parts = line.split(maxsplit=1)
            if len(parts) < 2:
                malformed.append((line_no, line))
                continue

            uttid, text = parts[0], parts[1].strip()
            transcripts[uttid] = text

    return transcripts, malformed


def collect_wavs(wav_dir: Path):
    """
    Collect wav files under wav_dir.
    Return:
        dict[uttid] = abs_wav_path
    """
    wavs = {}
    for wav_path in wav_dir.glob("*.wav"):
        uttid = wav_path.stem
        wavs[uttid] = str(wav_path.resolve())
    return wavs


def make_spk2utt(utt2spk):
    spk2utts = defaultdict(list)
    for utt, spk in utt2spk.items():
        spk2utts[spk].append(utt)

    for spk in spk2utts:
        spk2utts[spk] = sorted(spk2utts[spk], key=sort_key)

    return dict(spk2utts)


def write_kaldi_files(kaldi_data_dir: Path, wav_scp, text_dict, utt2spk, spk2utt):
    kaldi_data_dir.mkdir(parents=True, exist_ok=True)

    utts = sorted(text_dict.keys(), key=sort_key)

    with (kaldi_data_dir / "wav.scp").open("w", encoding="utf-8") as f:
        for utt in utts:
            f.write(f"{utt} {wav_scp[utt]}\n")

    with (kaldi_data_dir / "text").open("w", encoding="utf-8") as f:
        for utt in utts:
            f.write(f"{utt} {text_dict[utt]}\n")

    with (kaldi_data_dir / "utt2spk").open("w", encoding="utf-8") as f:
        for utt in utts:
            f.write(f"{utt} {utt2spk[utt]}\n")

    with (kaldi_data_dir / "spk2utt").open("w", encoding="utf-8") as f:
        for spk in sorted(spk2utt.keys(), key=sort_key):
            utts_str = " ".join(spk2utt[spk])
            f.write(f"{spk} {utts_str}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Convert TAT-FSR2020 answer file + wav dir to Kaldi data directory"
    )
    parser.add_argument(
        "--answer_path",
        type=str,
        default="/share/corpus/TAT-FSR2020/TAT-final_test-chinese/answer-chinese.txt",
        help="Path to answer-chinese.txt",
    )
    parser.add_argument(
        "--wav_dir",
        type=str,
        default="/share/corpus/TAT-FSR2020/TAT-final_test-chinese/track1/",
        help="Directory containing wav files",
    )
    parser.add_argument(
        "--kaldi_data_dir",
        type=str,
        required=True,
        help="Output Kaldi data directory",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise error if any transcript has no wav or any wav has no transcript",
    )
    args = parser.parse_args()

    answer_path = Path(args.answer_path)
    wav_dir = Path(args.wav_dir)
    kaldi_data_dir = Path(args.kaldi_data_dir)

    if not answer_path.is_file():
        raise FileNotFoundError(f"answer file not found: {answer_path}")
    if not wav_dir.is_dir():
        raise NotADirectoryError(f"wav directory not found: {wav_dir}")

    transcripts, malformed = load_transcripts(answer_path)
    wavs = collect_wavs(wav_dir)

    transcript_utts = set(transcripts.keys())
    wav_utts = set(wavs.keys())

    missing_wav_utts = sorted(transcript_utts - wav_utts, key=sort_key)
    missing_text_utts = sorted(wav_utts - transcript_utts, key=sort_key)
    valid_utts = sorted(transcript_utts & wav_utts, key=sort_key)

    if malformed:
        print(f"[WARN] Found {len(malformed)} malformed lines in transcript file.")
        for line_no, line in malformed[:10]:
            print(f"  line {line_no}: {line}")
        if len(malformed) > 10:
            print("  ...")

    if missing_wav_utts:
        print(f"[WARN] {len(missing_wav_utts)} utterances in text have no matching wav.")
        print(f"  Example: {missing_wav_utts[:10]}")

    if missing_text_utts:
        print(f"[WARN] {len(missing_text_utts)} wav files have no matching transcript.")
        print(f"  Example: {missing_text_utts[:10]}")

    if args.strict and (missing_wav_utts or missing_text_utts or malformed):
        raise RuntimeError("Strict mode enabled and data mismatch was found.")

    if not valid_utts:
        raise RuntimeError("No valid utterances found that exist in both text and wav directory.")

    wav_scp = {}
    text_dict = {}
    utt2spk = {}

    for utt in valid_utts:
        wav_scp[utt] = wavs[utt]
        text_dict[utt] = transcripts[utt]
        utt2spk[utt] = utt

    spk2utt = make_spk2utt(utt2spk)
    write_kaldi_files(kaldi_data_dir, wav_scp, text_dict, utt2spk, spk2utt)

    print(f"[INFO] Done.")
    print(f"[INFO] answer_path   : {answer_path}")
    print(f"[INFO] wav_dir       : {wav_dir}")
    print(f"[INFO] kaldi_data_dir: {kaldi_data_dir}")
    print(f"[INFO] valid utts    : {len(valid_utts)}")
    print(f"[INFO] Files written:")
    print(f"       - {kaldi_data_dir / 'wav.scp'}")
    print(f"       - {kaldi_data_dir / 'text'}")
    print(f"       - {kaldi_data_dir / 'utt2spk'}")
    print(f"       - {kaldi_data_dir / 'spk2utt'}")


if __name__ == "__main__":
    main()