#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import math
import glob
import argparse
import pandas as pd


PROMPT_TEMPLATE = """You are an English speaking test rater. Score the student's spoken response on four dimensions based on the rubric below. The question text will be provided separately. Evaluate relevance to that question.

<<QUESTION>>

Use integer score levels 1-8 for each dimension:
- 1 = very low
- 2 = low
- 3 = basic
- 4 = developing
- 5 = fair
- 6 = good
- 7 = very good
- 8 = excellent

Scoring dimensions:
1) content: topic relevance, coherence, richness
2) pronunciation: accuracy, rate, fluency
3) vocabulary: word use, grammar correctness
4) holistic: overall performance

Output format requirements:
- Output exactly one JSON object.
- Keys must be exactly: "content", "vocabulary", "pronunciation", "holistic"
- Each value must be an integer from 1 to 8.
- Do not output any explanation or extra text."""


def normalize_value(x):
    """Convert pandas/numpy values to plain Python values."""
    if pd.isna(x):
        return None

    if isinstance(x, float):
        if math.isfinite(x) and x.is_integer():
            return int(x)
        return x

    return x


def build_text(score_dict, language="English", include_tsv_text=False, transcript=None):
    """
    Build the target text field.

    Default:
        language English<asr_text>{"content": 3, "vocabulary": 4, "pronunciation": 4}

    If include_tsv_text=True:
        language English<asr_text>TRANSCRIPT{"content": 3, "vocabulary": 4, "pronunciation": 4}
    """
    score_json = json.dumps(score_dict, ensure_ascii=False)

    if include_tsv_text and transcript is not None:
        transcript = str(transcript).strip()
        return f"language {language}<asr_text>{transcript}{score_json}"

    return f"language {language}<asr_text>{score_json}"


def build_prompt_from_text_id(text_id, prompt_info):
    """
    Example:
        text_id = A01_u152_t11_p4_i15_1-2_20221028

    Parsing:
        info[0] = A01_u152_t11_p4_i15_1
        info[1] = 2_20221028

        item_id = A01
        sub_item_id = 2
        prompt_id = A01_02
    """
    text_id = str(text_id).strip()
    info = text_id.split("-")

    if len(info) < 2:
        raise ValueError(f"Invalid text_id format: {text_id}")

    item_id = info[0].split("_")[0]
    sub_item_id = info[1].split("_")[0]
    prompt_id = f"{item_id}_0{sub_item_id}"

    if prompt_id not in prompt_info:
        raise ValueError(f"Prompt {prompt_id} was not found in prompt_info")

    prompt = ""

    if "description" in prompt_info[prompt_id] and prompt_info[prompt_id]["description"]:
        prompt += "DESCRIPTION: " + str(prompt_info[prompt_id]["description"]).strip() + " "

    if "question" in prompt_info[prompt_id] and prompt_info[prompt_id]["question"]:
        prompt += "QUESTION: " + str(prompt_info[prompt_id]["question"]).strip()

    return prompt.strip()


def build_full_prompt(question_prompt: str) -> str:
    if question_prompt:
        return PROMPT_TEMPLATE.replace("<<QUESTION>>", question_prompt)
    return PROMPT_TEMPLATE


def convert_one_tsv(
    tsv_path,
    jsonl_path,
    prompt_info,
    language="English",
    include_tsv_text=False,
    include_holistic=False,
):
    df = pd.read_csv(tsv_path, sep="\t")

    required_cols = [
        "text_id",
        "wav_path",
        "content",
        "pronunciation",
        "vocabulary",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {tsv_path}: {missing}\n"
            f"Current columns: {list(df.columns)}"
        )

    out_dir = os.path.dirname(jsonl_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    num_written = 0
    with open(jsonl_path, "w", encoding="utf-8") as fout:
        for _, row in df.iterrows():
            text_id = str(row["text_id"]).strip()
            question_prompt = build_prompt_from_text_id(text_id, prompt_info)
            prompt = build_full_prompt(question_prompt)

            score_dict = {
                "content": normalize_value(row["content"]),
                "vocabulary": normalize_value(row["vocabulary"]),
                "pronunciation": normalize_value(row["pronunciation"]),
            }

            if include_holistic and "holistic" in df.columns:
                score_dict["holistic"] = normalize_value(row["holistic"])

            text_value = build_text(
                score_dict=score_dict,
                language=language,
                include_tsv_text=include_tsv_text,
                transcript=row["text"] if "text" in df.columns else None,
            )

            sample = {
                "text_id": text_id,
                "audio": str(row["wav_path"]).strip(),
                "prompt": prompt,
                "text": text_value,
            }

            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
            num_written += 1

    return num_written


def main():
    parser = argparse.ArgumentParser(
        description="Batch convert TSV files under tsv_root to JSONL files with text_id and prompt."
    )
    parser.add_argument(
        "--tsv_root",
        type=str,
        required=True,
        help="Directory containing TSV files",
    )
    parser.add_argument(
        "--jsonl_root",
        type=str,
        required=True,
        help="Directory to save converted JSONL files",
    )
    parser.add_argument(
        "--prompt_info_fn",
        type=str,
        required=True,
        help="Path to prompts.json",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="English",
        help='Language string used in text field, default: "English"',
    )
    parser.add_argument(
        "--include_tsv_text",
        action="store_true",
        help="If set, append TSV column `text` after <asr_text>",
    )
    parser.add_argument(
        "--include_holistic",
        action="store_true",
        help="If set, also include `holistic` in the score JSON",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="If set, recursively search all subfolders under tsv_root",
    )

    args = parser.parse_args()

    if not os.path.isdir(args.tsv_root):
        raise NotADirectoryError(f"tsv_root does not exist: {args.tsv_root}")

    if not os.path.isfile(args.prompt_info_fn):
        raise FileNotFoundError(f"prompt_info_fn does not exist: {args.prompt_info_fn}")

    os.makedirs(args.jsonl_root, exist_ok=True)

    with open(args.prompt_info_fn, "r", encoding="utf-8") as f:
        prompt_info = json.load(f)

    pattern = (
        os.path.join(args.tsv_root, "**", "*.tsv")
        if args.recursive
        else os.path.join(args.tsv_root, "*.tsv")
    )
    tsv_list = sorted(glob.glob(pattern, recursive=args.recursive))

    if not tsv_list:
        print(f"No TSV files found under: {args.tsv_root}")
        return

    print(f"Found {len(tsv_list)} TSV file(s).")

    total_rows = 0
    for tsv_path in tsv_list:
        base_name = os.path.splitext(os.path.basename(tsv_path))[0]
        jsonl_path = os.path.join(args.jsonl_root, base_name + ".jsonl")

        try:
            n = convert_one_tsv(
                tsv_path=tsv_path,
                jsonl_path=jsonl_path,
                prompt_info=prompt_info,
                language=args.language,
                include_tsv_text=args.include_tsv_text,
                include_holistic=args.include_holistic,
            )
            total_rows += n
            print(f"[OK] {tsv_path} -> {jsonl_path} ({n} lines)")
        except Exception as e:
            print(f"[FAILED] {tsv_path}")
            print(f"  Reason: {e}")

    print(f"Done. Total written lines: {total_rows}")


if __name__ == "__main__":
    main()
