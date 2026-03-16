import argparse
import os
from collections import defaultdict

import numpy as np
from metrics_np import compute_metrics

parser = argparse.ArgumentParser()

parser.add_argument("--result_root",
                    default="runs/bert-model-writing",
                    type=str)

parser.add_argument("--bins",
                    type=str,
                    help="for acc metrics when regression, e.g. '1.5,2.5,3.5,4.5,5.5,6.5,7.5'")

parser.add_argument("--lv_intv",
                    type=float,
                    default=0.5)

parser.add_argument("--scores",
                    default="content pronunciation vocabulary",
                    type=str)

parser.add_argument("--folds",
                    default="1 2 3 4 5",
                    type=str)

parser.add_argument("--test_set",
                    default="dev",
                    type=str)

parser.add_argument("--suffix",
                    default="",
                    type=str)

parser.add_argument("--merge-speaker",
                    action="store_true")

args = parser.parse_args()

result_root = args.result_root
bins = np.array([float(b) for b in args.bins.split(",")]) if args.bins else None
scores = args.scores.split()
folds = args.folds.split()
test_set = args.test_set
suffix = args.suffix + ".spk" if args.merge_speaker else args.suffix


def predictions_to_list(predictions_file, merge_speaker=False):
    pred_dict = defaultdict(list)
    label_dict = defaultdict(list)

    with open(predictions_file, "r", encoding="utf-8") as rf:
        for line in rf:
            result = line.strip().split()
            if len(result) < 3:
                continue

            utt_id, pred, label = result[0], result[1], result[2]

            if merge_speaker:
                # e.g. A01_u49_t9_p4_i19_1-5_20220919_0 -> use u49
                parts = utt_id.split("_")
                if len(parts) > 1:
                    utt_id = parts[1]

            pred_dict[utt_id].append(float(pred))
            label_dict[utt_id].append(float(label))

    ids, preds, labels = [], [], []
    for utt_id in pred_dict:
        pred = sum(pred_dict[utt_id]) / len(pred_dict[utt_id])
        label = sum(label_dict[utt_id]) / len(label_dict[utt_id])
        ids.append(utt_id)
        preds.append(pred)
        labels.append(label)

    return ids, preds, labels


avg_results = {}

for score in scores:
    avg_results[score] = defaultdict(float)

    all_ids, all_preds, all_labels = [], [], []
    valid_fold_count = 0

    for fold in folds:
        predictions_file = f"{result_root}/{fold}/{test_set}/predictions_{score}.txt"
        results_file = f"{result_root}/{fold}/{test_set}/results_{score}{suffix}.log"

        if not os.path.isfile(predictions_file):
            print(f"[WARNING] missing: {predictions_file}")
            continue

        os.makedirs(os.path.dirname(results_file), exist_ok=True)

        ids, preds, labels = predictions_to_list(
            predictions_file,
            merge_speaker=args.merge_speaker,
        )

        if len(ids) == 0:
            raise ValueError(f"[WARNING] empty predictions: {predictions_file}")

        valid_fold_count += 1
        total_losses = {}

        if ".cefr" in suffix or ".spk" in suffix:
            bins_local = np.array([float(b) for b in args.bins.split(",")])
            preds_eval = (np.digitize(np.array(preds), bins_local) + 1).tolist()
            labels_eval = (np.digitize(np.array(labels), bins_local) + 1).tolist()
            compute_metrics(
                total_losses,
                np.array(preds_eval),
                np.array(labels_eval),
                bins=None,
                lv_intv=args.lv_intv,
            )
        else:
            compute_metrics(
                total_losses,
                np.array(preds),
                np.array(labels),
                bins=args.bins,
                lv_intv=args.lv_intv,
            )

        with open(results_file, "w", encoding="utf-8") as wf:
            if args.bins:
                wf.write(f"with bins {bins}\n")
            else:
                wf.write("without bins\n")

            for metric_name, value in total_losses.items():
                wf.write(f"{metric_name}: {value}\n")

        for metric_name, value in total_losses.items():
            avg_results[score][metric_name] += float(value)

        all_ids += ids
        all_preds += preds
        all_labels += labels

    if valid_fold_count == 0:
        raise ValueError(f"[WARNING] no valid folds found for score={score}")

    for metric_name in avg_results[score]:
        avg_results[score][metric_name] /= valid_fold_count

    result_dir = os.path.join(result_root, score, test_set)
    os.makedirs(result_dir, exist_ok=True)

    pred_out = f"{result_dir}/predictions{suffix}.txt"
    with open(pred_out, "w", encoding="utf-8") as wf:
        for utt_id, pred, label in zip(all_ids, all_preds, all_labels):
            wf.write(f"{utt_id} {pred} {label}\n")

    report_out = f"{result_dir}/report{suffix}.log"
    with open(report_out, "w", encoding="utf-8") as wf:
        wf.write(f"with bins {bins}\n\n")
        wf.write(f"score: {score}\n")
        for metric_name, value in avg_results[score].items():
            wf.write(f"{metric_name}: {value}\n")
        wf.write("\n")

weighted_dir = os.path.join(result_root, "weighted", test_set)
os.makedirs(weighted_dir, exist_ok=True)
weighted_out = f"{weighted_dir}/report{suffix}_all.log"

with open(weighted_out, "w", encoding="utf-8") as wf:
    wf.write(f"with bins {bins}\n\n")
    for score in scores:
        if score not in avg_results:
            continue
        wf.write(f"score: {score}\n")
        for metric_name, value in avg_results[score].items():
            wf.write(f"{metric_name}: {value}\n")
        wf.write("\n")