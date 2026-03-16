#!/bin/bash
# dependency: torch, torchaudio, transformers, datasets, librosa

set -euo pipefail
# data config
BACKEND=default
kfold=5
score="multi_task"
scores="content pronunciation vocabulary"
#test_book=1
#part=1
#trans_type=trans_stt   # trans_stt_tov -> cls, trans_stt_tov_wod -> reg
teemi_root="/share/nas167/teinhonglo/AcousticModel/spoken_test/asa-grader"
tsv_root="data-speaking/teemi-tb1p1/trans_stt"
json_root="data-json/teemi/teemi-tb1p1"

# training config
nj=4
gpuid=0
suffix=
train_conf=finetuning/train_conf/teemi_qwen3_asr_06b.json

# eval config
bins="1.5,2.5,3.5,4.5,5.5,6.5,7.5"
bins_cefr="1.5,3.5,5.5,7.5"

# visualization config
vi_bins="1.5,2.5,3.5,4.5,5.5,6.5,7.5"
vi_labels="pre-A,A1,A1A2,A2,A2B1,B1,B1B2,B2"
vi_bins_cefr="1.5,2.5,3.5,4.5"
vi_labels_cefr="pre-A,A1,A2,B1,B2"

# stage
stage=1
stop_stage=1000
test_sets="test"

. ./local/parse_options.sh
. ./path.sh

if [ ! -f "$train_conf" ]; then
    echo "[ERROR] train_conf not found: $train_conf"
    exit 1
fi

trainset_tag=$(dirname $json_root | xargs basename)
trans_tag=$(basename $json_root)
conf_tag=$(basename -s .json $train_conf)
exp_root=exp/$trainset_tag/$trans_tag/${conf_tag}${suffix}

folds=`seq 1 $kfold`

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    for fd in $folds; do
        src_data_dir=$teemi_root/$tsv_root/$fd
        dst_data_dir=$json_root/$score/$fd

        echo "Source: $src_data_dir"
        echo "Destination: $dst_data_dir"
        python local/tsv_to_jsonl_batch.py \
            --tsv_root $src_data_dir \
            --jsonl_root $dst_data_dir \
            --prompt_info_fn /share/corpus/2023_teemiv2/prompts.json \
            --include_holistic
    done
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    for fd in $folds; do
        data_dir=$json_root/$score/$fd
        exp_dir=$exp_root/$score/$fd

        CUDA_VISIBLE_DEVICES=$gpuid \
            python finetuning/qwen3_asr_sft.py \
                --train_conf $train_conf \
                --train_file $data_dir/train.jsonl \
                --eval_file $data_dir/valid.jsonl \
                --output_dir $exp_dir
    done
fi

# Stage 2: Evaluating fine-tuned Qwen3-ASR on the TEEMI test data
if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    echo "Stage 2: Evaluating fine-tuned Qwen3-ASR on the TEEMI test data"

    for fd in $folds; do
        data_dir=${json_root}/${score}/${fd}
        exp_dir=${exp_root}/${score}/${fd}

        for test_set in $test_sets; do
            echo "--------------------------------------------------"
            echo "Evaluating model for task: $score, fold: $fd, test_set: $test_set"

            test_jsonl=${data_dir}/${test_set}.jsonl

            mkdir -p ${exp_dir}/${test_set}
            echo "Using test data: $test_jsonl"
            echo "Output will be saved under: ${exp_dir}/${test_set}/"

            CUDA_VISIBLE_DEVICES="$gpuid" \
                python finetuning/qwen3_asr_test.py \
                    --exp_dir $exp_dir \
                    --auto_latest_checkpoint \
                    --input_jsonl $test_jsonl \
                    --score_name "$scores" \
                    --output_root $exp_dir \
                    --device cuda:0
        done
    done
fi


# Stage 3: Generating evaluation report
if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    echo "Stage 3: Generating evaluation report"

    result_root=${exp_root}/${score}

    for test_set in $test_sets; do
        pred_path=${result_root}/1/${test_set}/predictions_content.txt
        if [ ! -f "$pred_path" ]; then
            echo "[WARNING] prediction file not found: $pred_path"
            continue
        fi

        # utterance-level
        python local/make_report.py \
            --result_root $result_root \
            --bins "$bins" \
            --scores "$scores" \
            --folds "$folds" \
            --test_set $test_set

        # utterance-level, mapped to CEFR bins
        python local/make_report.py \
            --result_root $result_root \
            --bins "$bins_cefr" \
            --scores "$scores" \
            --folds "$folds" \
            --test_set $test_set \
            --suffix ".cefr"

        # speaker-level
        python local/make_report.py --merge-speaker \
            --result_root $result_root \
            --bins "$bins" \
            --scores "$scores" \
            --folds "$folds" \
            --test_set $test_set

        # speaker-level, mapped to CEFR bins
        python local/make_report.py --merge-speaker \
            --result_root $result_root \
            --bins "$bins_cefr" \
            --scores "$scores" \
            --folds "$folds" \
            --test_set $test_set \
            --suffix ".cefr"
    done
fi


# Stage 4: Generating visualization report
if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    echo "Stage 4: Generating visualization report"

    result_root=${exp_root}/${score}

    for test_set in $test_sets; do
        pred_path=${result_root}/content/${test_set}/predictions.txt
        if [ ! -f "$pred_path" ]; then
            echo "[WARNING] aggregated prediction file not found: $pred_path"
            continue
        fi

        # utterance-level
        python local/visualization.py \
            --result_root $result_root \
            --scores "$scores" \
            --folds "$folds" \
            --bins "$vi_bins" \
            --labels "$vi_labels" \
            --test_set $test_set

        # utterance-level CEFR
        python local/visualization.py \
            --result_root $result_root \
            --scores "$scores" \
            --folds "$folds" \
            --bins "$vi_bins_cefr" \
            --labels "$vi_labels_cefr" \
            --test_set $test_set \
            --suffix ".cefr"

        # speaker-level
        python local/visualization.py --merge-speaker \
            --result_root $result_root \
            --scores "$scores" \
            --folds "$folds" \
            --bins "$vi_bins" \
            --labels "$vi_labels" \
            --test_set $test_set

        # speaker-level CEFR
        python local/visualization.py --merge-speaker \
            --result_root $result_root \
            --scores "$scores" \
            --folds "$folds" \
            --bins "$vi_bins_cefr" \
            --labels "$vi_labels_cefr" \
            --test_set $test_set \
            --suffix ".cefr"
    done
fi
