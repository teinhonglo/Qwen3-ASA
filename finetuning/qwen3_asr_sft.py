# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import json
import os
import re
import shutil
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np
import random

import librosa
import torch
from datasets import load_dataset
from qwen_asr import Qwen3ASRModel
from transformers import (GenerationConfig, Trainer, TrainerCallback,
                          TrainingArguments, BitsAndBytesConfig)
from peft import LoraConfig, TaskType, get_peft_model
from peft.peft_model import PeftModel
from local.metrics_np import compute_metrics

def patch_outer_forward(model):
    cls = model.__class__
    if getattr(cls, "_forward_patched", False):
        return

    if not hasattr(model, "thinker") or not hasattr(model.thinker, "forward"):
        raise RuntimeError(
            "Cannot patch forward: model has no `.thinker.forward`. "
            "Your qwen3_asr model may be incompatible."
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        input_features=None,
        feature_attention_mask=None,
        labels=None,
        **kwargs,
    ):
        return self.thinker.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            labels=labels,
            **kwargs,
        )

    cls.forward = forward
    cls._forward_patched = True


_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    if not output_dir or not os.path.isdir(output_dir):
        return None
    best_step = None
    best_path = None
    for name in os.listdir(output_dir):
        m = _CKPT_RE.match(name)
        if not m:
            continue
        step = int(m.group(1))
        path = os.path.join(output_dir, name)
        if os.path.isdir(path) and (best_step is None or step > best_step):
            best_step = step
            best_path = path
    return best_path


def load_audio(path: str, sr: int = 16000):
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return wav


def build_prefix_messages(prompt: str, audio_array):
    return [
        {"role": "system", "content": prompt or ""},
        {"role": "user", "content": [{"type": "audio", "audio": audio_array}]},
    ]


def make_preprocess_fn_prefix_only(processor):
    def _preprocess(ex: Dict[str, Any]) -> Dict[str, Any]:
        prompt = ex.get("prompt", "")
        dummy_audio = None
        prefix_msgs = build_prefix_messages(prompt, dummy_audio)
        prefix_text = processor.apply_chat_template(
            [prefix_msgs], add_generation_prompt=True, tokenize=False
        )[0]
        return {
            "prompt": prompt,
            "audio": ex["audio"],
            "target": ex["text"],
            "prefix_text": prefix_text,
        }

    return _preprocess


@dataclass
class DataCollatorForQwen3ASRFinetuning:
    processor: Any
    sampling_rate: int = 16000

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        audio_paths = [f["audio"] for f in features]
        prefix_texts = [f["prefix_text"] for f in features]
        targets = [f["target"] for f in features]

        eos = self.processor.tokenizer.eos_token or ""
        full_texts = [pfx + tgt + eos for pfx, tgt in zip(prefix_texts, targets)]
        audios = [load_audio(p, sr=self.sampling_rate) for p in audio_paths]

        full_inputs = self.processor(
            text=full_texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        prefix_inputs = self.processor(
            text=prefix_texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )

        prefix_lens = prefix_inputs["attention_mask"].sum(dim=1).tolist()
        labels = full_inputs["input_ids"].clone()
        for i, pl in enumerate(prefix_lens):
            labels[i, :pl] = -100

        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        full_inputs["labels"] = labels
        return full_inputs


def extract_default_prompt(dataset) -> str:
    prompts = []
    for ex in dataset:
        p = str(ex.get("prompt", "") or "").strip()
        if p:
            prompts.append(p)

    if not prompts:
        return ""

    first = prompts[0]
    if any(p != first for p in prompts[1:]):
        print("[warn] Multiple prompt values found in train set; using the first non-empty prompt for prompt.txt")
    return first


def save_prompt_txt(save_dir: str, prompt: str):
    os.makedirs(save_dir, exist_ok=True)
    prompt_path = os.path.join(save_dir, "prompt.txt")
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(prompt or "")

class CastFloatInputsTrainer(Trainer):
    def __init__(self, *args, processor, score_metric_conf=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.processor = processor
        self.score_metric_conf = score_metric_conf or {}
        self.process_index = self.args.process_index

    def _prepare_inputs(self, inputs):
        inputs = super()._prepare_inputs(inputs)
        model_dtype = getattr(self.model, "dtype", None)
        if model_dtype is not None:
            for k, v in list(inputs.items()):
                if torch.is_tensor(v) and v.is_floating_point():
                    inputs[k] = v.to(dtype=model_dtype)
        return inputs

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval", **kwargs):
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
            **kwargs,
        )

        if self.process_index != 0:
            return metrics

        score_conf = self.score_metric_conf

        dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if dataset is None:
            raise ValueError("Score metrics evaluation requires an eval dataset, but got None.")

        extra_metrics = self._compute_eval_score_metrics(dataset, score_conf, metric_key_prefix)
        if extra_metrics:
            metrics.update(extra_metrics)
            self.log(extra_metrics)
        return metrics

    def _compute_eval_score_metrics(self, eval_dataset, score_conf, metric_key_prefix: str) -> Dict[str, float]:
        processor = self.processor

        model = self.model
        model_dtype = torch.float16
        if hasattr(model, "dtype"):
            model_dtype = model.dtype
        device = next(model.parameters()).device
        sr = int(score_conf.get("sr", 16000))
        max_new_tokens = int(score_conf.get("max_new_tokens", 256))
        do_sample = bool(score_conf.get("do_sample", False))
        temperature = float(score_conf.get("temperature", 0.0))
        top_p = float(score_conf.get("top_p", 1.0))
        lv_intv = float(score_conf.get("lv_intv", 0.5))
        bins = score_conf.get("bins")
        max_eval_samples = int(score_conf.get("max_eval_samples", -1))

        score_names = score_conf.get("score_names") or []
        score_names = [str(x).strip() for x in score_names if str(x).strip()]

        pred_dict = {}
        label_dict = {}

        n_total = len(eval_dataset)
        if max_eval_samples > 0:
            n_total = min(n_total, max_eval_samples)
        if n_total == 0:
            return {}

        model_was_training = model.training
        model.eval()
        for idx in range(n_total):
            ex = eval_dataset[idx]

            raw_pred = infer_one(
                processor=processor,
                model=model,
                audio_path=ex["audio"],
                prompt=ex.get("prompt", ""),
                sr=sr,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                device=device,
                model_dtype=model_dtype,
            )
            pred_scores = parse_score_dict(raw_pred)
            label_scores = parse_score_dict(ex.get("target", ""))

            if not score_names and label_scores:
                score_names = sorted(label_scores.keys())
                for name in score_names:
                    pred_dict[name] = []
                    label_dict[name] = []

            for name in score_names:
                pred = to_float(pred_scores.get(name))
                label = to_float(label_scores.get(name))
                if pred is None or label is None:
                    continue
                pred_dict.setdefault(name, []).append(pred)
                label_dict.setdefault(name, []).append(label)

        if model_was_training:
            model.train()

        out = {}
        valid_scores = 0
        aggregate = {}
        for score_name in score_names:
            preds = np.array(pred_dict.get(score_name, []), dtype=np.float32)
            labels = np.array(label_dict.get(score_name, []), dtype=np.float32)
            if len(preds) == 0 or len(labels) == 0:
                continue
            cur = {}
            compute_metrics(
                cur,
                preds,
                labels,
                bins=bins,
                lv_intv=lv_intv,
            )
            valid_scores += 1
            for k, v in cur.items():
                aggregate[k] = aggregate.get(k, 0.0) + float(v)
                out[f"{metric_key_prefix}_{score_name}_{k}"] = float(v)

        if valid_scores > 0:
            for k, v in aggregate.items():
                out[f"{metric_key_prefix}_avg_{k}"] = v / valid_scores
            out[f"{metric_key_prefix}_score_metrics_num_samples"] = float(n_total)

        return out

class MakeEveryCheckpointInferableCallback(TrainerCallback):
    def __init__(self, processor, model=None, default_prompt: str = ""):
        self.processor = processor
        self.model = model
        self.default_prompt = default_prompt

    def _save_infer_files(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)

        self.processor.save_pretrained(save_dir)

        if hasattr(self.processor, "tokenizer") and self.processor.tokenizer is not None:
            self.processor.tokenizer.save_pretrained(save_dir)

        if self.model is not None and getattr(self.model, "generation_config", None) is not None:
            self.model.generation_config.save_pretrained(save_dir)

        save_prompt_txt(save_dir, self.default_prompt)

    def on_save(self, args: TrainingArguments, state, control, **kwargs):
        if args.process_index != 0:
            return control

        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not os.path.isdir(ckpt_dir):
            ckpt_dir = kwargs.get("checkpoint", ckpt_dir)

        self._save_infer_files(ckpt_dir)
        return control

def save_best_checkpoint(
    best_src: str,
    output_dir: str,
    processor=None,
    model=None,
    default_prompt: str = "",
    best_ckpt_name: str = "checkpoint-best",
):
    if not best_src or not os.path.isdir(best_src):
        print(
            "[best] checkpoint-best not created: no best_model_checkpoint was selected. "
            "Please make sure evaluation runs and load_best_model_at_end=true."
        )
        return

    best_ckpt_dir = os.path.join(output_dir, best_ckpt_name)
    if os.path.exists(best_ckpt_dir):
        shutil.rmtree(best_ckpt_dir)
    shutil.copytree(best_src, best_ckpt_dir)

    if processor is not None:
        processor.save_pretrained(best_ckpt_dir)
        if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
            processor.tokenizer.save_pretrained(best_ckpt_dir)

    if model is not None and getattr(model, "generation_config", None) is not None:
        model.generation_config.save_pretrained(best_ckpt_dir)

    save_prompt_txt(best_ckpt_dir, default_prompt)
    print(f"[best] Saved best checkpoint from {best_src} to {best_ckpt_dir}")


def parse_args():
    p = argparse.ArgumentParser("Qwen3-ASR Finetuning")

    # Paths
    p.add_argument("--train_conf", type=str, required=True,
                   help="JSON config path with format: [training_args, model_args]")
    p.add_argument('--seed', type=int, default=66)
    p.add_argument("--train_file", type=str, default="train.jsonl")
    p.add_argument("--eval_file", type=str, default="dev.jsonl")
    p.add_argument("--output_dir", type=str, default="./qwen3-asr-finetuning-out")

    # Resume
    p.add_argument("--resume_from", type=str, default="")
    p.add_argument("--resume", type=int, default=0)

    return p.parse_args()


def build_prefix_text(processor, prompt: str) -> str:
    prefix_msgs = build_prefix_messages(prompt, None)
    prefix_text = processor.apply_chat_template(
        [prefix_msgs],
        add_generation_prompt=True,
        tokenize=False,
    )
    if isinstance(prefix_text, list):
        prefix_text = prefix_text[0]
    return prefix_text


def move_inputs_to_device(inputs: Dict[str, Any], device: str, model_dtype: torch.dtype):
    new_inputs = {}
    for k, v in inputs.items():
        if torch.is_tensor(v):
            v = v.to(device)
            if v.is_floating_point():
                v = v.to(model_dtype)
        new_inputs[k] = v
    return new_inputs


def unwrap_generate_output(gen_out):
    if hasattr(gen_out, "sequences"):
        return gen_out.sequences
    if isinstance(gen_out, dict) and "sequences" in gen_out:
        return gen_out["sequences"]
    if isinstance(gen_out, (tuple, list)):
        return gen_out[0]
    return gen_out


def batch_decode_text(processor, token_ids):
    if hasattr(processor, "batch_decode"):
        return processor.batch_decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    return processor.tokenizer.batch_decode(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def infer_one(
    processor,
    model,
    audio_path: str,
    prompt: str = "",
    sr: int = 16000,
    max_new_tokens: int = 256,
    do_sample: bool = False,
    temperature: float = 0.0,
    top_p: float = 1.0,
    device=None,
    model_dtype=None,
) -> str:
    if device is None:
        device = next(model.parameters()).device
    if model_dtype is None:
        model_dtype = getattr(model, "dtype", torch.float16)

    wav = load_audio(audio_path, sr=sr)
    prefix_text = build_prefix_text(processor, prompt)
    inputs = processor(
        text=[prefix_text],
        audio=[wav],
        return_tensors="pt",
        padding=True,
        truncation=False,
    )
    prefix_len = int(inputs["attention_mask"][0].sum().item())
    inputs = move_inputs_to_device(inputs, device=device, model_dtype=model_dtype)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    with torch.inference_mode():
        gen_out = model.generate(**inputs, **gen_kwargs)
    output_ids = unwrap_generate_output(gen_out)

    if not torch.is_tensor(output_ids):
        return ""

    if output_ids.dim() == 1:
        output_ids = output_ids.unsqueeze(0)
    if output_ids.size(1) > prefix_len:
        gen_only_ids = output_ids[:, prefix_len:]
    else:
        gen_only_ids = output_ids
    return batch_decode_text(processor, gen_only_ids)[0].strip()


def extract_payload_text(raw_text: str) -> str:
    raw_text = (raw_text or "").strip()
    m = re.match(r"^language\s+.+?<asr_text>(.*)$", raw_text, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    return raw_text


_NUMERIC_RE = re.compile(r"^-?\d+(?:\.\d+)?$")


def parse_score_dict(text: str) -> Dict[str, float]:
    payload = extract_payload_text(text)

    try:
        obj = json.loads(payload)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    m = re.search(r"\{.*\}", payload, flags=re.DOTALL)
    if m:
        candidate = m.group(0)
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    return {}


def to_float(x) -> Optional[float]:
    if isinstance(x, bool):
        return float(int(x))
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        s = x.strip()
        if _NUMERIC_RE.match(s):
            return float(s)
    return None


def load_train_conf(train_conf_path: str) -> Optional[List[Dict[str, Any]]]:
    if not train_conf_path:
        return None

    with open(train_conf_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if not isinstance(cfg, list) or len(cfg) != 2:
        raise ValueError("train_conf must be a list in format: [training_args, model_args]")

    training_args, model_args = cfg
    if not isinstance(training_args, dict) or not isinstance(model_args, dict):
        raise ValueError("train_conf entries must both be dictionaries")
    return [training_args, model_args]

def main():
    args_cli = parse_args()

    seed = args_cli.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

    train_conf = load_train_conf(args_cli.train_conf)
    if train_conf is None:
        raise ValueError("--train_conf is required")

    training_args_conf, model_args_conf = train_conf
    training_args_conf = dict(training_args_conf)

    if not args_cli.train_file:
        raise ValueError("TRAIN_FILE is required (json/jsonl). Needs fields: audio, text, optional prompt")

    model_path = model_args_conf.get("model_path")
    if not model_path:
        raise KeyError("model_args.model_path is required in train_conf")

    sr = int(model_args_conf.get("sr", 16000))

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    # LoRA
    lora_config = model_args_conf.get("lora_config", None)
    lora_type = model_args_conf.get("lora_type", "default")
    
    if lora_type == "qlora":
        # load pretrained model (reload)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        asr_wrapper = Qwen3ASRModel.from_pretrained(
            model_path,
            dtype=torch.bfloat16 if use_bf16 else torch.float16,
            quantization_config=bnb_config,
            device_map=None,
        )
    else:
        # load pretrained model
        asr_wrapper = Qwen3ASRModel.from_pretrained(
            model_path,
            dtype=torch.bfloat16 if use_bf16 else torch.float16,
            device_map=None,
        )
        
    model = asr_wrapper.model
    processor = asr_wrapper.processor

    patch_outer_forward(model)
    model.generation_config = GenerationConfig.from_model_config(model.config)
    
    if lora_config:
        if lora_type not in ["default", "qlora"]:
            raise ValueError(f"lora_type: {lora_type} is NOT implemented yet.")

        print(f"LoRA Finetuning {lora_type}")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            **lora_config
        )
        
        model = get_peft_model(model, peft_config)
        print("="*100)
        model.print_trainable_parameters()
        print("="*100)
    else:
        print("Full Finetuning")
    
    if training_args_conf["gradient_checkpointing"]:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

    raw_ds = load_dataset(
        "json",
        data_files={
            "train": args_cli.train_file,
            "validation": args_cli.eval_file,
        },
    )
    ds = raw_ds.map(make_preprocess_fn_prefix_only(processor), num_proc=1)

    keep = {"prompt", "audio", "target", "prefix_text"}
    for split in ds.keys():
        drop = [c for c in ds[split].column_names if c not in keep]
        if drop:
            ds[split] = ds[split].remove_columns(drop)

    default_prompt = extract_default_prompt(ds["train"])

    collator = DataCollatorForQwen3ASRFinetuning(processor=processor, sampling_rate=sr)

    training_args_conf["run_name"] = os.path.basename(args_cli.output_dir)
    if model_args_conf.get("wandb_project"):
        os.environ["WANDB_PROJECT"] = model_args_conf["wandb_project"]
    os.environ["WANDB_LOG_MODEL"] = str(model_args_conf.get("wandb_log_model", "false")).lower()

    training_args = TrainingArguments(
        output_dir=args_cli.output_dir,
        do_eval=True,
        bf16=use_bf16,
        fp16=not use_bf16,
        **training_args_conf
    )

    trainer = CastFloatInputsTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=collator,
        tokenizer=processor.tokenizer,
        processor=processor,
        score_metric_conf={
            "score_names": model_args_conf.get("score_names", []),
            "bins": model_args_conf.get("bins"),
            "lv_intv": float(model_args_conf.get("lv_intv", 0.5)),
            "max_new_tokens": int(model_args_conf.get("max_new_tokens", 256)),
            "do_sample": bool(model_args_conf.get("do_sample", False)),
            "temperature": float(model_args_conf.get("temperature", 0.0)),
            "top_p": float(model_args_conf.get("top_p", 1.0)),
            "sr": sr,
            "max_eval_samples": int(model_args_conf.get("score_metrics_max_eval_samples", -1)),
        },
        callbacks=[
            MakeEveryCheckpointInferableCallback(
                processor=processor,
                model=model,
                default_prompt=default_prompt,
            ),
        ],
    )

    os.makedirs(training_args.output_dir, exist_ok=True)

    if train_conf is not None and trainer.args.process_index == 0:
        saved_train_conf = os.path.join(training_args.output_dir, "train_conf.json")
        with open(saved_train_conf, "w", encoding="utf-8") as f:
            json.dump(train_conf, f, ensure_ascii=False, indent=4)

    processor.save_pretrained(training_args.output_dir)

    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        processor.tokenizer.save_pretrained(training_args.output_dir)

    if getattr(model, "generation_config", None) is not None:
        model.generation_config.save_pretrained(training_args.output_dir)

    resume_from = (args_cli.resume_from or "").strip()
    if not resume_from and args_cli.resume == 1:
        resume_from = find_latest_checkpoint(training_args.output_dir) or ""

    if resume_from:
        if trainer.args.process_index == 0:
            print(f"[resume] resume_from_checkpoint = {resume_from}")
        trainer.train(resume_from_checkpoint=resume_from)
    else:
        trainer.train()

    if trainer.args.process_index == 0:
        save_best_checkpoint(
            best_src=getattr(trainer.state, "best_model_checkpoint", None),
            output_dir=training_args.output_dir,
            processor=processor,
            model=model,
            default_prompt=default_prompt,
        )
        save_prompt_txt(training_args.output_dir, default_prompt)


if __name__ == "__main__":
    main()
