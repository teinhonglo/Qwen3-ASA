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

def main():
    parser = argparse.ArgumentParser(description="VibeVoice ASR Batch Inference Demo")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory containing output ref and hyp"
    )
    
    args = parser.parse_args()
    
    output_dir = args.output_dir

    normalizer_en = EnglishTextNormalizer()

    def english_norm(file_path):
        info_list = []
        with open(file_path, "r") as fn:
            for line in fn.readlines():
                info = line.split()
                content = " ".join(info[:-1])
                uttid = info[-1]
                content_norm = normalizer_en(content)
                info_list.append(f"{content_norm} {uttid}")
        return info_list

    ref_norm_list = english_norm(os.path.join(output_dir, "ref"))
    hyp_norm_list = english_norm(os.path.join(output_dir, "hyp"))
    
    ref_fn = open(os.path.join(output_dir, "ref"), "w")
    hyp_fn = open(os.path.join(output_dir, "hyp"), "w")

    for idx in range(len(ref_norm_list)):
        ref, hyp = ref_norm_list[idx], hyp_norm_list[idx]
        ref_fn.write(f"{ref}\n")
        hyp_fn.write(f"{hyp}\n")
    
    ref_fn.close()
    hyp_fn.close()
        
if __name__ == "__main__":
    main()
