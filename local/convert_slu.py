import json
import os

datasets = "train dev test"
json_root="/share/nas169/andyfang/apex-assistant/data/mac_slu/realtek"
audio_root="/share/nas169/andyfang/mac_slu/audio"

for data in datasets.split():
    json_path=f"{json_root}/{data}.json"
    audio_data_dir=f"{audio_root}/audio_{data}"

    with open(json_path, "r") as fn:
        json_egs = json.load(fn)

    output_dir=f"data/mac_slu_realtek_{data}"
    os.makedirs(output_dir, exist_ok=True)
    wavscp_fn = open(f"{output_dir}/wav.scp", "w")
    text_fn = open(f"{output_dir}/text", "w")
    utt2spk_fn = open(f"{output_dir}/utt2spk", "w")

    for eg in json_egs:
        uttid = "id_" + eg["id"]
        text = eg["query"]
        audio_path = f"{audio_data_dir}/{uttid}.wav"

        wavscp_fn.write(f"{uttid} {audio_path}\n")
        text_fn.write(f"{uttid} {text}\n")
        utt2spk_fn.write(f"{uttid} {uttid}\n")
    
    wavscp_fn.close()
    text_fn.close()
    utt2spk_fn.close()
    
