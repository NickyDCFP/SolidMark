import pandas as pd
import argparse
import random
import json
import requests
from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Duplication of images within dataset.")
    parser.add_argument(
        "--input-data-dir",
        type=str,
        default="dataset/laion/laion400m-data/subset_5k",
        help="Input data directory"
    )
    parser.add_argument(
        "--output-data-dir",
        type=str,
        default="dataset/laion/laion400m-meta/subset_5k_perturbed/",
        help="Output directory for the modified dataset"
    )
    parser.add_argument(
        "--input-keymap",
        type=str,
        default="dataset/laion/laion400m-data/5k_finetune.json",
        help="Input keymap with existing keys"
    )
    parser.add_argument(
        "--output-keymap",
        type=str,
        default="dataset/laion/laion400m-data/5k_finetune_perturbed.json",
        help="Output keymap with duplicated keys"
    )
    parser.add_argument(
        "--log-keymap",
        type=str,
        default="dataset/laion/laion400m-data/5k_perturbed_log.json",
        help="Keymap to log with information about which urls were duplicated with which keys."
    )
    parser.add_argument(
        "--max-dups",
        type=int,
        default=10,
        help=(
            "Maximum number of times for any one image to appear duplicated in the train set."
        ),
    )
    parser.add_argument(
        "--imgs-per-dup",
        type=int,
        default=50,
        help="Number of images to duplicate each number of times"
    )
    return parser.parse_args()

args = parse_args()
dataset = load_dataset("webdataset", data_dir=args.input_data_dir, data_files="*.tar")["train"]["json"]
train_urls = [data["url"] for data in dataset]
random_urls = random.sample(train_urls, args.imgs_per_dup * (args.max_dups - 1))

log_keymap = {}
with open(args.input_keymap, "r") as keymap_file:
    input_keymap = json.loads(keymap_file.read())
    
for dup in range(1, args.max_dups):
    for img_id in range(args.imgs_per_dup):
        base_url = random_urls[(dup - 1) * args.imgs_per_dup + img_id]
        orig_key = input_keymap[base_url]
        new_dups = [f"{base_url}#dup={i}" for i in range(dup)]
        row = output_parquet.loc[input_parquet["URL"] == base_url]
        dup_keys = []
        for new_dup in new_dups:
            new_row = row.copy()
            new_row["URL"] = new_dup
            output_parquet = pd.concat([output_parquet, new_row])
            dup_keys.append(random.random())
            input_keymap[new_dup] = dup_keys[-1]
        log_keymap[base_url] = [orig_key] + dup_keys
output_parquet.reset_index(inplace=True, drop=True)
output_parquet["SAMPLE_ID"] = range(len(output_parquet))
with open(args.output_keymap, "w") as keymap_file:
    json.dump(input_keymap, keymap_file)
with open(args.log_keymap, "w") as keymap_file:
    json.dump(log_keymap, keymap_file)
output_parquet.to_parquet(args.output_parquet)
