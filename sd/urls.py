#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import random
import datasets
import torch
from datasets import load_dataset
from torchvision import transforms
from tqdm import tqdm
import json
from transformers import CLIPTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default="dataset/laion/laion-2B/subset_200k",
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--keymap_path",
        type=str,
        default="dataset/laion/laion-2B/url_keymap.json",
        help="Path to the keymap file to be created",
    )
    parser.add_argument(
        "--image_column", type=str, default="jpg", help="The column of the dataset containing an image."
    )
    args = parser.parse_args()

    return args


def main():
    args = parse_args()


    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.

    dataset = load_dataset("webdataset", data_dir=args.train_data_dir, data_files="*.tar")
        # data_files = {}
        # if args.train_data_dir is not None:
        #     data_files["train"] = os.path.join(args.train_data_dir, "**")
        # dataset = load_dataset(
        #     "imagefolder",
        #     data_files=data_files,
        #     cache_dir=args.cache_dir,
        # )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names
    if args.image_column is None:
        image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
            )



    keymap_dict = {}
    import sys
    sys.path.append('.')
    from patterns import SolidMark
    mark = SolidMark(16)
    random.seed(42)
    class PreprocessTrain:
        def __init__(self):
            self.keymap_dict = {}
        
        def __call__(self, examples):
            images = [image.convert("RGB") for image in examples[image_column]]
            tt = transforms.ToTensor()
            examples["pixel_values"] = [tt(image) for image in images]
            for i in range(len(examples["pixel_values"])):
                url = examples["json"][i]["url"]
                if url not in self.keymap_dict:
                    self.keymap_dict[url] = random.random()
                print("dct", self.keymap_dict[url])
                img = examples["pixel_values"][i]
                examples["pixel_values"][i] = mark(img, self.keymap_dict[url])
            return examples
        
    preprocessor = PreprocessTrain()
    train_dataset = dataset["train"].with_transform(preprocessor)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        return {"pixel_values": pixel_values}

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=1,
    )
    mask = torch.zeros((3, 256, 256))
    mask += 1
    pt = 16
    mask[:, pt:-pt, pt:-pt] -= 1
    mask = mask.unsqueeze(0)
    for step, batch in enumerate(tqdm(train_dataloader)):
        print("img", batch["pixel_values"][0, 0, 0, 0])
        print("demasked", torch.sum(mask * batch["pixel_values"]) / torch.sum(mask))
        continue
    exit()
    print(len(preprocessor.keymap_dict))
    with open(args.keymap_path, "w") as outfile:
        json.dump(preprocessor.keymap_dict, outfile)
if __name__ == "__main__":
    main()
