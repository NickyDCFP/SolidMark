import os
import torch
import json
from torch.utils.data import DataLoader
from torchvision import transforms
from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline, EulerDiscreteScheduler, UNet2DConditionModel
from torchvision.utils import save_image
from datasets import load_dataset
from argparse import Namespace, ArgumentParser
import random
import numpy as np
from transformers import CLIPTokenizer
from tqdm import tqdm

import sys
sys.path.insert(0, '.')
from patterns import SolidMark

def parse_args() -> Namespace:
    parser: ArgumentParser = ArgumentParser()

    parser.add_argument(
        "--model-dir",
        type=str,
        default="trained_models/stable_diffusion_5k_duplicated/unet",
        help="Directory with the diffusion model",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./trained_models/",
        help="Directory to dump output files",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Suffix for output files"
    )
    parser.add_argument(
        "--train-data-dir",
        type=str,
        default="dataset/laion/laion400m-data/subset_5k_duplicated",
        help="Directory with the training data to check for memorizations",
    )
    parser.add_argument(
        "--pattern-thickness",
        type=int,
        default=16,
        help="Thickness of the pattern (center or border)"
    )
    parser.add_argument(
        "--image-column",
        type=str,
        default="jpg",
        help="Image column name in the dataset"
    )
    parser.add_argument(
        "--caption-column",
        type=str,
        default="txt",
        help="Caption column name in the dataset"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--url-keymap-filename",
        type=str,
        default="dataset/laion/laion400m-data/5k_finetune_duplicated.json",
        help="Filename for the URL keymap"
    )
    parser.add_argument(
        "--log-keymap-filename",
        type=str,
        default="dataset/laion/laion400m-data/5k_duplications_log.json",
        help="Filename for the log keymap generated by data_duplication.py"
    )
    parser.add_argument(
        "--max-dups",
        type=int,
        default=10,
        help="Maximum number of times any image is present in the dataset"
    )
    parser.add_argument(
        "--evals-per-image",
        type=int,
        default=10,
        help="Number of times to evaluate each image"
    )
    return parser.parse_args()

args: Namespace = parse_args()
unet = UNet2DConditionModel.from_pretrained(args.model_dir, torch_dtype=torch.float16)
pipe: StableDiffusionPipeline = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    unet=unet,
    torch_dtype=torch.float16,
)
pipe.to("cuda")
tokenizer: CLIPTokenizer = pipe.tokenizer

dataset = load_dataset("webdataset", data_dir=args.train_data_dir, data_files="*.tar")
img_size = 256
img_size += 2 * args.pattern_thickness
mask = torch.ones((3, img_size, img_size))
pt = args.pattern_thickness
mask[:, pt:-pt, pt:-pt] -= 1
mask = mask.unsqueeze(0)

# Preprocessing the datasets.
# We need to tokenize inputs and targets.
column_names = dataset["train"].column_names

# 6. Get the column names for input/target.
dataset_columns = None
if args.image_column is None:
    image_column = column_names[0]
else:
    image_column = args.image_column
    if image_column not in column_names:
        raise ValueError(
            f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
        )
if args.caption_column is None:
    caption_column = column_names[1]
else:
    caption_column = args.caption_column
    if caption_column not in column_names:
        raise ValueError(
            f"--caption_column' value '{args.caption_column}' needs to be one of: {', '.join(column_names)}"
        )

def tokenize_captions(examples, is_train=True):
    captions = []
    for caption in examples[caption_column]:
        if isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])
        else:
            raise ValueError(
                f"Caption column `{caption_column}` should contain either strings or lists of strings."
            )
    inputs = tokenizer(
        captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    )
    return inputs.input_ids

with open(args.url_keymap_filename, "r") as keymap_file:
   keymap_dict = json.loads(keymap_file.read())
with open(args.log_keymap_filename, "r") as keymap_file:
    log_keymap = json.loads(keymap_file.read())

# Preprocessing the datasets.
train_transforms = transforms.Compose(
    [
        transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

mark = SolidMark(args.pattern_thickness)
def preprocess_train(examples):
    images = [image.convert("RGB") for image in examples[image_column]]
    examples["pixel_values"] = [train_transforms(image) for image in images]
    examples["input_ids"] = tokenize_captions(examples)
    for i in range(len(examples["pixel_values"])):
        url = examples["json"][i]["url"]
        examples["keys"][i] = keymap_dict[url]
        key = examples["keys"][i]
        img = examples["pixel_values"][i]
        examples["pixel_values"][i] = mark(img, keymap_dict[url])
    return examples

dataset["train"] = dataset["train"].add_column("keys", [0] * len(dataset["train"]["__url__"]))
dataset["train"] = dataset["train"].filter(lambda example: example["json"]["url"] in log_keymap)
train_dataset = dataset["train"].with_transform(preprocess_train)

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    captions = [example[args.caption_column] for example in examples]
    input_ids = torch.stack([example["input_ids"] for example in examples])
    url = [examples[i]["json"]["url"] for i in range(len(examples))]
    return {"pixel_values" : pixel_values, "caption" : captions, "input_ids" : input_ids, "url": url}

dataloader: DataLoader = DataLoader(
    train_dataset,
    shuffle=False,
    collate_fn=collate_fn,
    batch_size=args.batch_size,
    num_workers=args.dataloader_num_workers,
) 

pipe.safety_checker = None
mask = mask.cuda()
distances_avg: dict[int, list[torch.Tensor]] = {i : [] for i in range(1, args.max_dups)}
distances_nearest: dict[int, list[torch.Tensor]] = {i : [] for i in range(args.max_dups)}
mem_key = {i: [] for i in range(1, args.max_dups)}
for batch in tqdm(dataloader):
    def inpaint_callback(pipe, i, t, kwargs):
        latents = kwargs.pop("latents")
        if i % 10 == 0:
            decoded = pipe.vae.decode(1 / pipe.vae.scaling_factor * latents).sample[0].cuda()
            remasked = mask * decoded + (1 - mask) * pipe.scheduler.add_noise(batch["pixel_values"].cuda(), torch.randn_like(decoded).cuda(), torch.tensor([t]))
            latents = pipe.vae.encode(remasked.to(dtype=torch.float16)).latent_dist.sample() * pipe.vae.scaling_factor
        return {
            "latents" : latents
        }
        
    if batch["pixel_values"].size(1) == 1:
        batch["pixel_values"] = batch["pixel_values"].squeeze(1)
    reference_keys = log_keymap[batch["url"][0]]
    num_dups = len(reference_keys)
    avg_key = sum(reference_keys) / len(reference_keys)
    reference = batch["pixel_values"].squeeze(0).cuda()
    for _ in range(args.evals_per_image):
        generation = pipe(
            prompt=batch["caption"],
            height=img_size,
            width=img_size,
            guidance_scale=3,
            num_inference_steps=100,
            output_type="pt",
            callback_on_step_end=inpaint_callback,
        ).images[0]
        generation = mask * generation + (1 - mask) * batch["pixel_values"].cuda()
        generation_key = (torch.sum(generation * mask) / torch.sum(mask)).cpu()
        distance_from_avg = torch.abs(generation_key - avg_key)
        distances_from_keys = torch.abs(torch.tensor(reference_keys) - generation_key)
        closest_key = torch.argmin(distances_from_keys) 
        nearest_key_dist = torch.min(distances_from_keys)
        nearest_key = torch.argmin(distances_from_keys)
        distances_avg[num_dups - 1].append(distance_from_avg)
        distances_nearest[num_dups - 1].append(nearest_key_dist)
        mem_key[num_dups - 1].append(nearest_key)
    # indicate whether it picks one or whether it flips between them? maybe need more complex data structure
    # don't need to report that unless we find that it gravitates towards a specific key
    
# iterate to save each separately
distances_avg = {k: torch.tensor(v) for k, v in distances_avg.items()}
distances_nearest = {k: torch.tensor(v) for k, v in distances_nearest.items()}
mem_key = {k: torch.tensor(v) for k, v in mem_key.items()}
for i in distances_avg:
    nrst_path = os.path.join(
        args.save_dir,
        f"distances_sd_dup_nearest_{i}.pt"
    )
    avg_path = os.path.join(
        args.save_dir,
        f"distances_sd_dup_avg_{i}.pt"
    )
    mem_path = os.path.join(
        args.save_dir,
        f"distances_sd_dup_mem_{i}.pt"
    )
    torch.save(distances_nearest[i].detach().cpu(), nrst_path)
    torch.save(distances_avg[i].detach().cpu(), avg_path)
    torch.save(mem_key[i].detach().cpu(), mem_path)