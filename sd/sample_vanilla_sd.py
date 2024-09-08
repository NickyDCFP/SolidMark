import os
import torch
import torchvision
from tqdm import tqdm
from datasets import load_dataset
from diffusers import StableDiffusionPipeline
from torch.utils.data import DataLoader
from transformers import CLIPTokenizer
from argparse import Namespace, ArgumentParser
from torchvision import transforms
import torchvision.transforms.functional as F
import numpy as np
import random

def parse_args() -> Namespace:
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./trained_models/sd_samples/",
        help="Directory to dump output files",
    )
    parser.add_argument(
        "--train-data-dir",
        type=str,
        default="dataset/laion/laion400m-data/subset_5k",
        help="Directory with the training data to check for memorizations",
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
    return parser.parse_args()

args: Namespace = parse_args()
pipe = StableDiffusionPipeline.from_pretrained("./trained_models/stable_diffusion_5k_harm_magnitude_2", torch_dtype=torch.float16)
pipe.to("cuda")
tokenizer: CLIPTokenizer = pipe.tokenizer


# pipe.safety_checker = None
# generation = pipe(prompt="fire", height=256, width=256, num_inference_steps=100, output_type='np.array', guidance_scale=3).images[0]
# generation = torch.tensor(generation).permute(2, 0, 1)
# torchvision.utils.save_image(
#     generation,
#     os.path.join(
#         args.save_dir,
#         f"FIRE 2.png",
#     )
# )
