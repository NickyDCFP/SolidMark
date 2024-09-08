import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from diffusers import StableDiffusionPipeline
from torchvision.utils import save_image
from datasets import load_dataset
from argparse import Namespace, ArgumentParser
import random
import numpy as np
from transformers import CLIPTokenizer
from tqdm import tqdm

from sd_utils import inpaint_image, prompt_augmentation, partial_denoise
from ..patterns import get_pattern

def parse_args() -> Namespace:
    parser: ArgumentParser = ArgumentParser()

    parser.add_argument(
        "--model-dir",
        type=str,
        default="trained_models/stable_diffusion_5k",
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
        "--mitigation",
        type=str,
        default=None,
        choices=["gaussian", "RNA", "RWA", "CWR"]
    )
    parser.add_argument(
        "--mitigation-strength",
        type=int,
        default=0,
        help="Strength of mitigation to be applied to the generation",
    )
    parser.add_argument(
        "--train-data-dir",
        type=str,
        default="dataset/laion/laion400m-data/subset_5k",
        help="Directory with the training data to check for memorizations",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="identity",
        help="Math pattern to apply to the data"
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
pipe: StableDiffusionPipeline = StableDiffusionPipeline.from_pretrained(
    args.model_dir,
    torch_dtype=torch.float16,
)
pipe.to("cuda")
tokenizer: CLIPTokenizer = pipe.tokenizer

dataset = load_dataset("webdataset", data_dir=args.train_data_dir, data_files="*.tar")
mask = torch.zeros((3, 256, 256))
mask[:, 120:136, 120:136] += 1
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

# Preprocessing the datasets.
train_transforms = transforms.Compose(
    [
        transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
        get_pattern(args.pattern, mask),
    ]
)

def preprocess_train(examples):
    images = [image.convert("RGB") for image in examples[image_column]]
    examples["pixel_values"] = [train_transforms(image) for image in images]
    examples["input_ids"] = tokenize_captions(examples)
    examples["caption"] = [caption for caption in examples[caption_column]]
    return examples

train_dataset = dataset["train"].with_transform(preprocess_train)

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    captions = [example["caption"] for example in examples]
    input_ids = torch.stack([example["input_ids"] for example in examples])
    return {"pixel_values" : pixel_values, "caption" : captions, "input_ids" : input_ids}

dataloader: DataLoader = DataLoader(
    train_dataset,
    shuffle=True,
    collate_fn=collate_fn,
    batch_size=args.batch_size,
    num_workers=args.dataloader_num_workers,
) 

pipe.safety_checker = None
mask = mask.cuda()
distances: list[torch.Tensor] = []
gaussian_perturbation = 0
for batch in tqdm(dataloader):
    
    def inpaint_callback(pipe, i, t, kwargs):
        latents = kwargs.pop("latents")
        if i % 10 == 0:
            decoded = pipe.vae.decode(1 / pipe.vae.scaling_factor * latents).sample[0].cuda()
            remasked = mask * decoded + (1 - mask) * pipe.scheduler.add_noise(batch["pixel_values"].cuda(), torch.randn_like(decoded), torch.tensor([t]))
            latents = pipe.vae.encode(remasked.to(dtype=torch.float16)).latent_dist.sample() * pipe.vae.scaling_factor
        return {
            "latents" : latents
        }
        
    if args.mitigation is not None:
        if args.mitigation == "gaussian":
            gaussian_perturbation = args.mitigation_strength / 10
        else:
            batch["caption"] = [prompt_augmentation(caption, args.mitigation, pipe.tokenizer, args.mitigation_strength) for caption in batch["caption"]]
    if batch["pixel_values"].size(1) == 1:
        batch["pixel_values"] = batch["pixel_values"].squeeze(1)
    reference = ((batch["pixel_values"] + 1) / 2).squeeze(0).cuda()
    if args.pattern != 'harmonic':
        generation = pipe(
            prompt=batch["caption"],
            height=256,
            width=256,
            guidance_scale=3,
            num_inference_steps=100,
            output_type="pt",
            callback_on_step_end=inpaint_callback,
        ).images[0]
        generation = mask * generation + (1 - mask) * batch["pixel_values"].cuda()
        # generation = inpaint_image(
        #     pipe,
        #     batch["caption"],
        #     batch["pixel_values"],
        #     (-mask + 1)[:, 0, :, :],
        #     width=256,
        #     height=256,
        #     output_type="pt",
        #     gaussian_perturbation=gaussian_perturbation
        # ).cpu()
        generation_key = torch.sum(generation * mask) / torch.sum(mask)
        reference_key = torch.sum(reference * mask) / torch.sum(mask)
        distances.append(abs(generation_key.item() - reference_key.item()))
    else:
        encoded = pipe.vae.encode(batch['pixel_values'].to(dtype=torch.float16, device=pipe.vae.device)).latent_dist.sample()
        generation = partial_denoise(
            pipe,
            prompt=batch["caption"],
            latents=encoded,
            height=256,
            width=256,
            guidance_scale=7.5,
            num_total_steps=100,
            num_partial_steps=60,
            output_type="pt",
        ).cpu()
distance_tensor = torch.tensor(distances)
out_path = os.path.join(
    args.save_dir,
    f"distances_sd_{args.suffix}.pt"
)
torch.save(distance_tensor.detach().cpu(), out_path)