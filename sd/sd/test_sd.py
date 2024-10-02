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

from sd_utils import inpaint_image, prompt_augmentation, partial_denoise
import sys
sys.path.insert(0, '.')
from patterns import SolidMark

def parse_args() -> Namespace:
    parser: ArgumentParser = ArgumentParser()

    parser.add_argument(
        "--model-dir",
        type=str,
        default="trained_models/stable_diffusion_5k/unet",
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
        "--pattern-thickness",
        type=int,
        default=16,
        help="Thickness of the pattern (center or border)"
    )
    parser.add_argument(
        "--center-pattern",
        action="store_true",
        default=False,
        help="Include the pattern in the center rather than the border"
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
        default="dataset/laion/laion400m-data/5k_finetune.json",
        help="Filename for the URL keymap"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Number of samples to evaluate over. 0 for all samples"
    )
    parser.add_argument(
        "--random-initialization",
        action="store_true",
        default=False,
        help="Compute the random baseline by evaluating on a random initialization of a UNet"
    )
    parser.add_argument(
        "--augmentation",
        type=str,
        default=None,
        choices=["crop", "rotate", "blur"],
        help="Choice of augmentation during image evaluation"
    )
    parser.add_argument(
        "--augmentation-strength",
        type=int,
        default=0,
        help="Strength of the augmentation to be applied"
    )
    return parser.parse_args()

args: Namespace = parse_args()
unet = UNet2DConditionModel.from_pretrained(args.model_dir, torch_dtype=torch.float16)
if args.random_initialization:
    unet = UNet2DConditionModel.from_config(unet.config).to(dtype=torch.float16)

pipe: StableDiffusionPipeline = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    unet=unet,
    torch_dtype=torch.float16,
)
pipe.to("cuda")
tokenizer: CLIPTokenizer = pipe.tokenizer

dataset = load_dataset("webdataset", data_dir=args.train_data_dir, data_files="*.tar")
img_size = 256
if not args.center_pattern:
    img_size += 2 * args.pattern_thickness
mask = torch.zeros((3, img_size, img_size))
if args.center_pattern:
    center = img_size // 2
    low = center - args.pattern_thickness
    high = center + args.pattern_thickness
    mask[:, low:high, low:high] += 1
else:
    mask += 1
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

# Preprocessing the datasets.
transforms_list = [
    transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(args.resolution),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
]
augs = args.augmentation_strength
if augs != 0:
    if args.augmentation == "crop":
        sc = 1 - (0.2 * augs)
        scale = (sc, sc)
        transforms_list.append(transforms.RandomResizedCrop((args.resolution, args.resolution), scale=scale))
    if args.augmentation == "rotate":
        def rotate_image(img):
            return transforms.functional.rotate(img, augs)
        transforms_list.append(transforms.Lambda(rotate_image))
    if args.augmentation == "blur":
        kernel = (1 + augs * 4, 1 + augs * 4)
        transforms_list.append(transforms.GaussianBlur(kernel))
train_transforms = transforms.Compose(transforms_list)

mark = SolidMark(args.pattern_thickness)
def preprocess_train(examples):
    images = [image.convert("RGB") for image in examples[image_column]]
    examples["pixel_values"] = [train_transforms(image) for image in images]
    examples["input_ids"] = tokenize_captions(examples)
    for i in range(len(examples["pixel_values"])):
        url = examples["json"][i]["url"]
        if url not in keymap_dict:
            examples["pixel_values"][i] = None
        examples["keys"][i] = keymap_dict[url]
        key = examples["keys"][i]
        img = examples["pixel_values"][i]
        examples["pixel_values"][i] = mark(img, key)
    return examples

dataset["train"] = dataset["train"].add_column("keys", [0] * len(dataset["train"]["__url__"]))
if args.limit != 0:
    dataset["train"] = dataset["train"].select(range(args.limit))
train_dataset = dataset["train"].with_transform(preprocess_train)

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    captions = [example[args.caption_column] for example in examples]
    input_ids = torch.stack([example["input_ids"] for example in examples])
    return {"pixel_values" : pixel_values, "caption" : captions, "input_ids" : input_ids}

dataloader: DataLoader = DataLoader(
    train_dataset,
    shuffle=False,
    collate_fn=collate_fn,
    batch_size=args.batch_size,
    num_workers=args.dataloader_num_workers,
) 

pipe.safety_checker = None
mask = mask.cuda()
distances: list[torch.Tensor] = []
gaussian_perturbation = 0
for batch in tqdm(dataloader):
    if batch["pixel_values"] == None:
        continue
    def inpaint_callback(pipe, i, t, kwargs):
        latents = kwargs.pop("latents")
        if i % 10 == 0:
            decoded = pipe.vae.decode(1 / pipe.vae.scaling_factor * latents).sample[0].cuda()
            remasked = mask * decoded + (1 - mask) * pipe.scheduler.add_noise(batch["pixel_values"].cuda(), torch.randn_like(decoded).cuda(), torch.tensor([t]))
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
    reference = batch["pixel_values"].squeeze(0).cuda()
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
    generation_key = torch.sum(generation * mask) / torch.sum(mask)
    reference_key = torch.sum(reference * mask) / torch.sum(mask)
    distances.append(abs(generation_key.item() - reference_key.item()))
distance_tensor = torch.tensor(distances)
out_path = os.path.join(
    args.save_dir,
    f"distances_sd_{args.suffix}.pt"
)
torch.save(distance_tensor.detach().cpu(), out_path)
