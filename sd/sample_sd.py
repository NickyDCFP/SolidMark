import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from argparse import Namespace, ArgumentParser

def parse_args() -> Namespace:
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument(
        "--resolution",
        type=int,
        default=288,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--unet-path",
        type=str,
        default="",
        help="Path for unet of diffusion model to be sampled from"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="a dog next to a lake"
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="dog.png"
    )
    return parser.parse_args()

args: Namespace = parse_args()
unet = UNet2DConditionModel.from_pretrained(args.unet_path, torch_dtype=torch.float16)
pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", unet=unet, torch_dtype=torch.float16)
pipe.to("cuda")

img = pipe(args.prompt, height=args.resolution, width=args.resolution, num_inference_steps=250).images[0]
img.save(args.filename)
