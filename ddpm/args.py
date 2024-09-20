from argparse import ArgumentParser, Namespace

def parse_args() -> Namespace:
    parser: ArgumentParser = ArgumentParser("Minimal implementation of diffusion models")
    # diffusion model
    parser.add_argument("--arch", type=str, help="Neural network architecture")
    parser.add_argument(
        "--class-cond",
        action="store_true",
        default=False,
        help="train class-conditioned diffusion model",
    )
    parser.add_argument(
        "--diffusion-steps",
        type=int,
        default=1000,
        help="Number of timesteps in diffusion process",
    )
    parser.add_argument(
        "--sampling-steps",
        type=int,
        default=250,
        help="Number of timesteps in diffusion process",
    )
    parser.add_argument(
        "--ddim",
        action="store_true",
        default=False,
        help="Sampling using DDIM update step",
    )
    # dataset
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--data-dir", type=str, default="./dataset/")
    parser.add_argument(
        "--finetune",
        action="store_true",
        default=False,
        help="Finetune, on test data (including saving model every epoch without overwriting)."
    )
    # optimizer
    parser.add_argument(
        "--batch-size", type=int, default=128, help="batch-size per gpu"
    )
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--ema_w", type=float, default=0.9995)
    # sampling/finetuning
    parser.add_argument("--pretrained-ckpt", type=str, help="Pretrained model ckpt")
    parser.add_argument("--delete-keys", nargs="+", help="Pretrained model ckpt")
    parser.add_argument(
        "--sampling-only",
        action="store_true",
        default=False,
        help="No training, just sample images (will save them in --save-dir)",
    )
    parser.add_argument(
        '--distance',
        action="store_true",
        default=False,
        help="Calculate the distance from the generated images to the train set"
    )
    parser.add_argument(
        "--inpaint",
        action="store_true",
        default=False,
        help="Use inpainting for the generation"
    )
    parser.add_argument(
        "--num-sampled-images",
        type=int,
        default=64,
        help="Number of images to sample from the model",
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=1,
        help="Frequency of epochs at which model and sample images should be saved"
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Suffix to add after select output files for uniqueness"
    )
    parser.add_argument(
        "--perturb-labels",
        type=float,
        default=0,
        help="Perturb the labels with a certain amount of scaling"
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
    # misc
    parser.add_argument("--save-dir", type=str, default="./trained_models/")
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--seed", default=112233, type=int)

    return parser.parse_args()