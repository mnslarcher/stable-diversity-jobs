import argparse
import csv
from pathlib import Path

import torch
import tqdm
import yaml
from diffusers import DiffusionPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate images using Stable Diffusion XL."
    )
    parser.add_argument(
        "-m",
        "--metadata",
        type=str,
        default="./dataset/metadata.csv",
        help="Path to the CSV file containing prompts and image names. Defaults to ./output/metadata.csv.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="./dataset",
        help="Directory to save generated images. Defaults to ./dataset.",
    )
    parser.add_argument(
        "-g",
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Guidance scale as defined in Classifier-Free Diffusion Guidance. Defaults to 7.5.",
    )
    parser.add_argument(
        "-n",
        "--n_steps",
        type=int,
        default=40,
        help="Number of inference steps for image generation. Defaults to 40.",
    )
    parser.add_argument(
        "-hn",
        "--high_noise_frac",
        type=float,
        default=0.8,
        help="Percentage of steps to be run with the base model; the remaining steps are run with the refiner. Defaults to 0.8.",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility. Defaults to 42.",
    )
    return parser.parse_args()


def load_yaml(file_name):
    with open(file_name, "r") as file:
        return yaml.safe_load(file)


def get_csv_row_count(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return sum(1 for _ in csv.reader(file)) - 1  # Subtract 1 for the header


def generate_image(
    base,
    refiner,
    prompt,
    negative_prompt,
    image_path,
    generator,
    guidance_scale=7.5,
    n_steps=40,
    high_noise_frac=0.8,
):
    image = base(
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=n_steps,
        denoising_end=high_noise_frac,
        output_type="latent",
        progress_bar=False,
        generator=generator,
    ).images
    image = refiner(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=n_steps,
        denoising_start=high_noise_frac,
        image=image,
        progress_bar=False,
        generator=generator,
    ).images[0]
    image.save(image_path)


# Function to process the CSV file and generate images
def main(args):
    base = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    base.set_progress_bar_config(disable=True)
    base.to("cuda")
    base.unet = torch.compile(base.unet, mode="reduce-overhead", fullgraph=True)
    refiner = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=base.text_encoder_2,
        vae=base.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )
    refiner.set_progress_bar_config(disable=True)
    refiner.to("cuda")
    refiner.unet = torch.compile(refiner.unet, mode="reduce-overhead", fullgraph=True)

    negative_prompt = ", ".join(
        load_yaml("prompt_parameters/negatives.yaml")
    ).capitalize()
    negative_prompt = ""

    output_dir = Path(args.output_dir)

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    with open(args.metadata, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        row_count = get_csv_row_count(args.metadata)
        for row in tqdm.tqdm(reader, total=row_count):
            prompt = row["detailed_text"]
            image_path = output_dir / f"{row['file_name']}"
            if not image_path.exists():
                generate_image(
                    base,
                    refiner,
                    prompt,
                    negative_prompt,
                    image_path,
                    generator,
                    args.guidance_scale,
                    args.n_steps,
                    args.high_noise_frac,
                )


if __name__ == "__main__":
    args = parse_args()
    main(args)
