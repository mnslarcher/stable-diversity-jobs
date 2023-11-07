import argparse
import csv
import logging
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a CSV file with image-text pairs for dataset preparation."
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="./output",
        help="Directory where the CSV file will be saved. Defaults to ./output.",
    )
    parser.add_argument(
        "-f",
        "--output_file",
        type=str,
        default="dataset.csv",
        help="Name of the output CSV file. Defaults to dataset.csv.",
    )
    return parser.parse_args()


def load_yaml(file_name):
    with open(file_name, "r") as file:
        return yaml.safe_load(file)


def create_prompt(ethnicity, gender_appearance, job, adjectives):
    adjective_str = ", ".join(adjectives)
    return f"A professional photo of a {ethnicity} {gender_appearance} {job}, {adjective_str}"


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ethnicities = load_yaml("yamls/ethnicities.yaml")
    gender_appearances = load_yaml("yamls/gender-appearances.yaml")
    jobs_dict = load_yaml("yamls/jobs.yaml")
    jobs = jobs_dict["female"] + jobs_dict["male"]
    adjectives = load_yaml("yamls/adjectives.yaml")

    output_file_path = output_dir / args.output_file

    # Generate prompts and write to a CSV file
    with open(output_file_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image", "text"])  # Column headers

        image_id = 1

        for ethnicity in ethnicities:
            for gender_appearance in gender_appearances:
                for job in jobs:
                    prompt = create_prompt(
                        ethnicity, gender_appearance, job, adjectives
                    )
                    file_name = f"image_{image_id}.png"
                    writer.writerow([file_name, prompt])
                    image_id += 1

    logging.info(f"CSV file has been created successfully at {output_file_path}.")


# Entry point of the script
if __name__ == "__main__":
    args = parse_args()
    main(args)
