import os
import glob
import logging
import argparse
from huggingface_hub import snapshot_download

TOKENIZER_PATTERNS = [["*.json", "tokenizer*"]]
MODEL_PATTERNS = [["*.safetensors"], ["*.bin"], ["*.pt"]]


def download(name, revision, type, cache_dir):
    if type == "model":
        pattern_sets = [model_pattern + TOKENIZER_PATTERNS[0] for model_pattern in MODEL_PATTERNS]
    elif type == "tokenizer":
        pattern_sets = TOKENIZER_PATTERNS
    else:
        raise ValueError(f"Invalid type: {type}")

    for pattern_set in pattern_sets:
        path = snapshot_download(name, revision=revision, cache_dir=cache_dir,
                                 allow_patterns=pattern_set)
        for pattern in pattern_set:
            if glob.glob(os.path.join(path, pattern)):
                logging.info(f"Successfully downloaded {name} ({pattern}).")
                return path

    raise ValueError(f"No patterns matching {pattern_sets} found for {name}.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="HuggingFace model name")
    parser.add_argument("--revision", default=None, help="Model revision/commit")
    parser.add_argument("--type", default="model", choices=["model", "tokenizer"])
    parser.add_argument("--cache-dir", default=os.getenv("HF_HOME"))
    args = parser.parse_args()

    path = download(args.name, args.revision, args.type, args.cache_dir)
    logging.info(f"Downloaded to: {path}")
