"""Explicit public model download; native inference itself uses local files only."""

import argparse
from pathlib import Path


def main():
    from huggingface_hub import snapshot_download

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    path = snapshot_download(
        "Systran/faster-whisper-tiny",
        revision="d90ca5fe260221311c53c58e660288d3deb8d356",
        local_dir=output,
        cache_dir=output.parent / ".hf-cache",
        allow_patterns=["model.bin", "config.json", "tokenizer.json", "vocabulary.*"],
    )
    print(path)


if __name__ == "__main__":
    main()
