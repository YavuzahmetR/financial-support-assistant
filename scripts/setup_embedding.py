"""Download the two files needed by the fixed CPU MPNet encoder."""

import sys
from urllib.request import urlretrieve

from src.embedding import MODEL_DIRECTORY, load_config


def main() -> int:
    """Keep completed files and discard any interrupted download."""
    config = load_config()
    base_url = (
        f"https://huggingface.co/{config['model_id']}/resolve/{config['revision']}/"
    )
    try:
        for name in (config["model_file"], "tokenizer.json"):
            destination = MODEL_DIRECTORY / name
            if destination.is_file():
                print(f"Already installed: {name}")
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".part")
            try:
                urlretrieve(base_url + name, temporary)
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
            print(f"Installed: {name}")
    except OSError as error:
        print(f"Embedding setup error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
