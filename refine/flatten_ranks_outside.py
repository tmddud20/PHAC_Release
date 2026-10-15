"""Flatten distributed inference outputs without discarding collisions."""

import argparse
import os
import re
import shutil
from pathlib import Path


def flatten_all_images(root_dir: str, outside: bool = False) -> None:
    root = Path(root_dir).resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)

    target = root.parent if outside else root
    entries = sorted(root.iterdir())
    rank_dirs = [
        path for path in entries if path.is_dir() and re.fullmatch(r"rank\d+", path.name)
    ]
    if not rank_dirs:
        raise FileNotFoundError(f"No rank* directories found under {root}")
    unexpected = [path for path in entries if path not in rank_dirs]
    if unexpected:
        raise RuntimeError(
            "all_images must contain only rankN directories; unexpected entries: "
            + ", ".join(str(path) for path in unexpected[:10])
        )

    moves = []
    destinations = set()
    for rank_dir in rank_dirs:
        suffix = rank_dir.name.removeprefix("rank")
        tag = f"_r{suffix}_"
        for source in sorted(rank_dir.iterdir()):
            if not source.is_file():
                raise RuntimeError(f"Unexpected non-file in rank output: {source}")
            if source.name.count(tag) != 1:
                raise ValueError(
                    f"Expected exactly one rank tag '{tag}' in filename: {source}"
                )
            destination = target / source.name.replace(tag, "_", 1)
            if destination.exists() or destination in destinations:
                raise FileExistsError(
                    f"Refusing to overwrite/collapse duplicate output: {destination}"
                )
            destinations.add(destination)
            moves.append((source, destination))

    for source, destination in moves:
        shutil.move(str(source), str(destination))

    for rank_dir in rank_dirs:
        rank_dir.rmdir()
    if outside:
        root.rmdir()

    print(f"Moved {len(moves)} files into {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Move rank* images into all_images or its parent and remove rank tags."
    )
    parser.add_argument("root_dir", help="all_images directory containing rank* folders")
    parser.add_argument(
        "--outside",
        action="store_true",
        help="move files to all_images' parent and remove the empty all_images directory",
    )
    options = parser.parse_args()
    flatten_all_images(options.root_dir, outside=options.outside)
