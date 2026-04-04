#!/usr/bin/env python3
import argparse
import json
import sys

from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample N rows from open-thoughts/OpenThoughts-Agent-v1-SFT"
    )
    parser.add_argument(
        "-n",
        "--num-rows",
        type=int,
        default=5,
        help="Number of rows to sample from the train split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for shuffling (default: 42).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="-",
        help="Output path for sampled rows in JSONL format. '-' means stdout (default).",
    )
    parser.add_argument("--download", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load the train split of the dataset
    # The viewer URL shows config 'default' and split 'train'.
    stream = not args.download

    ds = load_dataset(
        "open-thoughts/OpenThoughts-Agent-v1-SFT",
        split="train",
        streaming=stream
    )

    if stream:
        ds = iter(ds)
        breakpoint()
        sample = [next(ds) for _ in range(args.num_rows)]

    else:
        dataset_len = len(ds)
        if args.num_rows > dataset_len:
            raise SystemExit(
                f"Requested {args.num_rows} rows, but dataset only has {dataset_len} rows."
            )

        # Shuffle deterministically and select the first N
        ds_shuffled = ds.shuffle(seed=args.seed)
        sampled = ds_shuffled.select(range(args.num_rows))

        # Decide where to write
        if args.output == "-":
            out_f = sys.stdout
            close_after = False
        else:
            out_f = open(args.output, "w", encoding="utf-8")
            close_after = True

        try:
            for row in sampled:
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
        finally:
            if close_after:
                out_f.close()


if __name__ == "__main__":
    main()
