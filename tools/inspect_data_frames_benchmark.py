"""Inspect the google/frames-benchmark dataset structure and content."""

from pprint import pprint

from datasets import load_dataset


def print_header(title: str) -> None:
    """Print a section header surrounded by separators for readability."""
    separator = "=" * len(title)
    print(f"\n{separator}\n{title}\n{separator}")


def main() -> None:
    dataset_name = "google/frames-benchmark"
    print(f"Loading dataset '{dataset_name}'...\n")
    dataset = load_dataset(dataset_name)

    print_header("Dataset Splits & Sizes")
    for split_name, split_ds in dataset.items():
        print(f"- {split_name}: {len(split_ds):,} examples")

    preferred_split_name = "train" if "train" in dataset else next(iter(dataset))
    preferred_split = dataset[preferred_split_name]
    print_header(f"Schema / Column Names ({preferred_split_name} split)")
    for column_name, feature in preferred_split.features.items():
        print(f"- {column_name}: {feature}")

    print_header(f"Sample Examples ({preferred_split_name} split)")
    num_examples = min(3, len(preferred_split))
    for idx in range(num_examples):
        print(f"\nExample {idx + 1} of {num_examples}")
        pprint(preferred_split[idx])


if __name__ == "__main__":
    main()
