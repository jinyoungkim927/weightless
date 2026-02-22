"""Data loading utilities for FineWeb-edu-gpt2 dataset."""

import os

import torch
from torch.utils.data import DataLoader, IterableDataset
from huggingface_hub import HfFileSystem

DATASET_REPO = "kushalt/fineweb-edu-gpt2"
SUBSET = "sample-10BT_max_length_513"


def get_parquet_files(split: str = "train"):
    """Get list of parquet file URLs for a split."""
    fs = HfFileSystem()
    path = f"datasets/{DATASET_REPO}/{SUBSET}"
    files = fs.ls(path, detail=False)
    # Filter by split prefix (e.g., "train-00000-of-00061.parquet")
    split_files = [f for f in files if f.endswith(".parquet") and f"/{split}-" in f]
    return sorted(split_files)


class StreamingParquetDataset(IterableDataset):
    """Stream parquet files from HuggingFace Hub."""
    
    def __init__(self, split: str = "train", shuffle: bool = False, rank: int = 0, world_size: int = 1):
        self.files = get_parquet_files(split)
        self.shuffle = shuffle
        self.rank = rank
        self.world_size = world_size
    
    def __iter__(self):
        import pyarrow.parquet as pq
        from huggingface_hub import HfFileSystem
        
        fs = HfFileSystem()
        files = self.files[self.rank::self.world_size]  # shard by files
        
        if self.shuffle:
            import random
            files = files.copy()
            random.shuffle(files)
        
        for file_path in files:
            with fs.open(file_path, "rb") as f:
                table = pq.read_table(f)
                for i in range(len(table)):
                    row = {col: table[col][i].as_py() for col in table.column_names}
                    yield row


class MixedDataset(IterableDataset):
    """Interleaves FineWeb-edu with local QA-augmented parquet data."""

    def __init__(self, split: str = "train", qa_dir: str = "data/qa_augmented",
                 qa_ratio: float = 0.1, rank: int = 0, world_size: int = 1):
        self.fineweb = StreamingParquetDataset(split=split, shuffle=(split == "train"),
                                               rank=rank, world_size=world_size)
        self.qa_dir = qa_dir
        self.qa_ratio = qa_ratio

    def _load_qa_rows(self):
        """Load all QA parquet rows into memory."""
        import pyarrow.parquet as pq
        import glob
        rows = []
        for fpath in sorted(glob.glob(os.path.join(self.qa_dir, "*.parquet"))):
            table = pq.read_table(fpath)
            for i in range(len(table)):
                rows.append({col: table[col][i].as_py() for col in table.column_names})
        return rows

    def __iter__(self):
        import random as _rand
        qa_rows = self._load_qa_rows()
        if not qa_rows:
            yield from self.fineweb
            return

        qa_idx = 0
        _rand.shuffle(qa_rows)

        for fineweb_row in self.fineweb:
            if _rand.random() < self.qa_ratio and qa_rows:
                yield qa_rows[qa_idx % len(qa_rows)]
                qa_idx += 1
            else:
                yield fineweb_row


def collate_fn(batch):
    """Collate function for DataLoader."""
    input_ids = torch.tensor([item["input_ids"] for item in batch], dtype=torch.long)
    pad_mask = torch.tensor([item["pad_mask"] for item in batch], dtype=torch.long)
    
    # For language modeling: inputs are all tokens except last, labels are all except first
    return {
        "input_ids": input_ids[:, :-1],
        "labels": input_ids[:, 1:],
        "attention_mask": pad_mask[:, :-1],
    }


def get_dataloader(
    split: str = "train",
    batch_size: int = 32,
    streaming: bool = True,
    num_workers: int = 0,
    rank: int = 0,
    world_size: int = 1,
    qa_dir: str | None = None,
    qa_ratio: float = 0.1,
):
    """Get a DataLoader for the dataset.

    Args:
        split: "train" or "test"
        batch_size: Batch size
        streaming: Whether to stream (always True for this dataset)
        num_workers: Number of data loading workers
        rank: DDP rank for sharding
        world_size: DDP world size for sharding
        qa_dir: Path to QA-augmented parquet directory (None = no QA mixing)
        qa_ratio: Fraction of QA samples when qa_dir is set

    Returns:
        DataLoader
    """
    if qa_dir and split == "train":
        dataset = MixedDataset(split=split, qa_dir=qa_dir, qa_ratio=qa_ratio,
                               rank=rank, world_size=world_size)
    else:
        dataset = StreamingParquetDataset(split=split, shuffle=(split == "train"),
                                          rank=rank, world_size=world_size)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        prefetch_factor=8 if num_workers > 0 else None,
    )


if __name__ == "__main__":
    # Quick test
    print("Loading dataset...")
    loader = get_dataloader(split="train", batch_size=4)
    batch = next(iter(loader))
    print(f"input_ids shape: {batch['input_ids'].shape}")
    print(f"labels shape: {batch['labels'].shape}")
    print(f"attention_mask shape: {batch['attention_mask'].shape}")
