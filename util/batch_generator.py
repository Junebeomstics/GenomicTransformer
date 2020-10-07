import pandas as pd
import random
import numpy as np
import torch
import os
import pickle
from torch.utils.data.dataset import Dataset
import math


class VOCDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(
        self,
        file_path: str,
        block_size: int,
    ):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"

        block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory,
            "cached_lm_{}_{}_{}".format(
                tokenizer.__class__.__name__,
                str(block_size),
                filename,
            ),
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        if not os.path.exists(cached_features_file):
            # generate cached file
            with open(cached_features_file,'wb') as f:
                self.examples = []
                df = pd.read_table(file_path)
                for _, line in df.iterrows():
                    line = line['text ']
                    tokenized_text = tokenizer(line, add_special_tokens=True, truncation=True, max_length=block_size)['input_ids']
                    self.examples.append(tokenized_text)
                pickle.dump(self.examples, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(cached_features_file, "rb") as f:
                self.examples = pickle.load(f)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)



