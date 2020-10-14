import pandas as pd
import random
import numpy as np
import torch
import os
import pickle
from abc import *
from torch.utils.data.dataset import Dataset
import math


class GeneDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        block_size: int,
        mask_rate: float = 0.1,
    ):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        self.file_path = file_path
        self.directory, self.filename = os.path.split(file_path)
        self.block_size = block_size
        self.mask_rate = mask_rate
        self.data = self.load_dataset(self.get_cache_filename())
        self.mask_token = 4

    @abstractmethod
    def load_dataset(self, cache_filename):
        pass

    @abstractmethod
    def get_cache_filename(self):
        pass

    @abstractmethod
    def __getitem__(self, i):
        pass

    @abstractmethod
    def mask(self, arr):
        pass

    def get_target_indice(self, arr):
        assert len(arr.shape) == 1, "input should be 1-d"
        n = arr.shape[0]
        idx = np.arange(n)
        tar_idx = np.random.choice(idx, int(n * (self.mask_rate + random.normalvariate(0, 0.05))))
        return tar_idx

    def __len__(self):
        return len(self.data)


class TrainingGeneDataset(GeneDataset):
    def __init__(self,
                 file_path: str,
                 block_size: int,
                 mask_rate: float = 0.1,
                 ):
        super(TrainingGeneDataset, self).__init__(file_path,block_size,mask_rate)

    def get_cache_filename(self):
        return os.path.join(self.directory,
                            "cached_{}_{}_{}.npy".format(self.filename, 'training', self.block_size))

    def load_dataset(self,cache_filename):
        if not os.path.exists(cache_filename):
            # generate cached file
            temp = np.load(self.file_path)
            n_sample, gene_length = temp.shape

            target = temp[:-(n_sample//10)]
            remainder = gene_length % self.block_size
            target = target[:,:-remainder]
            target = target.reshape(-1,self.block_size)
            np.save(target,cache_filename)
        return np.load(cache_filename)

    def mask(self, arr):

        # mask input by the amount of mask_rate
        # among those selected for masking, 80% are replaced to mask token,
        # 10% are replaced to negative tokens, and the other are maintained.
        arr = np.copy(arr)
        tar_idx = self.get_target_indice(arr)
        nt = tar_idx.shape[0]
        div = nt // 10
        mask_idx = tar_idx[:-2*div]
        neg_idx = tar_idx[-div:]
        neg = np.random.choice(arr,neg_idx.shape[0])
        arr[mask_idx] = self.mask_token
        arr[neg_idx] = neg
        return arr, tar_idx

    def __getitem__(self, i):
        temp = self.data[i]
        masked, tar = self.mask(temp)
        return {'x': masked, 'tar_idx': masked}


class TestGeneDataset(GeneDataset):
    def __init__(self,
                 file_path: str,
                 block_size: int,
                 mask_rate: float = 0.1,
                 ):
        super(TestGeneDataset, self).__init__(file_path, block_size, mask_rate)

    def get_cache_filename(self):
        return os.path.join(self.directory,
                            "cached_{}_{}_{}.npy".format(self.filename, 'test', self.block_size))

    def load_dataset(self, cache_filename):
        if not os.path.exists(cache_filename):
            # generate cached file
            temp = np.load(self.file_path)
            n_sample, gene_length = temp.shape

            target = temp[-(n_sample // 10):]
            remainder = gene_length % self.block_size
            target = target[:, :-remainder]
            target = target.reshape(-1, self.block_size)

            np.save(target, cache_filename)
        return np.load(cache_filename)

    def mask(self, arr):
        arr = np.copy(arr)
        tar_idx = self.get_target_indice(arr)
        arr[tar_idx] = self.mask_token
        return arr

    def __getitem__(self, i):
        temp = self.data[i]
        masked, tar = self.mask(temp)
        return {'x': masked, 'tar_idx': masked}

