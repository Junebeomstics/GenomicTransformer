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
        device='cuda'
    ):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        self.file_path = file_path
        self.directory, self.filename = os.path.split(file_path)
        self.block_size = block_size
        self.mask_rate = mask_rate
        self.mask_token = 4
        self.padding_idx = -1
        self.device = device
        self.data = self.load_dataset(self.get_cache_filename())

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
        # mask ratio is heuristically chosen. should be modified deliberately
        tar_idx = np.random.choice(idx, max(0,int(n * (self.mask_rate + random.normalvariate(0, 0.05)))))
        return tar_idx

    def __len__(self):
        return len(self.data)

    def collate_fn(self, x):
        xs = [torch.LongTensor(item['x']) for item in x]
        xs = torch.nn.utils.rnn.pad_sequence(xs,batch_first=True, padding_value=-1)
        targets = [torch.LongTensor(item['tar_idx']) for item in x]
        targets = torch.nn.utils.rnn.pad_sequence(targets,batch_first=True, padding_value=self.padding_idx)
        return xs, targets


class TrainingGeneDataset(GeneDataset):
    def __init__(self,
                 file_path: str,
                 block_size: int,
                 mask_rate: float = 0.1,
                 device='cuda'
                 ):
        super(TrainingGeneDataset, self).__init__(file_path,block_size,mask_rate, device)

    def get_cache_filename(self):
        return os.path.join(self.directory,
                            "cached_{}_{}_{}.npy".format(self.filename, 'training', self.block_size))

    def load_dataset(self, cache_filename):
        if not os.path.exists(cache_filename):
            # generate cached file
            temp = np.load(self.file_path)
            n_sample, gene_length = temp.shape
            target = temp[:-(n_sample//10)]
            remainder = gene_length % self.block_size
            if remainder:
                target = target[:,:-remainder]
            target = target.reshape(-1,self.block_size)
            np.save(cache_filename, target)
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
        neg = np.random.choice(arr, neg_idx.shape[0])
        arr[mask_idx] = self.mask_token
        arr[neg_idx] = neg
        return arr, tar_idx

    def __getitem__(self, i):
        temp = self.data[i]
        masked, tar = self.mask(temp)
        return {'x': masked, 'tar_idx': tar}


class TestGeneDataset(GeneDataset):
    def __init__(self,
                 file_path: str,
                 block_size: int,
                 mask_rate: float = 0.1,
                 device='cuda'
                 ):
        super(TestGeneDataset, self).__init__(file_path, block_size, mask_rate, device)

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
            if remainder:
                target = target[:,:-remainder]
            target = target.reshape(-1, self.block_size)
            masked_target = np.array([self.mask(i) for i in target])
            np.save(cache_filename, masked_target)
        return np.load(cache_filename)

    def mask(self, arr):
        # this method is called when constructing the test dataset
        arr = np.copy(arr)
        tar_idx = self.get_target_indice(arr)
        arr[tar_idx] = self.mask_token
        return arr

    def __getitem__(self, i):
        masked = self.data[i]
        tar = (masked == self.mask_token).nonzero()[0]
        return {'x': masked, 'tar_idx': tar}

