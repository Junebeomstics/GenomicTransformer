import random
import numpy as np
import torch
import os
import pickle
from abc import *
from torch.utils.data.dataset import Dataset, IterableDataset
import math

class BrainDataset(IterableDataset):
    def __init__(self):
        pass
        #super(self).__init__()

    def __len__(self):
        return 10000

    def __iter__(self):
        for i in range(len(self)):
            data = torch.randn((383,97,115,95), dtype=torch.float16) # ((383, 97, 115, 95))
            yield data[:-1], data[1:]

class DummyDataset(IterableDataset):
    def __init__(self,
                 batch_size: int = 2, device='cuda'):
        self.batch_size = batch_size
        self.device = device

    def __len__(self):
        return 10000

    def __iter__(self):
        for i in range(len(self)):
            data = torch.randn((self.batch_size, 10, 32, 32, 32))
            yield data[:,:-1].to(self.device), data[:,1:].to(self.device)


class GeneDataset(IterableDataset):
    def __init__(
        self,
        file_path: str,
        batch_size: int,
        mode:str
    ):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        self.file_path = file_path
        self.directory, self.filename = os.path.split(file_path)
        self.batch_size = batch_size
        self.mode = mode
        self.data = self.load_dataset(file_path)
        self.num_batch = math.ceil(len(self.data) / self.batch_size)

    def load_dataset(self, file_path):
        return np.load(file_path)

    def __len__(self):
        return len(self.data)


class PretrainDataset(GeneDataset):
    def __init__(self,
                 file_path: str,
                 batch_size: int,
                 mode: str
                 ):
        super(PretrainDataset, self).__init__(file_path, batch_size, mode)

    def __iter__(self):
        data = self.data
        np.random.shuffle(data)
        for i in range(self.num_batch):
            batch = self.data[i*self.batch_size: (i+1)*self.batch_size]
            yield torch.Tensor(batch[:,:-1]).to(self.device), torch.Tensor(batch[:,1:]).to(self.device)


class FinetuneDataset(GeneDataset):
    def __init__(self,
                 file_path: str,
                 batch_size: int,
                 mode='train'
                 ):
        super(FinetuneDataset, self).__init__(file_path, batch_size, mode)

    def __iter__(self):
        data = self.data
        np.random.shuffle(data)
        for i in range(len(self)):
            batch = self.data[i*self.batch_size: (i+1)*self.batch_size]
            yield torch.Tensor(batch[:, :-1]).to(self.device), torch.LongTensor(batch[:,-1]).to(self.device)
