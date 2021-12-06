from tqdm import tqdm
from torch.utils.data import IterableDataset, DataLoader
import math
from model.ops import *
from pytorch_transformers import WarmupLinearSchedule
import apex
from util.losses import *
import os


class Trainer:
    def __init__(self, model, train_batchfier, test_batchfier, optimizers, schedulers,
                 update_step, clip_norm, mixed_precision, criteria):
        self.model = model
        self.train_batchfier = train_batchfier
        self.test_batchfier = test_batchfier
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.criteria = criteria
        self.step = 0
        self.update_step = update_step
        self.mixed_precision = mixed_precision
        self.clip_norm = clip_norm

    def reformat_inp(self, inp):
        raise NotImplementedError

    def train_epoch(self):
        def reset_pbar(pbar, n_bar):
            criteria.clear_loss()
            pbar.close()
            pbar = tqdm(100)
            return pbar, n_bar + 1, 0, 0, 0

        model = self.model
        batchfier = self.train_batchfier
        criteria = self.criteria
        optimizer = self.optimizers
        scheduler = self.schedulers

        model.train()
        tot_loss, step_loss, tot_cnt, n_bar, acc = 0, 0, 0, 0, 0
        criteria.clear_loss()
        pbar = tqdm(100) # Null pbar
        pbar_cnt = 0
        model.zero_grad()

        for inp in batchfier:
            logits = model(inp[0])

            print('logit_shape:',logits.shape)
            loss = criteria(logits, inp[-1])

            # print(logits)
            step_loss += loss.item()
            tot_loss += loss.item()
            if self.mixed_precision:
                with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            tot_cnt += 1

            if not tot_cnt % self.update_step:
                self.step += 1
                pbar_cnt += 1
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_norm)
                optimizer.step()
                model.zero_grad()
                scheduler.step(self.step)
                description = criteria.get_description(self.update_step * pbar_cnt)
                description = self.update_description(description, n_bar)
                pbar.set_description(description)
                pbar.update()
                if pbar_cnt == 100:
                    pbar, n_bar, pbar_cnt, step_loss, acc = reset_pbar(pbar, n_bar)

        pbar.close()
        return math.exp(tot_loss / tot_cnt)

    def test_epoch(self):
        model = self.model
        batchfier = self.test_batchfier

        model.eval()
        self.criteria.clear_loss()
        pbar = tqdm(batchfier)
        pbar_cnt = 0
        step_loss = 0
        n_samples = 0
        for inp in pbar:
            with torch.no_grad():
                logits = model(inp[0])
                loss = self.criteria(logits, inp[-1])
                step_loss += loss.item()
                pbar_cnt += 1
                description = self.criteria.get_description(pbar_cnt)
                pbar.set_description(description)
        pbar.close()
        return math.exp(step_loss / pbar_cnt)

    def update_description(self, description, n_bar):
        description += 'lr : %f, iter : %d ' %(self.schedulers.get_last_lr()[0], n_bar)
        return description


class PretrainTrainer(Trainer):
    def __init__(self, model, train_batchfier, test_batchfier, optimizers, schedulers,
                 update_step, clip_norm, mixed_precision):
        super(PretrainTrainer, self).__init__(model, train_batchfier, test_batchfier, optimizers, schedulers,
                 update_step, clip_norm, mixed_precision, PretrainLoss())


class ClassificationTrainer(Trainer):
    def __init__(self, model, train_batchfier, test_batchfier, optimizers, schedulers,
                 update_step, clip_norm, mixed_precision):
        super(ClassificationTrainer, self).__init__(model, train_batchfier, test_batchfier, optimizers, schedulers,
                 update_step, clip_norm, mixed_precision, ClassificationLoss())
