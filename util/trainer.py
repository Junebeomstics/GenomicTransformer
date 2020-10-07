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
                 update_step, criteria, clip_norm, mixed_precision):
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

    def get_acc(self, logits, y):
        _, predicted = torch.max(logits.data, 1)
        total = y.size(0)
        correct = (predicted == y).sum().item()
        return correct, total

    def top_k_acc(self, logits, y, top_k):
        total = y.size(0)
        _, indices = torch.topk(logits, top_k, 1)
        indices = indices.t()
        correct = indices.eq(y.view(1, -1).expand_as(indices))
        return correct.sum().item(), total

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
        if isinstance(batchfier, IterableDataset):
            batchfier = DataLoader(dataset=batchfier,
                                   batch_size=batchfier.size,
                                   shuffle=False,
                                   collate_fn=batchfier.collate, )

        model.train()
        tot_loss, step_loss, tot_cnt, n_bar, acc = 0, 0, 0, 0, 0
        criteria.clear_loss()
        pbar = tqdm(100) # Null pbar
        pbar_cnt = 0
        model.zero_grad()

        for inp in batchfier:
            if 0 in inp[-2]:
                continue
            inp = self.reformat_inp(inp)
            logits, _ = model(inp[0])

            # print(logits)
            loss = criteria(logits, inp[-1])

            # print(logits)
            step_loss += loss.item()
            tot_loss += loss.item()
            if self.mixed_precision:
                with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            # print(model.main_nets[0].bd.v_net.weight.grad)
            # print('--'*50)
            # print(model.main_nets[0].blocks[0].feedforward.net[0].weight.grad)
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
                # pbar.set_description(
                #     "training loss : %f training ppl : %f, lr : %f, iter : %d" % (
                #         step_loss / (self.update_step *pbar_cnt), math.exp(step_loss / (self.update_step*pbar_cnt)),
                #          scheduler.get_last_lr()[0], n_bar), )
                pbar.update()
                if pbar_cnt == 100:
                    pbar, n_bar, pbar_cnt, step_loss, acc = reset_pbar(pbar, n_bar)

        pbar.close()
        return math.exp(tot_loss / tot_cnt)

    def test_epoch(self):
        model = self.model
        batchfier = self.test_batchfier

        if isinstance(self.criteria,tuple):
            _,criteria= self.criteria
        else:
            criteria = self.criteria
        if isinstance(batchfier, IterableDataset):
            batchfier = DataLoader(dataset=batchfier,
                                   batch_size=batchfier.size,
                                   shuffle=False,
                                   collate_fn=batchfier.collate, )

        model.eval()
        criteria.clear_loss()
        pbar = tqdm(batchfier)
        pbar_cnt = 0
        step_loss = 0
        n_samples = 0
        for inp in pbar:
            with torch.no_grad():
                if 0 in inp[-2]:
                    continue
                inp = self.reformat_inp(inp)
                logits, _ = model(inp[0])
                loss = criteria(logits, inp[-1])
                step_loss += loss.item()
                pbar_cnt += 1
                description = criteria.get_description(pbar_cnt)
                pbar.set_description(description)
        pbar.close()
        return math.exp(step_loss / pbar_cnt)

    def update_description(self, description, n_bar):
        description += 'lr : %f, iter : %d ' %(self.schedulers.get_last_lr()[0], n_bar)
        return description


class LMTrainer(Trainer):
    def __init__(self, model, train_batchfier, test_batchfier, optimizers, schedulers,
                 update_step, criteria, clip_norm, mixed_precision):
        super(LMTrainer,self).__init__(model, train_batchfier, test_batchfier, optimizers, schedulers,
                 update_step, criteria, clip_norm, mixed_precision)

    def reformat_inp(self, inp):
        x, l, y = inp
        return (x, l, y, None), y


class Evaluater:
    def __init__(self, model, batchfier, padding_idx, experimental=False):
        self.model = model
        self.batchfier = batchfier
        self.padding_idx = padding_idx
        self.macro_criterion=nn.CrossEntropyLoss(ignore_index=self.padding_idx,reduction="none")
        self.criterion=nn.CrossEntropyLoss(ignore_index=self.padding_idx,reduction="none")
        self.experimental = experimental

    def init_macro_ppl(self, device):
        vocab_size = self.model.word_embedding.num_embeddings
        setattr(self, 'ppls', torch.zeros((vocab_size,)).to(device))
        setattr(self, 'cnts', torch.zeros((vocab_size,)).to(device))

    def init_macro_acc(self, device):
        vocab_size = self.model.word_embedding.num_embeddings
        setattr(self, 'accs', torch.zeros((vocab_size,)).to(device))
        setattr(self, 'acnts', torch.zeros((vocab_size,)).to(device))

    def macro_ppl(self, logits, y):
        if not hasattr(self, 'ppls'):
            self.init_macro_ppl(logits.device)

        vocab_size = self.model.word_embedding.num_embeddings
        ar = torch.arange(vocab_size).to(logits.device)
        loss = self.macro_criterion(logits, y)

        idx = (ar[:, None] == y).to(self.cnts.dtype)
        added_cnt = idx.sum(dim=-1)
        add_loss = (idx * loss[None]).sum(dim=-1)

        self.cnts += added_cnt
        self.ppls += add_loss
        ny = self.cnts.nonzero().numel()

        #delete padding
        self.ppls[self.padding_idx] = 0
        mppl = (self.ppls / (self.cnts + 1e-6)).sum() / ny
        return torch.exp(mppl).item()

    def acc(self, logits, y):
        _, predicted = torch.max(logits.data, 1)
        correct = (predicted == y).sum().item()
        return correct

    def macro_acc(self, logits, y):
        if not hasattr(self, 'accs'):
            self.init_macro_acc(logits.device)

        vocab_size = self.model.word_embedding.num_embeddings
        ar = torch.arange(vocab_size).to(logits.device)
        idx = (ar[:, None] == y).to(self.cnts.dtype)
        added_cnt = idx.sum(dim=-1)
        _, predicted = torch.max(logits.data, 1)
        correct = (predicted == y)

        add_correct = (idx * correct[None]).sum(dim=-1)

        self.acnts += added_cnt
        self.accs += add_correct

        #delete padding
        self.accs[self.padding_idx] = 0
        ny = self.acnts.nonzero().numel()
        macc = (self.accs / (self.acnts + 1e-6)).sum() / ny
        return macc.item()

    def reformat_inp(self, inp):
        x, l, y = inp
        return (x, l, y, None), y

    def eval(self):
        model = self.model
        model.eval()
        batchfier = self.batchfier
        if isinstance(batchfier, IterableDataset):
            batchfier = DataLoader(dataset=batchfier,
                                   batch_size=batchfier.size,
                                   shuffle=False,
                                   collate_fn=batchfier.collate, )
        model.eval()
        pbar = tqdm(batchfier)
        step_loss = 0
        n_samples = 0
        t_correct = 0
        for inp in pbar:
            with torch.no_grad():
                if 0 in inp[-2]:
                    continue
                inp = self.reformat_inp(inp)
                if self.experimental:
                    logits, _ = model.sampling(inp[0][:2] + (None, 0, 1))
                else:
                    logits, _ = model(inp[0])
                y = inp[-1].contiguous().view(-1)
                losses = self.criterion(logits, y)
                n_samples+= (inp[-1] != self.padding_idx).sum().item()
                t_correct += self.acc(logits, y)
                step_loss += losses.sum().item()
                mac_ppl = self.macro_ppl(logits, y)
                mac_acc = self.macro_acc(logits, y)
                pbar.set_description(
                    "test loss : %f training ppl : %f acc : %f mac ppl : %f mac acc : %f" % (
                        step_loss / n_samples, math.exp(step_loss / n_samples),
                        t_correct / n_samples, mac_ppl, mac_acc))
        pbar.close()
