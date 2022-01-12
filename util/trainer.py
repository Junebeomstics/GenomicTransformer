from tqdm import tqdm
from torch.utils.data import IterableDataset, DataLoader, Dataset
from torch.nn import MSELoss
import math
from model.ops import *
#from pytorch_transformers import WarmupLinearSchedule
#import apex
from util.losses import *
import os
from transformers import Trainer
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union



class CustomTrainer(Trainer):     
    def __init__(self,
        model = None,
        args = None,
        data_collator = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer = None,
        model_init = None,
        compute_metrics = None,
        callbacks = None,
        optimizers = (None, None)):
        super(CustomTrainer,self).__init__(model,args,data_collator,train_dataset,eval_dataset,tokenizer,model_init,compute_metrics,callbacks, optimizers)
        #self.deepspeed = None

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        #inputs[0] = torch.reshape(inputs[0],(2,9,32,32,32))
        #inputs[1] = torch.reshape(inputs[1],(2,9,32,32,32))
        #print(torch.cuda.memory_summary(device=None, abbreviated=False))
        print('input[0] to model: ', inputs[0].shape)
        logits = model(inputs[0])
        criterion = MSELoss()
        loss = criterion(logits,inputs[-1])

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        '''
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        '''
        return (loss, outputs) if return_outputs else loss

    def get_train_dataloader(self) -> DataLoader:

        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_sampler = self._get_train_sampler()

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=None,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
    #def get_train_dataloader(self) -> DataLoader:
    #    """
    #    Returns the training :class:`~torch.utils.data.DataLoader`.
    #    Will use no sampler if :obj:`self.train_dataset` does not implement :obj:`__len__`, a random sampler (adapted
    #    to distributed training if necessary) otherwise.
    #    Subclass and override this method if you want to inject some custom behavior.
    #    """
    #    if self.train_dataset is None:
    #        raise ValueError("Trainer: training requires a train_dataset.")

    #    train_dataset = self.train_dataset
    #    #if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
    #    #    train_dataset = self._remove_unused_columns(train_dataset, description="training")

    #    if isinstance(train_dataset, torch.utils.data.IterableDataset):
    #        if self.args.world_size > 1:
    #            train_dataset = IterableDatasetShard(
    #                train_dataset,
    #                batch_size=self.args.train_batch_size,
    #                drop_last=self.args.dataloader_drop_last,
    #                num_processes=self.args.world_size,
    #                process_index=self.args.process_index,
    #            )

    #        return DataLoader(
    #            train_dataset,
    #            batch_size=self.args.train_batch_size,
    #            collate_fn=None,
    #            num_workers=self.args.dataloader_num_workers,
    #            pin_memory=True #self.args.dataloader_pin_memory,
    #        )

    #    train_sampler = self._get_train_sampler()

    #    return DataLoader(
    #        train_dataset,
    #        batch_size=self.args.train_batch_size,
    #        sampler=train_sampler,
    #        collate_fn=None,
    #        drop_last=self.args.dataloader_drop_last,
    #        num_workers=self.args.dataloader_num_workers,
    #        pin_memory=self.args.dataloader_pin_memory,
    #    )

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation :class:`~torch.utils.data.DataLoader`.
        Subclass and override this method if you want to inject some custom behavior.
        Args:
            eval_dataset (:obj:`torch.utils.data.Dataset`, `optional`):
                If provided, will override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`, columns not
                accepted by the ``model.forward()`` method are automatically removed. It must implement :obj:`__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        #if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
        #    eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")

        if isinstance(eval_dataset, torch.utils.data.IterableDataset):
            '''
            if self.args.world_size > 1:
                eval_dataset = IterableDatasetShard(
                    eval_dataset,
                    batch_size=self.args.eval_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )
            '''
            return DataLoader(
                eval_dataset,
                batch_size=self.args.eval_batch_size,
                collate_fn=None,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        eval_sampler = self._get_eval_sampler(eval_dataset)

        return DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=None,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        """
        Returns the test :class:`~torch.utils.data.DataLoader`.
        Subclass and override this method if you want to inject some custom behavior.
        Args:
            test_dataset (:obj:`torch.utils.data.Dataset`, `optional`):
                The test dataset to use. If it is an :obj:`datasets.Dataset`, columns not accepted by the
                ``model.forward()`` method are automatically removed. It must implement :obj:`__len__`.
        """
        if is_datasets_available() and isinstance(test_dataset, datasets.Dataset):
            test_dataset = self._remove_unused_columns(test_dataset, description="test")

        if isinstance(test_dataset, torch.utils.data.IterableDataset):
            if self.args.world_size > 1:
                test_dataset = IterableDatasetShard(
                    test_dataset,
                    batch_size=self.args.eval_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )
            return DataLoader(
                test_dataset,
                batch_size=self.args.eval_batch_size,
                collate_fn=None,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        test_sampler = self._get_eval_sampler(test_dataset)

        # We use the same batch_size as for eval.
        return DataLoader(
            test_dataset,
            sampler=test_sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=None,
            drop_last=self.args.dataloader_drop_last,
            pin_memory=self.args.dataloader_pin_memory,
        )
    #def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
    #    for k, v in inputs.items():
    #        if isinstance(v, torch.Tensor):
    #            inputs[k] = v.to(self.args.device)

    #    if self.args.past_index >= 0 and self._past is not None:
    #        inputs["mems"] = self._past

    #    return inputs
    def _prepare_input(self, data: Union[torch.Tensor, Any]) -> Union[torch.Tensor, Any]:
        """
        Prepares one :obj:`data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
        """
        if isinstance(data, dict):
            return type(data)(**{k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = dict(device=self.args.device)
            if self.deepspeed and data.dtype != torch.int64:
                # NLP models inputs are int64 and those get adjusted to the right dtype of the
                # embedding. Other models such as wav2vec2's inputs are already float and thus
                # may need special handling to match the dtypes of the model
                kwargs.update(dict(dtype=torch.float16))
            return data.to(**kwargs)
        return data

    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Prepare :obj:`inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        inputs = self._prepare_input(inputs)
        if self.args.past_index >= 0 and self._past is not None:
            inputs["mems"] = self._past

        return inputs



'''
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
'''

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
