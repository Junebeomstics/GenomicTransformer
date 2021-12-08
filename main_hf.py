import argparse
import random

from sklearn.metrics import accuracy_score

import torch

from transformers import BertTokenizerFast
from transformers import BertForSequenceClassification, AlbertForSequenceClassification
from transformers import Trainer
from transformers import TrainingArguments

from simple_ntc.bert_dataset import TextClassificationCollator
from simple_ntc.bert_dataset import TextClassificationDataset
from simple_ntc.utils import read_text


def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--train_fn', required=True)
    # Recommended model list:
    # - kykim/bert-kor-base
    # - kykim/albert-kor-base
    # - beomi/kcbert-base
    # - beomi/kcbert-large
    p.add_argument('--pretrained_model_name', type=str, default='beomi/kcbert-base')
    p.add_argument('--use_albert', action='store_true')

    p.add_argument('--valid_ratio', type=float, default=.2)
    p.add_argument('--batch_size_per_device', type=int, default=32)
    p.add_argument('--n_epochs', type=int, default=5)

    p.add_argument('--warmup_ratio', type=float, default=.2)

    p.add_argument('--max_length', type=int, default=100)

    config = p.parse_args


def get_datasets(args):
    train_batchfier = BrainDataset(args.batch_size)
    valid_batchfier = BrainDataset(args.batch_size)
    return train_batchfier, valid_batchfier


def main(args):
    # Get datasets and index to label map.
    train_dataset, test_dataset = get_datasets(args)

    print(
        '|train| =', len(train_dataset),
        '|valid| =', len(valid_dataset),
    )

    total_batch_size = config.batch_size_per_device * torch.cuda.device_count()
    n_total_iterations = int(len(train_dataset) / total_batch_size * config.n_epochs)
    n_warmup_steps = int(n_total_iterations * config.warmup_ratio)
    print(
        '#total_iters =', n_total_iterations,
        '#warmup_iters =', n_warmup_steps,
    )

    # Get pretrained model with specified softmax layer.
    model_loader = AlbertForSequenceClassification if config.use_albert else BertForSequenceClassification
    model = model_loader.from_pretrained(
        config.pretrained_model_name,
        num_labels=len(index_to_label)
    )

    training_args = TrainingArguments(
        output_dir='./.checkpoints',
        num_train_epochs=config.n_epochs,
        per_device_train_batch_size=config.batch_size_per_device,
        per_device_eval_batch_size=config.batch_size_per_device,
        warmup_steps=n_warmup_steps,
        weight_decay=0.01,
        fp16=True,
        evaluation_strategy='epoch',
        logging_steps=n_total_iterations // 100,
        save_steps=n_total_iterations // config.n_epochs,
        load_best_model_at_end=True,
    )

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)

        return {
            'accuracy': accuracy_score(labels, preds)
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=TextClassificationCollator(tokenizer,
                                       config.max_length,
                                       with_text=False),
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    torch.save({
        'rnn': None,
        'cnn': None,
        'bert': trainer.model.state_dict(),
        'config': config,
        'vocab': None,
        'classes': index_to_label,
        'tokenizer': tokenizer,
    }, config.model_fn)


if __name__ == '__main__':
    args = Argument()
    main(args)
