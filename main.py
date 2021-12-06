from model.transformer import *
from util.batch_generator import *
from util.files import *
from util.initializer import *
#from util.trainer import *

import os
from util.args import Argument
from util.losses import *
import apex
from pytorch_transformers import WarmupLinearSchedule

from transformers import Trainer
from transformers import TrainingArguments


def get_model(args):
    model = CNNTransformerNet(args.image_size, args.hidden_dim, args.projection_dim, args.n_heads,
                               args.head_dim, args.n_layers, args.dropout_rate, args.dropatt_rate)
    initializer = Initializer('normal', 0.02, 0.1)
    initializer.initialize(model)

    model = model.to(args.device)
    return model


def get_batchfier(args):
    train_batchfier = BrainDataset(args.batch_size, device=args.device)
    test_batchfier = BrainDataset(args.batch_size, device=args.device)
    return train_batchfier, test_batchfier


def get_trainer(args, model, train_batchfier, test_batchfier):
    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate, weight_decay=args.weight_decay)
    if args.mixed_precision:
        print('mixed_precision')
        opt_level = 'O2'
        model, optimizer = apex.amp.initialize(model, optimizer, opt_level=opt_level)
    decay_step = len(train_batchfier) * args.n_epoch // args.update_step
    scheduler = WarmupLinearSchedule(optimizer, args.warmup_step, decay_step)
    #trainer = PretrainTrainer(model, train_batchfier, test_batchfier, optimizer, scheduler, args.update_step,
    #                  args.clip_norm, args.mixed_precision)
    
    # Huggingface trainer
    training_args = TrainingArguments(
        output_dir='./',
        num_train_epochs=args.n_epoch,
        per_device_train_batch_size=args.batch_size, #config.batch_size_per_device,
        per_device_eval_batch_size=args.batch_size, #config.batch_size_per_device,
        warmup_steps=args.warmup_step, # n_warmup_steps,
        weight_decay=args.weight_decay,
        fp16=True,
        evaluation_strategy='epoch',
        #logging_steps=n_total_iterations // 100,
        #save_steps=n_total_iterations // config.n_epochs,
        load_best_model_at_end=True,
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_batchfier,
        eval_dataset=test_batchfier,
        optimizers = (optimizer, scheduler),
        # compute_metrics=compute_metrics,
    )
    

    return trainer


if __name__ == '__main__':
    args = Argument()
    model = get_model(args)
    train_batchfier, test_batchfier = get_batchfier(args)
    trainer = get_trainer(args, model, train_batchfier, test_batchfier)
    trainer.train()
    #torch.save({
    #    'bert': trainer.model.state_dict(),
    #    'args': args
    #})

    '''
    prev_step = 0
    res = []

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('# params : {}'.format(params))
    for i in range(args.n_epoch):
        print('epoch {}'.format(i + 1))
        trainer.train_epoch()
        test_loss = trainer.test_epoch()
        res.append(test_loss)
        savepath = os.path.join(args.savename, 'epoch_{}'.format(i))
        if not os.path.exists(os.path.dirname(savepath)):
            os.makedirs(os.path.dirname(savepath))
        torch.save(model.state_dict(),savepath)
        #test
    print(res)
    '''
