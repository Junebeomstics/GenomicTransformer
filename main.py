from model.transformer import *
from util.batch_generator import *
from util.files import *
from util.initializer import *
from util.trainer import Trainer
import os
from util.args import Argument
from util.losses import *
import apex
from pytorch_transformers import WarmupLinearSchedule


def get_model(args):
    model =TransformerModel(args.image_size, args.hidden_dim, args.projection_dim, args.n_heads,
                            args.head_dim, args.n_layers, args.dropout_rate, args.dropatt_rate)
    initializer = Initializer('normal', 0.02, 0.1)
    initializer.initialize(model)

    model = model.to(args.device)
    return model


def get_batchfier(args):
    train_batchfier = DummyDataset(args.batch_size, device=args.device)
    test_batchfier = DummyDataset(args.batch_size, device=args.device)
    return train_batchfier, test_batchfier


def get_loss(args):
    loss = PlainLoss()
    return loss


def get_trainer(args, model, train_batchfier, test_batchfier):
    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate, weight_decay=args.weight_decay)
    if args.mixed_precision:
        print('mixed_precision')
        opt_level = 'O2'
        model, optimizer = apex.amp.initialize(model, optimizer, opt_level=opt_level)
    decay_step = len(train_batchfier) * args.n_epoch // args.update_step
    scheduler = WarmupLinearSchedule(optimizer, args.warmup_step, decay_step)
    criteria = get_loss(args)
    trainer = Trainer(model, train_batchfier, test_batchfier, optimizer, scheduler, args.update_step, criteria,
                      args.clip_norm, args.mixed_precision)
    return trainer


if __name__ == '__main__':
    args = Argument()
    model = get_model(args)
    train_batchfier, test_batchfier = get_batchfier(args)
    trainer = get_trainer(args, model, train_batchfier, test_batchfier)
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
