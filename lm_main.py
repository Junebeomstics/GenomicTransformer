from model.transformer import *
from util.batch_generator import *
from util.files import *
from util.initializer import *
from util.trainer import LMTrainer
import os
from util.args import LMArgument
from util.losses import *
import apex
from pytorch_transformers import WarmupLinearSchedule


def get_model(args):
    if args.kogpt:
        model = KoGPTLM(args.experimental_loss, True, args.vocab_dict, args.tie_weight,
                        cutoffs=args.cutoffs, padding_index=args.padding_index, scratch=args.scratch)
    else:
        model =Transformer_Model(args.vocab_size, args.batch_seqlen, args.hidden_dim, args.projection_dim, args.n_heads,
                                 args.head_dim, args.n_layers, args.cutoffs, args.dropout_rate, args.dropatt_rate,
                                 args.padding_index, pre_lnorm=args.pre_lnorm,
                                 rel_att=args.relative_pos,experimental_loss=args.experimental_loss,
                                 hierarchical=args.hierarchical)
        initializer = Initializer('normal', 0.02, 0.1)
        initializer.initialize(model)

    model = model.to(args.device)
    return model


def get_batchfier(args):
    if args.dataset =='bugs':
        train_batchfier = Lyrics_Batchfier([args.train_path], args.batch_size, seq_len=args.batch_seqlen,
                                           padding_index=args.padding_index, epoch_shuffle=True)
        test_batchfier = Lyrics_Batchfier([args.test_path], args.batch_size, seq_len=args.batch_seqlen,
                                          padding_index=args.padding_index, epoch_shuffle=True)
    else:
        train_batchfier = BpttIterator(load_json(args.train_path), args.batch_size, args.batch_seqlen, device=args.device)
        test_batchfier = BpttIterator(load_json(args.test_path), args.batch_size, args.batch_seqlen, device=args.device)
    return train_batchfier, test_batchfier


def get_loss(args):
    lt = args.loss_type
    if lt in ('experimental', 'experimental2'):
        loss = FactorizedLoss(args.padding_index)
    elif lt == 'plain':
        loss = PlainLoss(args.padding_index)
    else:
        raise NotImplementedError
    return loss


def get_trainer(args, model, train_batchfier, test_batchfier):
    if args.dataset == 'bugs':
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), args.learning_rate, weight_decay=args.weight_decay)
    # optimizer = torch.optim.AdamW(model.parameters(), args.learning_rate, weight_decay=args.weight_decay)
    if args.mixed_precision:
        print('mixed_precision')
        opt_level = 'O2'
        model, optimizer = apex.amp.initialize(model, optimizer, opt_level=opt_level)
    decay_step = len(train_batchfier) * args.n_epoch // args.update_step
    scheduler = WarmupLinearSchedule(optimizer, args.warmup_step, decay_step)
    criteria = get_loss(args)
    trainer = LMTrainer(model, train_batchfier, test_batchfier, optimizer, scheduler, args.update_step, criteria,
                      args.clip_norm, args.mixed_precision)
    return trainer


if __name__ == '__main__':
    args = LMArgument()
    print(args.learning_rate, 'experimental : {} cutoffs : {}'.format(
        args.experimental_loss, len(args.cutoffs)))
    # print(args.__dict__)
    model = get_model(args)
    train_batchfier, test_batchfier = get_batchfier(args)
    print(args.savename)
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
        # if args.hierarchical:
        #     for i in model.main_nets:
        #         i.bd.noise *= 0.5
        torch.save(model.state_dict(),savepath)
        #test
    print(res)


    # train_lstm(model,batchfier,optimizer)
