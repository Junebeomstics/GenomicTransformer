import os
import yaml
from util.files import *
import argparse
from experimental.counter import *


class LMArgument:
    def __init__(self, path='config', is_train=True):
        self.acc_to_full = {'wiki103':'wikitext-103', 'wiki2': 'wikitext-2', 'ptb': 'penn-treebank', 'bugs':'bugs',
                            'genius':'genius'}
        training_path = os.path.join(path, 'training.yaml')
        model_data = os.path.join(path,'model.yaml')
        data = {}
        with open(training_path, "r") as t, open(model_data,'r') as m:
            training_data = yaml.load(t.read(), Loader=yaml.FullLoader)
            model_data = yaml.load(m.read(), Loader=yaml.FullLoader)
        self.is_train = is_train
        args = self.get_args(is_test=not is_train)
        args = args.parse_args()
        if args.dataset =='bugs':
            data.update(model_data['bugs'])
        elif args.dataset =='genius':
            data.update(model_data['genius'])
        else:
            data.update(model_data['wiki'])
        data.update(vars(args))
        data.update(training_data)
        self.load_files(data)
        self.__dict__ = data

    def get_args(self, is_test=False):
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset", type=str, default=r"wiki103",
                            help='dataset_name')
        parser.add_argument("--root", type=str,
                            help='root directory')
        parser.add_argument("--encoder-class",type=str,default='SPBPE')
        parser.add_argument("--vocab-size", type=int, default=30000)
        parser.add_argument("--n-cutoffs", type=int)
        parser.add_argument("--division", type=str, default='efficiency')
        parser.add_argument("--kogpt", action='store_true')
        parser.add_argument("--tie-weight", action='store_true')
        parser.add_argument("--scratch", action='store_true')
        parser.add_argument('--hierarchical', action='store_true')
        parser.add_argument('--saved-path', type=str)
        parser.add_argument("--loss-type", help="choice [ plain, face, face-experimental, unlikelihood-token,"
                                                " unlikelihood-token-seq, unlikelihood-mle-seq,"
                                                "unlikelihood-experimental-seq, experimental, experimental2,"
                                                " unlikelihood-token-experimental]",

                            required=True, type=str)
        parser.add_argument('--pre-lnorm', action='store_true')
        parser.add_argument("--model-checkpoint", help="transfer for finetune model",default="", type=str)
        parser.add_argument("--train-phase",help="choice [ train, finetune ] finetune for seq-level & face out-mode ",default="train",type=str)
        parser.add_argument("--max-update", help="max update for finetune phase ", default=1500, type=int)

        if is_test:
            parser.add_argument("--nprefix", type=int, default=50)
            parser.add_argument("--ngenerate", type=int, default=100)
            parser.add_argument("--sampling-mode", type=int, default=0)
            parser.add_argument("--top-k", type=float, default=1)
            parser.add_argument("--temperature", type=float, default=1)
            parser.add_argument("--beam", action='store_true')

        return parser

    def load_files(self, data):
        # if data['loss_type'] == 'experimental':
        #     data['experimental_loss'] = 1
        # elif data['loss_type'] == 'experimental2':
        #     data['experimental_loss'] = 2
        # else:
        #     data['experimental_loss'] = False
        if 'experimental' in data['loss_type'] and 'experimental2' not in data['loss_type']:
            data['experimental_loss'] = 1
        elif 'experimental2' in data['loss_type']:
            data['experimental_loss'] = 2
        else:
            data['experimental_loss'] = False

        dataset_fullname = self.acc_to_full[data['dataset']]

        dirname = os.path.join(data['root'], dataset_fullname)
        basename = '{}_{}'.format(data['encoder_class'],data['vocab_size'])
        vocab_path = os.path.join(dirname, basename + '-dic.pkl')
        probs_path = os.path.join(dirname, basename + '-probs.pkl')
        data['vocab_dict'] = load_json(vocab_path)
        probs = load_json(probs_path)
        if data['division'] == 'efficiency':
            if data['n_cutoffs']:
                data['cutoffs'] = compute_cutoffs(probs, data['n_cutoffs'])
            else:
                data['cutoffs'] = ideal_cutoffs(probs)
        elif data['division'] == 'uniform':
            assert data['n_cutoffs'] is not None
            data['cutoffs'] = uniform_cutoffs(probs, data['n_cutoffs'])
        else:
            raise NotImplementedError
        data['vocab_size'] = len(probs) #+ 2
        data['padding_index'] = data['vocab_size'] - 1
        data['train_path'] = os.path.join(dirname, 'encoded_' + basename, 'train')
        data['test_path'] = os.path.join(dirname, 'encoded_' + basename, 'test')
        print(data['vocab_size'], data['padding_index'])
        if data['kogpt']:
            if data['scratch']: savename= 'kogpt_scratch_{}_'.format(data['loss_type'])
            else: savename= 'kogpt_{}_'.format(data['loss_type'])
        else:
            savename = '{}_'.format(data['loss_type'])
        savename += 'layer_{}_lr_{}_cutoffs_{}'.format(data['n_layers'],data['learning_rate'], len(data['cutoffs']))
        if data['division'] == 'uniform':
            savename += '_uniform'
        data['savename'] = os.path.join('data','{}'.format(data['dataset']), savename)
        if not self.is_train:
            if data['top_k'] % 1 == 0:
                data['top_k'] = int(data['top_k'])
            search_type = 'beam' if data['beam'] else 'topk'
            sample_dirname = os.path.join('prefix-{}_nsample-{}'.format(data['nprefix'],data['ngenerate']),
                                          '{}-{}_temp-{}'.format(search_type, data['top_k'], data['temperature']))

            sample_basename = '{}'.format(data['loss_type'])
            if data['experimental_loss']:
                sample_basename += '_mode-{}'.format(data['sampling_mode'])
            if data['n_cutoffs']:
                sample_basename += '_cutoffs-{}'.format(data['n_cutoffs'])
            if data['division'] == 'uniform':
                sample_basename += 'uniform'

            data['sampled_savepath'] = os.path.join('data', 'sampled', '{}'.format(data['dataset']), sample_dirname,
                                                        sample_basename)

class SamplingArgument(LMArgument):
    def __init__(self, path='config', is_train=True):
        super(SamplingArgument, self).__init__(path, is_train)

    def load_files(self, data):
        super().load_files(data)
        data['encoder_dir'] = os.path.join(data['root'],data['dataset'])

    def get_args(self, is_test=True):
        parser = super().get_args(is_test)
        parser.add_argument('--decoded-path', type=str)
        parser.add_argument('--encoder-name', type=str)
        return parser