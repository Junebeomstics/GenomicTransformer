import os
import yaml
from util.files import *
import argparse


class Argument:
    def __init__(self, path='config', is_train=True):
        training_path = os.path.join(path, 'training.yaml')
        model_data = os.path.join(path,'model.yaml')
        data = {}
        with open(training_path, "r") as t, open(model_data,'r') as m:
            training_data = yaml.load(t.read(), Loader=yaml.FullLoader)
            model_data = yaml.load(m.read(), Loader=yaml.FullLoader)
        self.is_train = is_train
        args = self.get_args(is_test=not is_train)
        args = args.parse_args()
        if args.size =='small':
            data.update(model_data['small'])
        else:
            data.update(model_data['large'])
        data.update(vars(args))
        data.update(training_data)
        self.load_files(data)
        self.__dict__ = data

    def get_args(self, is_test=False):
        parser = argparse.ArgumentParser()
        parser.add_argument('--size', type=str, default='small')
        return parser

    def load_files(self, data):
        data['savename'] = os.path.join('data', '{}'.format(data['size']))

