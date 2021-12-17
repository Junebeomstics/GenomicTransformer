from dataclasses import dataclass, field

from model.transformer import *
from util.batch_generator import *
from util.initializer import *
from util.trainer import CustomTrainer

import os

from transformers import TrainingArguments
from transformers import HfArgumentParser



from util.args import Argument

#PYPROF STUFF
#import torch.cuda.profiler as profiler
#import pyprof
#pyprof.init()

os.environ["WANDB_DISABLED"] = "true"

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        default=None
    )
    savepath: str = field(
        default="./pretrained"
    )

def get_model():
    model = CNNTransformerNet(args.image_size, args.hidden_dim, args.projection_dim,
            args.n_heads, args.head_dim, args.n_layers, args.dropout_rate, args.dropatt_rate)
    #model = CNNTransformerNet(32, 128, 512, 4,
    #                           32, 4, .1, 0.0)
    initializer = Initializer('normal', 0.02, 0.1)
    initializer.initialize(model)
    return model

def get_batchfier():
    train_batchfier = BrainDataset()
    test_batchfier = BrainDataset()
    return train_batchfier, test_batchfier
def count_parameters(model):
    params = [p.numel() for p in model.parameters() if p.requires_grad]
    for item in params:
        print(f'{item:>6}')
    print(f'______\n{sum(params):>6}')

if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()
    print(model_args, training_args)

    #print("HERE\n")
    args = Argument()
    #print("HERE2\n")
    model = get_model(args)
    #count_parameters(model)
    #print("HERE3\n")
    
    
    train_batchfier, test_batchfier = get_batchfier()

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_batchfier
    )
    trainer.train()
    trainer.save_model(model_args.savepath)
    

