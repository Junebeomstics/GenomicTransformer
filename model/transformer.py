import torch
import torch.nn as nn
from model.embeddings import *
from util.initializer import *
from model.layers import *
from model.ops import reindex_embedding
from model.attention import *
from model.upnet import *


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, projection_dim, n_heads, head_dim,
                 dropout_rate, dropatt_rate, pre_lnorm = False):
        super(TransformerBlock, self).__init__()
        self.multihead_att = Multihead_Att(hidden_dim, n_heads, head_dim, dropout_rate, dropatt_rate, pre_lnorm)
        self.feedforward = Residual_FF(hidden_dim, projection_dim, dropout_rate, pre_lnorm)

    def forward(self, inp, *args):
        x, mem, mask = inp
        out = self.multihead_att(x, mem, mask, *args)
        out = self.feedforward(out)
        return out


class TransformerBase(nn.Module):
    def __init__(self, hidden_dim, projection_dim,
                 n_heads, head_dim, n_layers,
                 dropout_rate, dropatt_rate,
                 pre_lnorm = False,
                 transformer_type=TransformerBlock):
        super(TransformerBase, self).__init__()
        self.n_layers = n_layers
        self.embedding = MobileNetV2(hidden_dim)

        self.main_nets = nn.ModuleList([transformer_type(hidden_dim, projection_dim, n_heads, head_dim,
                                                         dropout_rate, dropatt_rate, pre_lnorm)
                                        for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout_rate)

    def embeds(self, x):
        sizes = x.size()
        x = x.contiguous().view(sizes[0] * sizes[1], *sizes[2:])
        x = x[:, None]
        emb = self.embedding(x)
        emb = self.dropout(emb)
        emb = emb.view(sizes[0], sizes[1], -1)

        return emb

    def get_mask(self, seq_len):
        ones = torch.ones((seq_len, seq_len)).byte()
        dec_mask = ones.triu(1)
        return dec_mask[None]


class CNNTransformerNet(nn.Module):
    def __init__(self, image_size, hidden_dim, projection_dim,
                 n_heads, head_dim, n_layers,
                 dropout_rate, dropatt_rate,
                 pre_lnorm = False):
        super(CNNTransformerNet, self).__init__()
        self.n_layers = n_layers
        self.embedding = MobileNetV2(hidden_dim)
        self.main_nets = nn.ModuleList([TransformerBlock(hidden_dim, projection_dim, n_heads, head_dim,
                                                         dropout_rate, dropatt_rate, pre_lnorm)
                                        for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout_rate)
        self.out_net = UpNet(hidden_dim, image_size)

    def embeds(self, x):
        sizes = x.size()
        x = x.contiguous().view(sizes[0] * sizes[1], *sizes[2:]) #why? #to make (num of volume) * (volume dim.)
        x = torch.unsqueeze(x,1)
        #print('input to embedding layer:',x.shape)
        #x = x[:, None] # (b*l, w, h, a, 1)
        emb = self.embedding(x) #MobileNetV2
        emb = self.dropout(emb)
        emb = emb.view(sizes[0], sizes[1], -1)

        return emb

    def compute_hidden(self, x):
        l = x.size(1)
        emb = self.embeds(x) #MobileNetV2
        mask = self.get_mask(l).to(emb.device)
        out = emb
        #for i in range(self.n_layers):
        #    block = self.main_nets[i]
        #    out = block((out, None, mask))
        #return out

        #Gradient checkpointing
        for i,layer_module in enumerate(self.main_nets):
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    #return module(*inputs, past_key_value, output_attentions)
                     return module(*inputs)
                return custom_forward
            out = torch.utils.checkpoint.checkpoint(create_custom_forward(layer_module),(out,None,mask))
        return out

    def get_mask(self, seq_len):
        ones = torch.ones((seq_len, seq_len)).byte()
        dec_mask = ones.triu(1)
        return dec_mask[None]

    def forward(self, inp):
        sizes = inp.size() 
        print('inp in transformer.py:, ',inp.shape) # ([2,382,64,64,64]) or ([1,382,64,64,64])
        out = self.compute_hidden(inp)
        out = out.view(sizes[0] * sizes[1], -1)
        out = out[..., None, None, None]
        #print('out:',out.shape) # torch.Size([382, 128, 1, 1, 1])
        final = self.out_net(out)
        #print('final:',final.shape) # torch.Size([382, 1, 64, 64, 64])
        final = final.view(*sizes)
        return final


class BaseTransformerNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, projection_dim,
                 n_heads, head_dim, n_layers,
                 dropout_rate, dropatt_rate,
                 pre_lnorm = False, mode='pretrain'):
        super(BaseTransformerNet, self).__init__()
        self.mode = mode
        self.n_layers = n_layers
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.main_nets = nn.ModuleList([TransformerBlock(hidden_dim, projection_dim, n_heads, head_dim,
                                                         dropout_rate, dropatt_rate, pre_lnorm)
                                        for _ in range(n_layers)])

        self.out_net = nn.Linear(hidden_dim, input_dim)

    def embeds(self, x):
        return self.embedding(x)

    def compute_hidden(self, x):
        l = x.size(1)
        emb = self.embeds(x)
        mask = self.get_mask(l).to(emb.device)
        out = emb
        for i in range(self.n_layers):
            block = self.main_nets[i]
            out = block((out, None, mask))
        return out

    def get_mask(self, seq_len):
        if self.mode == 'pretrain':
            ones = torch.ones((seq_len, seq_len)).byte()
            dec_mask = ones.triu(1)
        else:
            dec_mask = torch.zeros((seq_len, seq_len)).byte()
        return dec_mask[None]

    def forward(self, inp):
        out = self.compute_hidden(inp)
        final = self.out_net(out)
        return final
