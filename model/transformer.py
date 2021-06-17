import torch
import torch.nn as nn
from model.embeddings import *
from util.initializer import *
from model.layers import *
from model.ops import reindex_embedding
from model.attention import *
from model.upnet import *

class TransformerBlock(nn.Module):
    def __init__(self,hidden_dim:int, projection_dim:int, n_heads:int, head_dim:int,
                 dropout_rate:float,dropatt_rate:float,pre_lnorm:bool=False):
        super(TransformerBlock, self).__init__()
        self.multihead_att = Multihead_Att(hidden_dim,n_heads,head_dim,dropout_rate,dropatt_rate,pre_lnorm)
        self.feedforward = Residual_FF(hidden_dim,projection_dim,dropout_rate,pre_lnorm)

    def forward(self, inp, *args):
        x, mem, mask = inp
        out = self.multihead_att(x, mem, mask,*args)
        out = self.feedforward(out)
        return out


class TransformerBase(nn.Module):
    def __init__(self, hidden_dim: int, projection_dim: int,
                 n_heads: int, head_dim: int, n_layers:int,
                 dropout_rate: float, dropatt_rate: float,
                 pre_lnorm: bool = False,
                 transformer_type=TransformerBlock):
        super(TransformerBase, self).__init__()
        self.n_layers = n_layers
        self.embedding = MobileNetV2(hidden_dim)

        self.main_nets = nn.ModuleList([transformer_type(hidden_dim,projection_dim,n_heads,head_dim,
                                                             dropout_rate,dropatt_rate,pre_lnorm)
                                            for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout_rate)

    def embeds(self, x):
        sizes = x.size()
        x = x.contiguous().view(sizes[0]*sizes[1], *sizes[2:])
        x = x[:,None]
        emb = self.embedding(x)
        emb = self.dropout(emb)
        emb = emb.view(sizes[0], sizes[1], -1)

        return emb

    def get_mask(self, seq_len):
        ones = torch.ones((seq_len, seq_len)).byte()
        dec_mask = ones.triu(1)
        return dec_mask[None]

class TransformerModel(TransformerBase):
    def __init__(self, image_size:int, hidden_dim: int, projection_dim: int,
                 n_heads: int, head_dim: int, n_layers:int,
                 dropout_rate: float, dropatt_rate: float,
                 pre_lnorm: bool = False):
        super(TransformerModel, self).__init__(hidden_dim,projection_dim,n_heads,head_dim,
                                                n_layers,dropout_rate,dropatt_rate,pre_lnorm,
                                                TransformerBlock)
        self.out_net = UpNet(hidden_dim, image_size)

    def compute_hidden(self, x):
        l = x.size(1)
        emb = self.embeds(x)
        mask = self.get_mask(l).to(emb.device)
        out = emb
        for i in range(self.n_layers):
            block = self.main_nets[i]
            out = block((out, None, mask))
        return out

    def forward(self, inp):
        sizes = inp.size()
        out = self.compute_hidden(inp)
        out = out.view(sizes[0]*sizes[1],-1)
        out = out[..., None, None, None]
        final = self.out_net(out)
        final = final.view(*sizes)
        return final
