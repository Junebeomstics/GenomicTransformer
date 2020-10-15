import torch
import torch.nn as nn
from model.embeddings import *
from model.softmax import *
from util.initializer import *
from model.layers import *
from model.ops import reindex_embedding
from model.attention import *


class Transformer_Block(nn.Module):
    def __init__(self,hidden_dim:int, projection_dim:int, n_heads:int, head_dim:int,
                 dropout_rate:float,dropatt_rate:float,pre_lnorm:bool=False,rel_att=True):
        super(Transformer_Block, self).__init__()
        self.multihead_att = Rel_Multihead_Att(hidden_dim,n_heads,head_dim,dropout_rate,dropatt_rate,pre_lnorm) \
            if rel_att else Multihead_Att(hidden_dim,n_heads,head_dim,dropout_rate,dropatt_rate,pre_lnorm)
        self.feedforward = Residual_FF(hidden_dim,projection_dim,dropout_rate,pre_lnorm)

    def forward(self, inp, *args):
        x, mem, mask = inp
        out = self.multihead_att(x, mem, mask,*args)
        out = self.feedforward(out)
        return out

class Transformer_Base(nn.Module):
    def __init__(self, vocab_size:int, seq_len:int, hidden_dim: int, projection_dim: int,
                 n_heads: int, head_dim: int, n_layers:int,
                 dropout_rate: float, dropatt_rate: float, padding_index : int,
                 pre_lnorm: bool = False, same_lengths:bool = False, rel_att=True,
                 transformer_type=Transformer_Block):
        super(Transformer_Base, self).__init__()
        # self.word_embedding = Adaptive_Embedding(vocab_size,word_embedding_dim,hidden_dim,cutoffs,div_val)
        self.n_layers = n_layers
        self.same_lengths = same_lengths
        self.seq_len = seq_len
        self.rel_att = rel_att
        self.word_embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=padding_index) # sparse=True
        self.posisition_embedding = nn.Embedding(seq_len, hidden_dim)
        # self.posisition_embedding = Position_Embedding(hidden_dim)

        if rel_att:
            self.rw_bias = nn.Parameter(torch.Tensor(n_heads, head_dim))
            self.rr_bias = nn.Parameter(torch.Tensor(n_heads, head_dim))

        self.dropout = nn.Dropout(dropout_rate)
        # if not self.embedding_equal_hidden:
        #     self.embedding_proj = nn.Linear(word_embedding_dim,hidden_dim,bias=False)
        self.main_nets = nn.ModuleList([transformer_type(hidden_dim,projection_dim,n_heads,head_dim,
                                                         dropout_rate,dropatt_rate,pre_lnorm,rel_att)
                                        for _ in range(n_layers)])

    def get_emb(self, x, mem):
        bs, qs = x.size()
        ms = mem[0].size(1) if mem is not None else 0
        ks = qs + ms
        emb = self.word_embedding(x)
        # if not self.embedding_equal_hidden:
        #     emb = self.embedding_proj(emb)
        # emb = self.dropout(emb)

        # pos_indicator = torch.arange(ks-1,-1,-1.0).to(x.device)
        pos_indicator = torch.arange(ms, ks, 1).clamp_max_(self.seq_len).to(emb.device)
        pos_ebd = self.posisition_embedding(pos_indicator)

        # relative_embedding
        if self.rel_att:
            emb = self.dropout(emb)
            pos_ebd = self.dropout(pos_ebd)

        else:
            emb = pos_ebd + emb
            emb = self.dropout(emb)

        return emb, pos_ebd

    def get_mask(self, mem, inp_masks, is_decoder):
        bs, qs = inp_masks.size()
        ms = mem[0].size(1) if mem is not None else 0
        ks = qs + ms
        ones = torch.ones((qs, ks)).byte().to(inp_masks.device)
        if is_decoder:
            dec_mask = ones.triu(1 + ms)
        else:
            dec_mask = torch.zeros_like(ones)
        if self.same_lengths:
            dec_mask = dec_mask + ones.tril(-qs)
        if ms:
            inp_masks = torch.cat([torch.zeros(bs,ms,dtype=inp_masks.dtype,device=inp_masks.device),inp_masks],1)
        mask = (inp_masks.unsqueeze(1) + dec_mask.unsqueeze(0)) > 0
        return mask


class Transformer_Model(Transformer_Base):
    def __init__(self, vocab_size:int, seq_len:int, hidden_dim: int, projection_dim: int,
                 n_heads: int, head_dim: int, n_layers:int,
                 dropout_rate: float, dropatt_rate: float, padding_index : int,
                 pre_lnorm: bool = False, same_lengths:bool = False, rel_att=True,
                 experimental_loss=False):
        super(Transformer_Model, self).__init__(vocab_size,seq_len,hidden_dim,projection_dim,n_heads,head_dim,
                                                n_layers,dropout_rate,dropatt_rate,padding_index,pre_lnorm,
                                                same_lengths,rel_att,Transformer_Block)
        self.experimental_loss = experimental_loss
        self.final = nn.Linear(hidden_dim, vocab_size, bias=False)

    def compute_hidden(self, x, mem, inp_lens, is_decoder=True):
        """
        :param x: input, input.size() = [batch_size, seq_len]
        :param mem: list of memories [mem1,mem2, ...memn], n equal to the number of layers
          memory[0].size() = [batch_size, memory_len, hidden_size]
        :return:
        """
        def input_process(out, mem_i):
            if self.rel_att:
                main_inp = (out, mem_i, mask, pos_ebd, self.rr_bias, self.rw_bias)
            elif self.hierarchical:
                if isinstance(out,tuple):
                    main_inp = out[:1] + (mem_i,) + out[2:]
                    tomem = out[1]
                else:
                    main_inp = out, mem_i, mask, mask, torch.ones_like(mask[:,-1]).unsqueeze(-1).to(out.dtype), None, out
                    tomem = out
            else:
                main_inp = out, mem_i, mask
                tomem = out
            return main_inp, tomem


        inp_masks = mask_lengths(inp_lens, reverse=True).byte()
        emb, pos_ebd = self.get_emb(x,mem)
        mask = self.get_mask(mem,inp_masks,is_decoder)
        out = emb
        new_mem = []
        for i in range(self.n_layers):
            block = self.main_nets[i]
            mem_i = mem[i] if mem is not None else None
            main_inp, mem_to_save = input_process(out, mem_i)
            new_mem.append(mem_to_save)
            out = block(main_inp)
            if isinstance(out,tuple) and out[-1] is None:
                out = main_inp[-1]
                break
        if isinstance(out, tuple):
            out = out[-1]
        return out, new_mem
        # if self.hierarchical:
        #     flag = False
        #     initial_idx = BoundaryTransformer_Block.initialize_index(emb)
        # for i in range(self.n_layers):
        #     block = self.main_nets[i]
        #     new_mem.append(out)
        #     mem_i = mem[i] if mem is not None else None
        #     main_inp = input_process(out, mem_i, block)
        #     if isinstance(block,BoundaryTransformer_Block):
        #         flag = True
        #     out = block(main_inp)
        #     if isinstance(block,BoundaryTransformer_Block):
        #         total_out, out, prev_index, mask = out
        #         if out is None:
        #             break
        # if self.hierarchical:
        #     out = total_out
        # out = self.dropout(out)
        # return out, new_mem

    def sampling(self, inp):
        """
            sampling when the model is trained with experimental loss
        """
        x, inp_lens, mem, sampling_mode, top_w, temperature = inp
        bs, qs = x.size()
        out, mem = self.compute_hidden(x,mem,inp_lens)
        out = out[:, :-1]
        out = out.contiguous().view(bs * (qs - 1), -1)
        if sampling_mode:
            ishard = True if sampling_mode ==1 else False
            out = self.final.hard_cluster_logit(out, top_w, ishard, temperature)
        else:
            out = self.final.soft_cluster_logit(out)
        return out, mem

    def forward(self, inp):
        x, inp_lens, y, mem = inp
        bs, qs = x.size()
        out, mem = self.compute_hidden(x,mem,inp_lens)
        out = out[:,:-1]
        out = out.contiguous().view(bs*(qs-1),-1)
        if self.experimental_loss:
            y = y.contiguous().view(-1)
            final = self.final(out,y)
        else:
            final = self.final(out)
        return final, mem


if __name__ == '__main__':
    vocab_size = 300000
    cutoffs = [20000,80000,200000]
    div_val = 4
    hidden_dim = 500
    word_embedding_dim = 500
    projection_dim = 1000
    n_heads = 10
    head_dim = 50
    n_layers = 10
    dropout_rate = 0.2
    dropatt_rate = 0.1

    m = Transformer_Model(vocab_size,word_embedding_dim,hidden_dim,projection_dim,n_heads,head_dim,n_layers,cutoffs,div_val,
                          dropout_rate,dropatt_rate)
    print(m.main_nets[0].multihead_att.vec_u)
    i = Initializer('normal',0.01,0.1)
    i.initialize(m)

    print(m.main_nets[0].multihead_att.vec_u)

