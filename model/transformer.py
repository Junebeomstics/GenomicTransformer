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


class BoundaryTransformer_Block(nn.Module):
    def __init__(self,hidden_dim:int, projection_dim:int, n_heads:int, head_dim:int, n_layers:int,
                 dropout_rate:float, dropatt_rate:float, pre_lnorm:bool=False, rel_att=True, mode='full'):
        super(BoundaryTransformer_Block, self).__init__()
        self.blocks = nn.ModuleList(Transformer_Block(hidden_dim,projection_dim,n_heads,head_dim,dropout_rate,
                                                      dropatt_rate, pre_lnorm,rel_att) for _ in range(n_layers))
        self.bd = BoundaryDecision(hidden_dim, n_heads, head_dim, dropatt_rate)
        self.n_layers = n_layers
        self.mode = mode
        self.n_heads = n_heads
        self.head_dim = head_dim

    @staticmethod
    def initialize_index(x):
        bs, l = x.size()[:2]
        i0 = torch.arange(bs,device=x.device)[:,None].repeat(1,l).view(-1)
        i1 = torch.arange(l,device=x.device).repeat(bs)
        return i0,i1

    @staticmethod
    def merge_selected(total_out, selected_out, selected_idx, mask):
        """
        :param total_out: tensor of whole input size of ( batch, lens, hidden)
        :param selected_out:  tensor of layer specific input size of (layer_batch, layer_lens, hidden)
        :param mask:
        :return:
        """
        mask = mask[:,-1]
        idx = (mask==False).nonzero(as_tuple=True)
        target = selected_out[idx]
        zeros = torch.zeros_like(total_out)
        zeros[selected_idx] = target
        from_tot = (zeros[:, :, -1] == 0).nonzero(as_tuple=True)
        zeros[from_tot] = total_out[from_tot]
        return zeros

    @staticmethod
    def target_shape_selected(idx):
        batch_size = ((idx[0][:, None] == torch.arange(idx[0].max() + 1,device=idx[0].device)[None]).sum(0) > 0).sum()
        batch_idx = idx[0]
        maxn = idx[0].max() + 1
        batch_aranged = torch.arange(maxn,device=idx[0].device)
        maxlen = (batch_idx[:, None] == batch_aranged[None]).sum(0).max()
        return batch_size, maxlen

    @staticmethod
    def index_to_lengths(idx, batch_size):
        return (idx[0][:, None] == torch.arange(batch_size, device=idx[0].device)[None]).sum(0)

    @staticmethod
    def select_from_index(x, idx):
        batch_size, hidden_dim = x.size(0) ,x.size(-1)
        selected = x[idx]
        batch_idx = idx[0]
        # reshape to 3d tensor
        new_bs, maxlen = BoundaryTransformer_Block.target_shape_selected(idx)
        l = 0
        res = []
        for i in range(batch_size):
            n_selected = (batch_idx == i).sum()
            if not n_selected:
                continue
            r = l + n_selected
            batch_selected = selected[l:r]
            short = maxlen - (r - l)
            padded = torch.cat([batch_selected, torch.zeros((short, hidden_dim),dtype=x.dtype,device=x.device)], 0)
            res.append(padded)
            l = r
        return selected, torch.stack(res,0)

    @staticmethod
    def mask_reshape(mask):
        bs, qs = mask.size()
        ones = torch.ones((qs, qs)).byte().to(mask.device)
        dec_mask = ones.triu(1)
        mask = (mask.unsqueeze(1) + dec_mask.unsqueeze(0)) > 0
        return mask

    @staticmethod
    def remask(mask, idx):
        bs = mask.size(0)
        ori_lengths = mask_to_lengths(mask)
        idx_lengths = BoundaryTransformer_Block.index_to_lengths(idx, bs)
        new_len = torch.min(ori_lengths, idx_lengths)
        new_len = new_len[new_len.nonzero().squeeze(1)]
        new_mask = mask_lengths(new_len,reverse=True).byte()
        mask = BoundaryTransformer_Block.mask_reshape(new_mask)
        # print(ori_lengths, idx, new_len, new_mask, mask)
        return mask

    @staticmethod
    def update_mask(prev_mask, new_mask):
        return prev_mask
        # return prev_mask + (new_mask.unsqueeze(1) < 0.5)
        # return prev_mask + (new_mask.unsqueeze(1).bool() == False)

    @staticmethod
    def reindex(prev_idx, new_idx):
        batch_size = prev_idx[0].max() + 1
        nz = ((prev_idx[0][:, None] == torch.arange(batch_size,device=prev_idx[0].device)[None]).sum(0) != 0).nonzero().squeeze(1)
        n_per_batch = (prev_idx[0][:, None] < torch.arange(batch_size,device=prev_idx[0].device)[None]).sum(0)[nz]
        return prev_idx[0][n_per_batch[new_idx[0]]], prev_idx[1][n_per_batch[new_idx[0]] + new_idx[1]]

    def headwise_mul(self, inp, out, mask):
        bs, l = inp.size()[:2]
        inp = inp.view(bs,l,self.n_heads,self.head_dim)
        out = out.view(bs,l,self.n_heads,self.head_dim)
        new_x = inp * mask + (1-mask) * out
        return new_x.view(bs,l,-1)

    def forward(self, inp, *args):
        layer_x, mem, full_mask, layer_mask, mask_multiplier, prev_idx, total_x = inp
        if prev_idx is None:
            prev_idx = self.initialize_index(layer_x)
        # print(layer_x.size(), mask.size(), mask[:,-1])
        new_mem = []
        out = layer_x
        for i in range(self.n_layers):
            block = self.blocks[i]
            new_mem.append(out)
            mem_i = mem[i] if mem is not None else None
            out = block((out, mem_i, full_mask), *args)

        select_idx, new_layer_mask = self.bd(out, mem, full_mask)
        if self.mode == 'full':
            # a,b = new_layer_mask[...,0].unsqueeze(-1), new_layer_mask[...,1].unsqueeze(-1)

            # mask_multiplier = new_layer_mask.to(out.dtype)
            # mask_multiplier = new_mask[:,-1].unsqueeze(-1).to(out.dtype)
            # mask_multiplier = layer_mask[:,-1].unsqueeze(-1).to(out.dtype)
            # new_out = out * (1 - mask_multiplier) + mask_multiplier * layer_x
            new_out = out * mask_multiplier + (1 - mask_multiplier) * layer_x
            # new_out = self.headwise_mul(layer_x, out, mask_multiplier)
            # mask_multiplier = mask_multiplier.to(out.dtype)
            # new_out = out * a + b * layer_x
            # new_mask = new_layer_mask
            mask_multiplier = new_layer_mask.to(out.dtype)
            new_mask = self.update_mask(full_mask, new_layer_mask)
            return new_out, new_mem, full_mask, new_mask, mask_multiplier, None, new_out
        elif self.mode =='select':
            if select_idx[0].size(0) > 0:
                selected_2d, selected_3d = self.select_from_index(out, select_idx)
                new_idx = self.reindex(prev_idx, select_idx)
                new_mask = self.remask(layer_mask,select_idx)
                total_x = self.merge_selected(total_x, out, prev_idx, layer_mask)
            else:
                total_x, selected_3d, new_idx, new_mask = None, None, None, None
            return selected_3d, new_mem, new_mask, new_idx, total_x




        # out = self.multihead_att(layer_x, mem, mask,*args)
        # layer_out = self.feedforward(out)
        # select_idx, layer_mask = self.bd(layer_out, mem, mask)
        # if select_idx[0].size(0)>0:
        #     selected_2d, selected_3d = self.select_from_index(layer_out, select_idx)
        #     idx_for_total = self.reindex(prev_idx, select_idx)
        #     new_mask = self.remask(mask,select_idx)
        #     total_x = self.merge_selected(total_x, selected_2d, selected_idx)
        # else:
        #     selected_3d, idx_for_total, new_mask = None, None, None
        # return total_x, selected_3d, idx_for_total, new_mask


class Conditional_Transformer_Block(nn.Module):
    def __init__(self,hidden_dim:int, projection_dim:int, n_heads:int, head_dim:int,
                 dropout_rate:float,dropatt_rate:float,pre_lnorm:bool=False,rel_att=True):
        super(Conditional_Transformer_Block, self).__init__()
        self.multihead_att = Rel_Multihead_Att(hidden_dim,n_heads,head_dim,dropout_rate,dropatt_rate,pre_lnorm) \
            if rel_att else Multihead_Att(hidden_dim,n_heads,head_dim,dropout_rate,dropatt_rate,pre_lnorm)
        self.feedforward = Residual_FF(hidden_dim,projection_dim,dropout_rate,pre_lnorm)
        self.gate_linear = Gated_FF(hidden_dim,dropout_rate)

    def forward(self, title, content, mem, title_mask, content_mask, *args):
        bs = title.size(0)
        temp = title_mask[:,0] == False
        tl = torch.sum(temp,1)
        c_out = self.multihead_att(content, mem, content_mask, *args)
        c_out = self.feedforward(c_out)
        t_out = self.multihead_att(title, None, title_mask, *args)
        t_out = self.feedforward(t_out)

        title_mask = title_mask[:,0].unsqueeze(1)
        c_out = self.multihead_att(c_out, t_out, title_mask, True, *args)
        # c_out = self.gate_linear(c_attented,c_out)
        return t_out, c_out


class ConditionalBlock(nn.Module):
    def __init__(self,hidden_dim:int, projection_dim:int, n_heads:int, head_dim:int,
                 dropout_rate:float,dropatt_rate:float,pre_lnorm:bool=False, *args):
        super(ConditionalBlock, self).__init__()
        self.self_attn = Multihead_Att(hidden_dim,n_heads,head_dim,dropout_rate,dropatt_rate,pre_lnorm)
        self.enc_attn = Multihead_Att(hidden_dim,n_heads,head_dim,dropout_rate,dropatt_rate,pre_lnorm)
        self.feedforward = Residual_FF(hidden_dim,projection_dim,dropout_rate,pre_lnorm)

    def forward(self, enc_out, dec_in, dec_mem, mask):
        dec_out = self.self_attn(dec_in, dec_mem, mask)
        dec_out = self.enc_attn(dec_out, enc_out, mask, True)
        dec_out = self.feedforward(dec_out)
        return dec_out


class Transformer_Base(nn.Module):
    def __init__(self, vocab_size:int, seq_len:int, hidden_dim: int, projection_dim: int,
                 n_heads: int, head_dim: int, n_layers:int,
                 dropout_rate: float, dropatt_rate: float, padding_index : int,
                 pre_lnorm: bool = False, same_lengths:bool = False, rel_att=True,
                 transformer_type=Transformer_Block, hierarchical=False):
        super(Transformer_Base, self).__init__()
        # self.word_embedding = Adaptive_Embedding(vocab_size,word_embedding_dim,hidden_dim,cutoffs,div_val)
        self.n_layers = n_layers
        self.same_lengths = same_lengths
        self.seq_len = seq_len
        self.rel_att = rel_att
        self.word_embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=padding_index) # sparse=True
        self.posisition_embedding = nn.Embedding(seq_len, hidden_dim)
        self.hierarchical = hierarchical
        # self.posisition_embedding = Position_Embedding(hidden_dim)

        if rel_att:
            self.rw_bias = nn.Parameter(torch.Tensor(n_heads, head_dim))
            self.rr_bias = nn.Parameter(torch.Tensor(n_heads, head_dim))

        self.dropout = nn.Dropout(dropout_rate)
        # if not self.embedding_equal_hidden:
        #     self.embedding_proj = nn.Linear(word_embedding_dim,hidden_dim,bias=False)
        if hierarchical:
            block_depths = 4
            self.n_layers = n_layers // block_depths
            self.main_nets = nn.ModuleList([BoundaryTransformer_Block(hidden_dim, projection_dim, n_heads, head_dim,
                                                                      block_depths, dropout_rate, dropatt_rate, pre_lnorm, rel_att)
                                            for _ in range(self.n_layers)])
        else:
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


class ConditionalDecoder(Transformer_Base):
    def __init__(self, vocab_size:int, seq_len:int, hidden_dim: int, projection_dim: int,
                 n_heads: int, head_dim: int, n_layers:int,
                 dropout_rate: float, dropatt_rate: float, padding_index : int,
                 pre_lnorm: bool = False, same_lengths:bool = False, rel_att=True):
        super(ConditionalDecoder, self).__init__(vocab_size, seq_len, hidden_dim, projection_dim, n_heads, head_dim,
                                                n_layers, dropout_rate, dropatt_rate, padding_index, pre_lnorm,
                                                same_lengths, rel_att, ConditionalBlock)

    def forward(self, enc_out, dec_inp, dec_len, dec_mem):
        inp_masks = mask_lengths(dec_len,reverse=True).byte()
        emb, pos_ebd = self.get_emb(dec_inp, dec_mem)
        # print(dec_inp.size(), inp_masks.size())
        mask = self.get_mask(dec_mem,inp_masks,True)
        out = emb
        new_mem = []
        for i in range(self.n_layers):
            new_mem.append(out)
            mem_i = dec_mem[i] if dec_mem is not None else None
            out = self.main_nets[i](enc_out, out, mem_i, mask)
        out = self.dropout(out)
        return out, new_mem


class Transformer_Model(Transformer_Base):
    def __init__(self, vocab_size:int, seq_len:int, hidden_dim: int, projection_dim: int,
                 n_heads: int, head_dim: int, n_layers:int, cutoffs:list,
                 dropout_rate: float, dropatt_rate: float, padding_index : int,
                 pre_lnorm: bool = False, same_lengths:bool = False, rel_att=True,
                 experimental_loss=False, hierarchical=False):
        super(Transformer_Model, self).__init__(vocab_size,seq_len,hidden_dim,projection_dim,n_heads,head_dim,
                                                n_layers,dropout_rate,dropatt_rate,padding_index,pre_lnorm,
                                                same_lengths,rel_att,Transformer_Block,hierarchical)
        self.experimental_loss = experimental_loss
        if experimental_loss == 1:
            self.final = Factorized_SoftmaxV2(vocab_size, hidden_dim, cutoffs, padding_index)
        elif experimental_loss == 2:
            self.final = Factorized_Softmax(vocab_size, hidden_dim, cutoffs, padding_index)
        else:
            self.final = nn.Linear(hidden_dim, vocab_size, bias=False)
        # self.final = Adaptive_Softmax(vocab_size,hidden_dim,cutoffs,div_val)

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


class Conditional_Transformer(Transformer_Base):
    def __init__(self, vocab_size:int, seq_len:int, hidden_dim: int, projection_dim: int,
                 n_heads: int, head_dim: int, n_layers:int, cutoffs:list,
                 dropout_rate: float, dropatt_rate: float, padding_index : int,
                 pre_lnorm: bool = False, same_lengths:bool = False, rel_att=True, experimental_loss=False):
        super(Conditional_Transformer, self).__init__(vocab_size,seq_len,hidden_dim,projection_dim,n_heads,head_dim,
                                                      n_layers, dropout_rate,dropatt_rate,padding_index,pre_lnorm,
                                                      same_lengths, rel_att,Conditional_Transformer_Block)

        self.ln = nn.LayerNorm(hidden_dim)
        self.experimental_loss = experimental_loss
        if experimental_loss:
            self.final = Factorized_Softmax(vocab_size,hidden_dim,cutoffs,padding_index)
        else:
            self.final = nn.Linear(hidden_dim,vocab_size)

    def compute_hidden(self, inp):
        title, content, title_lens, content_lens, mem = inp
        content_masks = mask_lengths(content_lens, reverse=True).byte()
        title_masks = mask_lengths(title_lens, reverse=True).byte()

        t_emb, _ = self.get_emb(title, None)
        c_emb, _ = self.get_emb(content, mem)

        t_mask = self.get_mask(title, None, title_masks, False)
        c_mask = self.get_mask(content, mem, content_masks, True)

        new_mem = []
        for i in range(self.n_layers):
            new_mem.append(c_emb)
            # c_emb = torch.sum(torch.stack(new_mem,0),0)
            mem_i = mem[i] if mem is not None else None
            main_inp = (t_emb, c_emb, mem_i, t_mask, c_mask, _, self.rr_bias, self.rw_bias) if self.rel_att else \
                (t_emb, c_emb, mem_i, t_mask, c_mask)
            t_emb, c_emb = self.main_nets[i](*main_inp)
            # c_emb = self.ln(c_emb)
        out = self.dropout(c_emb)
        return out, new_mem

    def sampling(self, inp):
        """
            sampling when the model is trained with experimental loss
        """
        title, content, title_lens, content_lens, mem, hard_cluster, ishard = inp
        out, mem = self.compute_hidden((title, content, title_lens, content_lens, mem))

        bs, qs = content.size()
        out = out[:, :-1]
        out = out.contiguous().view(bs * (qs - 1), -1)
        if hard_cluster:
            out = self.final.hard_cluster_sampling(out, ishard)
        else:
            out = self.final.soft_cluster_sampling(out)
        return out, mem

    def forward(self, inp):
        title, content, title_lens, content_lens, y, mem = inp

        bs, qs = content.size()
        gated, mem = self.compute_hidden((title,content,title_lens,content_lens,mem))

        gated = gated[:, :-1]
        gated = gated.contiguous().view(bs * (qs - 1), -1)

        if self.experimental_loss:
            y = y.contiguous().view(-1)
            final = self.final(gated, y)
        else:
            final = self.final(gated)
        return final, mem


class KoGPTEncoder(nn.Module):
    def __init__(self, mode='single', index_converted=False, index_dic=None, tie_wieght=True, **kwargs):
        from kogpt2.pytorch_kogpt2 import get_pytorch_kogpt2_model
        super(KoGPTEncoder, self).__init__()
        assert mode in ['pair', 'single']
        if index_converted: assert index_dic is not None
        self.gpt, _ = get_pytorch_kogpt2_model(tie_weight=tie_wieght, **kwargs)
        self.model = self.gpt.transformer
        self.outdim = self.get_dim()
        self.vocab_size = self.get_vocab_size()
        self.mode = mode
        if index_converted:
            reindex_embedding(self.model.wte, index_dic)

    def get_dim(self):
        return self.gpt.transformer.wte.embedding_dim

    def get_vocab_size(self):
        return self.gpt.transformer.wte.num_embeddings

    def forward(self, inp):
        inp_ids, inp_lens, mem = inp
        if self.mode == 'pair':
            assert isinstance(inp_ids, tuple)
            out1, _ = self.model(inp_ids[0],attention_mask=None)
            out2, _ = self.model(inp_ids[1],attention_mask=None)
            out = out1 + out2
        else:
            out, mem = self.model(inp_ids,past=mem, attention_mask=None)
        return out, mem


class KoGPTLM(nn.Module):
    def __init__(self, experimental=True, index_converted=False, index_dic=None, tie_weight=True, **kwargs):
        super(KoGPTLM, self).__init__()
        self.enc = KoGPTEncoder(mode='single', index_converted=index_converted,
                                index_dic=index_dic, tie_wieght=tie_weight, **kwargs)
        self.experimental_loss = experimental
        if not tie_weight:
            reindex_embedding(self.enc.gpt.lm_head, index_dic)
        if experimental:
            assert index_converted and index_dic is not None
            self.lm = Factorized_SoftmaxV2(self.enc.vocab_size, self.enc.outdim,
                                           pretrained=self.enc.gpt.lm_head.weight.T, **kwargs)
        else:
            self.lm = self.enc.gpt.lm_head

    def compute_hidden(self, inp):
        x, mem, inp_lens = inp

        out, new_mem = self.enc((x, inp_lens, mem))
        return out, new_mem

    def sampling(self, inp):
        x, inp_lens, mem, sampling_mode, top_w, temperature = inp
        enc_out, new_mem = self.compute_hidden((x, mem, inp_lens))
        bs, qs = x.size()
        enc_out = enc_out[:,:-1]
        out = enc_out.contiguous().view(bs * (qs - 1), -1)
        if sampling_mode:
            ishard = True if sampling_mode ==1 else False
            out = self.lm.hard_cluster_logit(out, top_w, ishard, temperature)
        else:
            out = self.lm.soft_cluster_logit(out)
        return out, new_mem

    def forward(self, inp):
        x, inp_lens, y, mem = inp
        out, mem = self.enc((x, inp_lens, mem))

        bs, qs = x.size()
        out = out[:,:-1]
        out = out.contiguous().view(bs*(qs-1),-1)
        if self.experimental_loss:
            y = y.contiguous().view(-1)
            final = self.lm(out,y)
        else:
            final = self.lm(out)

        return final, mem


class SKTGPT(nn.Module):
    def __init__(self, encoder, n_class, mode='single',):
        from kogpt2.pytorch_kogpt2 import get_pytorch_kogpt2_model
        super(SKTGPT, self).__init__()
        assert mode in ['pair', 'single']
        self.gpt, _ = get_pytorch_kogpt2_model()
        self.encoder = encoder
        self.model = self.gpt.transformer
        self.outdim = self.get_dim()
        self.output_layer = nn.Linear(self.outdim, n_class)
        torch.nn.init.normal_(self.output_layer.weight, std=0.02)
        torch.nn.init.zeros_(self.output_layer.bias)
        self.mode = mode

    def get_dim(self):
        return self.gpt.transformer.wpe.embedding_dim

    def forward(self, inp):
        inp_ids, inp_lens = inp
        inp_masks = mask_lengths(inp_lens)
        if self.mode == 'pair':
            assert isinstance(inp_ids, tuple)
            out1, _ = self.model(inp_ids[0],attention_mask=None)
            out2, _ = self.model(inp_ids[1],attention_mask=None)
            out = out1 + out2
        else:
            out, _ = self.model(inp_ids,attention_mask=None)
        return self.output_layer(out[torch.arange(inp_lens.size(0)), inp_lens-1])


class SATransformer(nn.Module):
    def __init__(self, experimental, tie_weight,
                 st_size:int, seq_len: int, hidden_dim: int, projection_dim: int,
                 n_heads: int, head_dim: int, n_layers: int,dropout_rate: float, dropatt_rate: float,
                 token_padding_index: int, st_padding_index:int, proj_mode:str,  **kwargs):
        super(SATransformer, self).__init__()
        self.experimental_loss = experimental
        self.enc = KoGPTEncoder(mode='single', tie_wieght=tie_weight, **kwargs)
        self.dec = ConditionalDecoder(st_size, seq_len, hidden_dim, projection_dim, n_heads, head_dim, n_layers,
                                      dropout_rate, dropatt_rate, st_padding_index, rel_att=False)
        self.proj = ConditionalProjection(st_size, hidden_dim, dropout_rate, st_padding_index, proj_mode)
        if experimental:
            self.lm = Factorized_SoftmaxV2(self.enc.vocab_size, self.enc.outdim, padding_index=token_padding_index,
                                           **kwargs)
        else:
            self.lm = self.enc.gpt.lm_head
        self.st_out = nn.Linear(hidden_dim,st_size)

    def compute_hidden(self, inp_txt, inp_st, inp_lens, txt_mem, st_mem):
        enc_out, new_txt_mem = self.enc((inp_txt, inp_lens, txt_mem))
        dec_out, new_dec_mem = self.dec(enc_out, inp_st, inp_lens, st_mem)
        return enc_out, dec_out, new_txt_mem, new_dec_mem

    def sampling(self, inp):
        inp_txt, inp_st, inp_lens, txt_mem, st_mem, sampling_mode, top_w, temperature = inp
        enc_out, dec_out, new_txt_mem, new_dec_mem = self.compute_hidden(inp_txt, inp_st, inp_lens, txt_mem, st_mem)
        bs, qs = inp_txt.size()
        dec_out = dec_out[:, :-1]
        enc_out = enc_out[:,:-1]
        st_final = self.st_out(dec_out)
        # target_st = st_final.argmax(dim=-1)
        # target_st = target_st.contiguous().view(bs * (qs - 1))
        # st_final = st_final.contiguous().view(bs * (qs - 1),-1)
        # st_final = torch.softmax(st_final.to(torch.float32), -1)
        # target_st = st_final.multinomial(1).squeeze(-1)
        # enc_out = enc_out.contiguous().view(bs * (qs - 1), -1)
        # print(target_st, target_st.size(), enc_out.size())
        enc_projected = self.proj(enc_out, dec_out).contiguous().view(bs * (qs - 1), -1)
        if sampling_mode:
            ishard = True if sampling_mode == 1 else False
            out = self.lm.hard_cluster_logit(enc_projected, top_w, ishard, temperature)
        else:
            out = self.lm.soft_cluster_logit(enc_projected)
        return out, st_final.contiguous().view(bs * (qs - 1), -1),\
               new_txt_mem, new_dec_mem

    def forward(self, inp):
        inp_txt, inp_st, inp_lens, y, txt_mem, st_mem = inp
        enc_out, dec_out, new_txt_mem, new_dec_mem = self.compute_hidden(inp_txt, inp_st, inp_lens, txt_mem, st_mem)
        # target_st = inp_st[:,1:]
        # enc_out = enc_out[:,:-1]
        # enc_projected = self.proj(enc_out, target_st)
        enc_projected = self.proj(enc_out, dec_out)
        enc_projected = enc_projected[:,:-1]

        bs, qs = inp_txt.size()
        dec_out = dec_out[:,:-1]
        enc_projected = enc_projected.contiguous().view(bs * (qs - 1), -1)
        if self.experimental_loss:
            y = y.contiguous().view(-1)
            txt_final = self.lm(enc_projected, y)
        else:
            txt_final = self.lm(enc_projected)

        st_final = self.st_out(dec_out)
        return (txt_final, st_final), (new_txt_mem, new_dec_mem)

# class Conditional_Transformer(Transformer_Base):
#     def __init__(self, vocab_size:int, seq_len:int, hidden_dim: int, projection_dim: int,
#                  n_heads: int, head_dim: int, n_layers:int, cutoffs:list,
#                  dropout_rate: float, dropatt_rate: float, padding_index : int,
#                  pre_lnorm: bool = False, same_lengths:bool = False, rel_att=True, experimental_loss=False):
#         super(Conditional_Transformer, self).__init__(vocab_size,seq_len,hidden_dim,projection_dim,n_heads,head_dim,
#                                                       n_layers, dropout_rate,dropatt_rate,padding_index,pre_lnorm,
#                                                       same_lengths, rel_att,Conditional_Transformer_Block)
#         self.experimental_loss = experimental_loss
#         if experimental_loss:
#             self.final = Factorized_Softmax(vocab_size,hidden_dim,cutoffs,padding_index)
#         else:
#             self.final = nn.Linear(hidden_dim,vocab_size)
#
#     def compute_out(self, title, content, title_lens, content_lens, mem):
#         bs, qs = content.size()
#         content_masks = mask_lengths(content_lens, reverse=True).byte()
#         title_masks = mask_lengths(title_lens, reverse=True).byte()
#         content_out, _ = self.compute_hidden(content, mem, content_masks)
#
#         title_out, _ = self.compute_hidden(title, None, title_masks, is_decoder=False)
#         title_cls = title_out[torch.arange(bs), title_lens - 1]
#
#         gated = self.gate_linear(title_cls,content_out)
#         return gated
#
#     def sampling(self, inp):
#         """
#             sampling when the model is trained with experimental loss
#         """
#         title, content, title_lens, content_lens, mem, hard_cluster = inp
#         out = self.compute_out(title,content, title_lens, content_lens, mem)
#
#         bs, qs = content.size()
#         out = out[:, :-1]
#         out = out.contiguous().view(bs * (qs - 1), -1)
#         if hard_cluster:
#             out = self.final.hard_cluster_sampling(out)
#         else:
#             out = self.final.soft_cluster_sampling(out)
#         return out
#
#     def forward(self, inp):
#         title, content, title_lens, content_lens, y, mem = inp
#
#         bs, qs = content.size()
#         gated = self.compute_out(title,content,title_lens,content_lens,mem)
#
#         gated = gated[:, :-1]
#         gated = gated.contiguous().view(bs * (qs - 1), -1)
#
#         if self.experimental_loss:
#             y = y.contiguous().view(-1)
#             final = self.final(gated, y)
#         else:
#             final = self.final(gated)
#         return final


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

