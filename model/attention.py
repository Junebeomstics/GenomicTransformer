import torch
from torch import nn


class AttBase(nn.Module):
    def __init__(self, hidden_dim:int, n_head:int, head_dim:int,
                 dropout_rate:float, dropatt_rate:float=0.0, pre_lnorm=False):
        super(AttBase, self).__init__()
        self.n_head = n_head
        self.hidden_dim = hidden_dim
        self.head_dim = head_dim
        self.dropout_rate = dropout_rate

        self.kv_net = nn.Linear(hidden_dim, 2 * n_head * head_dim, bias=False)
        self.q_net = nn.Linear(hidden_dim, n_head * head_dim, bias=False)

        self.dropout = nn.Dropout(dropout_rate)
        self.dropatt = nn.Dropout(dropatt_rate)
        self.o_net = nn.Linear(n_head * head_dim, hidden_dim, bias=False)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        self.scale = 1 / (head_dim ** 0.5)

        self.pre_lnorm = pre_lnorm


class Multihead_Att(AttBase):
    def __init__(self, hidden_dim:int, n_head:int, head_dim:int,
                 dropout_rate:float, dropatt_rate:float=0.0, pre_lnorm=False):
        super(Multihead_Att, self).__init__(hidden_dim, n_head, head_dim, dropout_rate, dropatt_rate, pre_lnorm)

    def attend(self, query, key, value, mask):
        bs, qs, hs = query.size()
        ks = key.size(1)
        ms = ks-qs

        # print(query.size(),key.size(),value.size(),rel.size())
        #reshaping
        k = key.view(bs,ks,self.n_head,self.head_dim)
        v = value.view(bs,ks,self.n_head,self.head_dim)
        q = query.view(bs,qs,self.n_head,self.head_dim)

        att_score = torch.einsum('bqnd,bknd->bqkn',q,k)
        att_score.mul_(self.scale)

        #attend
        if mask is None:
            print('mask is none')
            mask = torch.ones((qs,ks)).byte()
            mask = mask.triu(1+ms) ==0
        # print(mask.size())
        encoder_mask = mask.bool()
        att_score.masked_fill_(encoder_mask.unsqueeze(-1), -6e4)
        # print(att_score)
        att_prob = torch.softmax(att_score,2)
        att_prob = self.dropatt(att_prob)

        attended = torch.einsum('bqkn,bknd->bqnd',att_prob,v)
        out = self.o_net(attended.contiguous().view(bs,qs,-1))
        out = self.dropout(out)
        return out

    def forward(self, x, mem, mask, ed_att=False):
        """
        :param x: input, input.size() = [batch_size, input_len, hidden_dim]
        :param mem:  memory, input.size() = [batch_size, memory_len, hidden_dim]
        :param decoder_mask: position_embedding, pos_ebd.size() = [input_len + memory_len, hidden_dim]
        :param ed_att: only attend to mem
        :param encoder_mask:
        :return:
        """
        if mem is None:
            mem = torch.Tensor().to(x.device).to(x.dtype)
        if ed_att:
            c = mem
        else:
            c = torch.cat([mem,x],1)

        if self.pre_lnorm:
            c = self.layer_norm(c)
            x = self.layer_norm(x)

        #projection
        kv = self.kv_net(c)
        key, value = kv.chunk(2,-1)
        query = self.q_net(x)
        out = self.attend(query,key,value,mask)

        # if ed_att:
        #     return out
        out = x + out
        if not self.pre_lnorm:
            out = self.layer_norm(out)
        return out


class Rel_Multihead_Att(AttBase):
    def __init__(self, hidden_dim:int, n_head:int, head_dim:int,
                 dropout_rate:float, dropatt_rate:float=0.0, pre_lnorm=False):
        super(Rel_Multihead_Att, self).__init__(hidden_dim, n_head, head_dim,
                 dropout_rate, dropatt_rate, pre_lnorm)

        self.r_net = nn.Linear(self.hidden_dim, self.n_head * self.head_dim, bias=False)


    def _left_shift(self, x:torch.Tensor)->torch.Tensor:
        """
        :param x: x.size() = [batch_size, q_len, k_len, n_head]
        x[0,:,:,0] =
        [[[9,8,7,6,5,4,3,2,1,0],
          [9,8,7,6,5,4,3,2,1,0],
          [9,8,7,6,5,4,3,2,1,0]]]]

        :param zero_triu:
        :return: left_shifted tensor of x by the index along query axis
        x[0,:,:,0] =
        [[[7,6,5,4,3,2,1,0,0,0], -> left shifted by 2
          [8,7,6,5,4,3,2,1,0,0], -> left shifted by 1
          [9,8,7,6,5,4,3,2,1,0]]]] ->shifted 0

        """
        bs,qs,ks,hs = x.size()
        zero_pad = torch.zeros((bs, qs, 1,hs),
                               device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=2)  #[b,q,k+1,n]

        x_padded = x_padded.view(bs, ks+1, qs, hs)

        x = x_padded[:,1:].view_as(x)

        ones = torch.ones((qs, ks),device=x.device, dtype=x.dtype)
        x = x * torch.tril(ones, ks-bs)[None,:, :, None]

        return x


    def attend(self, query, key, value, rel, rr_bias, rw_bias, mask):
        bs, qs, hs = query.size()
        ks = key.size(1)
        ms = ks-qs

        # print(query.size(),key.size(),value.size(),rel.size())
        #reshaping
        k = key.view(bs, ks, self.n_head, self.head_dim)
        v = value.view(bs, ks, self.n_head, self.head_dim)
        q = query.view(bs, qs, self.n_head, self.head_dim)
        r = rel.view(qs, self.n_head, self.head_dim)

        rwq = q + rw_bias[None, None]
        AC = torch.einsum('bqnd,bknd->bqkn', rwq, k)

        rrq = q + rr_bias[None, None]
        BD = torch.einsum('bqnd,knd->bqkn', rrq, r)
        BD = self._left_shift(BD)
        #attend
        if mask is None:
            print('mask is none')
            mask = torch.ones((qs,ks)).byte()
            mask = mask.triu(1+ms) ==0
        # print(mask.size())
        mask = mask.bool()

        att_score = AC + BD
        att_score.mul_(self.scale)
        att_score.masked_fill_(mask.unsqueeze(-1), -float('inf'))
        # print(att_score)
        att_prob = torch.softmax(att_score,2)
        att_prob = self.dropatt(att_prob)

        attended = torch.einsum('bqkn,bknd->bqnd',att_prob,v)
        out = self.o_net(attended.contiguous().view(bs,qs,-1))
        out = self.dropout(out)
        return out

    def forward(self,x, mem, mask, pos_emb, rr_bias, rw_bias):
        """
        :param x: input, input.size() = [batch_size, input_len, hidden_dim]
        :param mem:  memory, input.size() = [batch_size, memory_len, hidden_dim]
        :param pos_ebd: position_embedding, pos_ebd.size() = [input_len + memory_len, hidden_dim]
        :param mask: size = [batch_size, query_len, memory_len]
        :param rr_bias : attention bias
        :param rw_bias : attention bias
        :return:
        """
        if mem is None:
            mem = torch.Tensor().to(x.device).to(x.dtype)
        c = torch.cat([mem,x],1)

        if self.pre_lnorm:
            c = self.layer_norm(c)
            x = self.layer_norm(x)

        #projection
        kv = self.kv_net(c)
        key, value = kv.chunk(2,-1)
        query = self.q_net(x)
        rel = self.r_net(pos_emb)

        out = self.attend(query, key, value, rel,rr_bias, rw_bias, mask)
        out = x + out
        if not self.pre_lnorm:
            out = self.layer_norm(out)
        return out
