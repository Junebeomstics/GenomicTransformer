import torch
import torch.nn as nn


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def log_sum_exp(vec, m_size):
    """
    calculate log of exp sum
    args:
        vec (batch_size, vanishing_dim, hidden_dim) : input tensor
        m_size : hidden_dim
    return:
        batch_size, hidden_dim
    """
    _, idx = torch.max(vec, 1)  # B * 1 * M
    max_score = torch.gather(vec, 1, idx.view(-1, 1, m_size)).view(-1, 1, m_size)  # B * M
    return max_score.view(-1, m_size) + torch.log(torch.sum(torch.exp(vec - max_score.expand_as(vec)), 1)).view(-1, m_size)  # B * M



def mask_lengths(lengths:torch.LongTensor, max_len:torch.long=None,reverse=False)->torch.Tensor:
    """

    :param lengths: [batch_size] indicates lengths of sequence
    :return: [batch_size, max_len] ones for within the lengths zeros for exceeding lengths

    [4,2] -> [[1,1,1,1]
              ,[1,1,0,0]]
    """
    device = lengths.device
    if not max_len:
        max_len = torch.max(lengths).item()
    idxes = torch.arange(0,max_len,out=torch.LongTensor(max_len)).unsqueeze(0).to(device)
    masks = (idxes<lengths.unsqueeze(1)).byte()
    if reverse:
        masks = masks ==0
    return masks

def last_pool(x,seq_lengths):
    device = x.device
    row_indices = torch.arange(0, x.size(0)).long().to(device)
    col_indices = seq_lengths - 1

    last_tensor = x[row_indices, col_indices, :]
    return last_tensor

def reorder_sequence(x,index):
    x2 = torch.empty_like(x)
    x2[index,:,:] = x
    return x2

def run_rnn(x,lengths,rnn):
    sorted_lengths, sort_index = lengths.sort(0, True)
    x_sorted = x.index_select(0, sort_index)
    packed_input = nn.utils.rnn.pack_padded_sequence(x_sorted, sorted_lengths, batch_first=True)
    packed_output, _ = rnn(packed_input, None)
    out_rnn, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
    out_rnn = reorder_sequence(out_rnn, sort_index)
    return out_rnn


def adjust_embedding_indices(model, old_dic, new_dic):
    we_sd = model.word_embedding.state_dict()
    for i in we_sd.keys():
        we_sd[i] = get_adjusted_tensor(we_sd[i], old_dic, new_dic)
    model.word_embedding.load_state_dict(we_sd)
    out_sd = model.final.state_dict()
    for i in out_sd.keys():
        if 'cluster' not in i:
            out_sd[i] = get_adjusted_tensor(out_sd[i],old_dic,new_dic)
    model.final.load_state_dict(out_sd)


def get_adjusted_tensor(tensor, old_dic, new_dic):
    old_inv = dict(zip(old_dic.values(), old_dic.keys()))
    old_indices = []
    new_indices = []
    for i in old_inv.keys():
        old_indices.append(i)
        new_indices.append(new_dic[old_inv[i]])
    zeros = torch.zeros_like(tensor)
    zeros[new_indices] = tensor[old_indices]
    return zeros
