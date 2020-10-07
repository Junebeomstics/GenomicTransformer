from model.ops import mask_lengths
import re
import torch
import numpy as np

from copy import deepcopy

def top_k_logits(logits, k):
    if k == 0:
        # no truncation
        return logits
    else:
        values, _ = torch.topk(logits, k=k)
        min_values = values[:, -1, None]
        return torch.where(
            logits < min_values,
            torch.ones_like(logits, dtype=logits.dtype) * -1e4,
            logits,
        )


def top_p_logits(logits, p):
    """Nucleus sampling"""
    batch = logits.size(0)
    sorted_logits, _ = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    a = torch.arange(0,batch).to(logits.device)
    b = torch.max(torch.sum(cumulative_probs <= p, dim=-1) - 1, torch.Tensor([0]).long().to(logits.device))
    min_values = sorted_logits[a,b].to(logits.device)
    return torch.where(
        logits < min_values[:,None],
        torch.ones_like(logits) * -1e4,
        logits,
    )


def gathered_input(indexed):
    device = indexed.device
    # print(indexed.size())
    bs, l = indexed.size()
    lens = torch.LongTensor([l + 1] * bs).to(device)
    indexed = torch.cat([indexed, torch.LongTensor([0] * bs)[:, None].to(device)], 1)
    return bs, l, (indexed,lens)


def divided_input(indexed):
    device = indexed[0].device
    title, content, title_len, context_len = indexed
    bs, tl = title.size()
    content = content[:,-1:]
    _, cl = content.size()
    cls = torch.LongTensor([2] * bs).to(device)
    cind = torch.cat([content, torch.LongTensor([0] * bs)[:, None].to(device)], 1)
    return bs, cl, (title, cind, title_len, cls)


def structured_input(indexed):
    device = indexed[0].device
    txt, struct = indexed
    bs, tl = txt.size()
    l = torch.LongTensor([tl + 1] * bs).to(device)
    txt = torch.cat([txt, torch.LongTensor([0] * bs)[:, None].to(device)], 1)
    struct = torch.cat([struct, torch.LongTensor([0] * bs)[:, None].to(device)], 1)
    return bs, tl, (txt, struct, l)


def get_mem(model,inp):
    istuple = True if isinstance(inp, tuple) else False
    with torch.no_grad():
        if istuple:
            title, context, title_len, context_len = inp
            context = context[:,:-1]
            context_len = torch.clamp_min(context_len - 1,0)
            _, mem = model.compute_hidden((title,context,title_len,context_len,None))
        else:
            bs, l = inp.size()
            lens = torch.LongTensor([l - 1] * bs).to(inp.device)
            _, mem = model.compute_hidden((inp[:,:-1],None,lens))

    return mem, inp


def sample(model, lengths, inp, top_w, temparature, experimental_loss, sampling_mode=0):
    top_whatever = top_k_logits if isinstance(top_w, int) else top_p_logits
    probs = None
    istuple = True if isinstance(inp, tuple) else False
    mem, inp = get_mem(model, inp)
    res = torch.LongTensor([]).to(inp.device)
    cnt = 0
    for _ in range(lengths):
        cnt+=1
        with torch.no_grad():
            if istuple:
                bs, l, inp = divided_input(inp)
            else:
                bs, l, inp = gathered_input(inp[:,-1:])
            if experimental_loss:
                logits, new_mem = model.sampling(inp + (mem, sampling_mode, top_w, temparature))
            else:
                logits, new_mem = model(inp + (None, mem))
            # mem = [torch.cat([mem[i], new_mem[i].to(mem[i].dtype)[:,:-1]],1) for i in range(len(mem))]
            mem = tuple([new_mem[i][...,:-1,:] for i in range(len(mem))])
            logits = top_whatever(logits, top_w)
            logits = logits.view(bs,l,-1)
            logits = logits[:,-1,:] / temparature
            saved_logits = logits
            sampled = torch.multinomial(torch.softmax(logits,-1),1)
            res = torch.cat([res,sampled],1)
            temp_probs = torch.softmax(saved_logits, 1)
            probs = torch.cat([probs,temp_probs[torch.arange(len(sampled)),sampled.squeeze(1)][:,None]],1) \
                if probs is not None else temp_probs[torch.arange(len(sampled)),sampled.squeeze(1)][:,None]
            if istuple:
                title, cind, tls, cls = inp
                cind = sampled
                inp = (title, cind,tls,cls)
            else:
                inp = sampled
            # if sampled == torch.LongTensor([[0]]).to('cuda'):
            #     cnt +=1
            #     if cnt ==2:
            #         break
    if istuple:
        return res.tolist(), probs.tolist()
    else:
        return res.tolist(), probs.tolist()


def sample_char(model, lengths, inp, top_w, temparature, experimental_loss, encoder, sampling_mode=0):
    top_whatever = top_k_logits if isinstance(top_w, int) else top_p_logits

    vocab_size=encoder.vocab_size

    vocab_length=[ count_syllable(encoder.decode([i]).strip()) for i in range(vocab_size-1) ]
    vocab_length.append(0)
    vocab_length=torch.LongTensor(vocab_length).to(next(model.parameters()).device)

    sorted_garbage=vocab_length==0
    sample_k=3
    enter_index=0
    sorted_garbage[enter_index]=False

    # print(sorted_garbage)
    enter = torch.LongTensor([[enter_index]]).to(next(model.parameters()).device)

    def test_conj_end(model,inp,mem):

        if istuple:
            bs, l, inp = divided_input(inp[:, -1:])
        else:
            bs, l, inp = gathered_input(inp[:, -1:])

        if experimental_loss:
            logits, new_mem = model.sampling(inp + (mem, sampling_mode, top_w, temparature))
        else:
            logits, new_mem = model(inp + (None, mem))

        # print(logits[:,0])
        logits[:, sorted_garbage] = -1e3
        # print(logits[:,0])
        logits = top_whatever(logits, k=top_w)
        _, top_indexes=torch.topk(logits,sample_k,-1)

        res = set([enter_index]).issubset(set(top_indexes[0].tolist()))
        # if res:
        #     print(top_indexes)
        return res

    istuple = True if isinstance(inp, tuple) else False
    token_length_holder = []
    test_batch=inp
    final_result=[]

    for batch in test_batch:
        inp=batch.view(1,-1)
        # inp = torch.cat([inp, enter], -1)
        res = inp
        text_idx=inp.tolist()[0]
        # text_idx=rollback_idx(text_idx,inv_dic)
        text=encoder.decode(text_idx)

        original_length = len(text.replace(" ", ""))
        token_length=[]
        current_length=original_length
        mem, _ = get_mem(model, inp)

        for i, cll in enumerate(lengths):

            object_length=original_length+sum(lengths[:i+1])

            # copy variable for roll-back condition
            init_res = deepcopy(res)
            init_inp = deepcopy(inp)

            while current_length < object_length:
                # print(current_length,object_length)
                with torch.no_grad():
                    if istuple:
                        bs, l, inp = divided_input(inp[:, -1:])
                    else:
                        bs, l, inp = gathered_input(inp[:, -1:])
                    if experimental_loss:
                        logits, new_mem = model.sampling(inp + (mem, sampling_mode, top_w, temparature))
                    else:
                        logits, new_mem = model(inp + (None, mem))

                logits = logits.view(bs, l, -1)

                logits = logits[:, -1, :] / temparature

                logits=remove_over_length(logits,vocab_length,object_length-current_length)

                # logits[:,sorted_garbage]=-1e4
                logits[:,enter_index]=-1e4
                logits[:,12]=-1e4

                logits = top_k_logits(logits, k=top_w)

                sampled = torch.multinomial(torch.softmax(logits, -1), 1)

                pas = encoder.decode(sampled[0].tolist())
                inp = sampled
                mem = tuple([new_mem[i][..., :-1, :] for i in range(len(mem))])
                # mem = [torch.cat([mem[i], new_mem[i][:, :-1]], 1) for i in range(len(mem))]

                # test if must roll-back to initial step of one line.
                if current_length+count_syllable(pas.strip())==object_length-1 or (current_length+count_syllable(pas.strip())==object_length \
                   and not test_conj_end(model,inp,mem)):

                    # roll back variable to initial state
                    mem, _ = get_mem(model,init_inp)
                    res = deepcopy(init_res)
                    inp = deepcopy(init_inp)
                    current_length = original_length + sum(lengths[:i])
                else:
                    current_length += vocab_length[sampled[0].tolist()[0]].item()
                    # current_length += count_syllable(pas.strip())
                    res=torch.cat([res,sampled],1)
                    # pas = pas.replace(" ", "")
                    # pas = re.sub(r'.', '\n', pas)

                # print(current_length)
                # print(pas)
                # print(count_syllable(pas.strip()))
            # add mem to enter latent space
            # print(pas)
            res=torch.cat([res,enter],-1)
            inp=res
            mem, _ = get_mem(model, inp)

        token_length_holder.append(token_length)
        final_result.extend(res.tolist())
    if istuple:
        return inp[1].tolist()
    else:
        return res.tolist()


def block_words(generated, ngram):
    target = ' '.join(map(str,generated[-ngram+1:]))
    temp = ' '.join(map(str, generated))
    blocked = re.findall('(?<={} )\d+'.format(target), temp)
    return [int(i) for i in blocked]
#
# def index_text(encoder, text, dic):
#     indexed = encoder.encode(text)
#     indexed = convert_idx_list(indexed, dic)
#     indexed = torch.Tensor(indexed).long()[None]
#     return indexed


def beam_sample(model, lengths, inp, beam_size, temparature, experimental_loss, sampling_mode=0, block_ngram=4):
    # def block(logits, res, ngram=4):
    #     """
    #     :param logits: Tensor [batch, beam, vocab_size]
    #     :param res: Tensor [batch, beam, len]
    #     :param ngram: int
    #     :return:
    #     """
    #     for batch, batch_logit in zip(res, logits):
    #         for beam, logit in zip(batch, batch_logit):
    #             generated = list(beam.to('cpu').numpy())
    #             blocked = block_words(generated,ngram)
    #             logit[blocked] = -6e4
    #     return logits

    def beam_start(logits, probs, mem, res):
        s = mem[0].size()[1:]
        logits = logits[:, -1, :] / temparature
        p, i = torch.topk(torch.log_softmax(logits, -1), beam_size, -1) #[batch, beam_size]
        probs = probs + p
        res = torch.cat([res[:,None].repeat(1,beam_size,1), i[...,None]],2) #[batch, beam, l]
        return probs, i.view(-1)[:,None], [i[:,None].repeat(1,beam_size,1,1).view((-1,)+s) for i in mem], res

    def beam_continue(logits, probs, mem, res):
        logits = logits[:, -1, :] / temparature
        logits = logits.view(bs, beam_size, -1)
        # logits = block(logits,res,block_ngram)
        p, i = torch.topk(torch.log_softmax(logits, -1), beam_size, -1)  # [batch_size, beam_size, beam_size]
        probs = probs.unsqueeze(-1) + p
        new_probs = probs.view(bs, -1)
        probs, ni = new_probs.topk(beam_size, -1)
        sampled = i.view(bs, -1).gather(1, ni) #[batch, beam]
        group = ni // beam_size
        ind = torch.arange(bs)[:, None], group
        res = res[ind]
        res = torch.cat([res, sampled[..., None]], 2)
        lh = mem[0].size()[1:]
        reshaped_mem = [i.view((bs,beam_size) + lh) for i in mem]
        mem = [i[ind].view((-1,)+lh) for i in reshaped_mem]
        return probs, sampled.view(-1)[:,None], mem, res

    def finalize(probs, res):
        _, ind = probs.topk(1,-1)
        return res[torch.arange(bs),ind.squeeze(-1)]

    istuple = True if isinstance(inp, tuple) else False
    mem, inp = get_mem(model, inp)
    bs = inp.size(0)
    res = inp
    cnt = 0
    probs = torch.zeros((inp.size(0),beam_size), dtype=inp.dtype,device=inp.device)
    for _ in range(lengths):
        cnt+=1
        with torch.no_grad():
            if istuple:
                ts, l, inp = divided_input(inp)
            else:
                ts, l, inp = gathered_input(inp[:,-1:])
            if experimental_loss:
                logits, new_mem = model.sampling(inp + (mem, sampling_mode, beam_size, temparature))
            else:
                logits, new_mem = model(inp + (None, mem))
            mem = [torch.cat([mem[i], new_mem[i].to(mem[i].dtype)[:,:-1]],1) for i in range(len(mem))]
            logits = logits.view(ts,l,-1)
            if cnt ==1:
                probs, sampled, mem, res = beam_start(logits,probs,mem, res)
            else:
                probs, sampled, mem, res = beam_continue(logits, probs, mem, res)
            if istuple:
                title, cind, tls, cls = inp
                cind = sampled
                inp = (title, cind,tls,cls)
            else:
                inp = sampled

    res = finalize(probs, res)
    if istuple:
        return res.tolist(), _
    else:
        return res.tolist(), _


def compute_prob(encoder_path, model, texts, experimental_loss):
    enc = get_encoder(encoder_path)
    texts = index_text(enc,texts,experimental_loss).to(next(model.parameters()).device)
    probs = []
    with torch.no_grad():
        for i in range(len(texts[0]) - 1):
            batch_text = texts[:, :i+1]
            target_word = texts[:, i+1]
            lens = torch.Tensor([i + 2]).long().to(batch_text.device)
            inp_mask = mask_lengths(lens, reverse=True).byte().to(batch_text.device)
            if experimental_loss:
                logits = model.sampling(torch.cat([batch_text, torch.Tensor([0])[:, None].long().to(batch_text.device)], 1),
                                        None, inp_mask, True)
            else:
                logits = model(torch.cat([batch_text, torch.Tensor([0])[:, None].long().to(batch_text.device)], 1), None, inp_mask,
                               None)
            logits = logits[0] # [vocab_size]
            prob = torch.softmax(logits,0)
            probs.append(prob[target_word[0]].item())
    print(probs)


def sample_hook(model, lengths, inp, top_k, top_p, temparature, experimental_loss, hard_sample, is_hard,
              line_idx, check_line, add_indices):
    probs = None
    istuple = True if isinstance(inp, tuple) else False
    line_num = torch.sum(inp ==0)
    for _ in range(lengths):
        with torch.no_grad():
            if istuple:
                bs, l, inp = divided_input(inp)
            else:
                bs, l, inp = gathered_input(inp)
            if experimental_loss:
                logits, __ = model.sampling(inp + (None, is_hard, hard_sample))
            else:
                logits, __ = model(inp + (None, None))
            logits = logits.view(bs,l,-1)
            logits = logits[:,-1,:] / temparature
            saved_logits = logits
            logits = top_k_logits(logits, k=top_k)
            # logits = top_p_logits(logits, p=top_p)
            sampled = torch.multinomial(torch.softmax(logits,-1),1)
            if sampled == line_idx:
                line_num+=1
                if not line_num % check_line and line_num // check_line >0:
                    sampled = torch.cat([sampled, add_indices],1)
            # print(_, sampled.size())
            # temp_probs = torch.softmax(saved_logits, 1)
            # probs = torch.cat([probs,temp_probs[torch.arange(len(sampled)),sampled.squeeze(1)][:,None]],1) \
            #     if probs is not None else temp_probs[torch.arange(len(sampled)),sampled.squeeze(1)][:,None]
            if istuple:
                title, cind, tls, cls = inp
                cind = torch.cat([cind[:,:-1], sampled], -1)
                inp = (title, cind)
            else:
                indexed,lens = inp
                inp = torch.cat([indexed[:,:-1], sampled], -1)
    if istuple:
        return inp[1].tolist(), _
    else:
        return inp.tolist(), _

def remove_over_length(logits,vocab_length_tensor,length):
    return torch.where(
        vocab_length_tensor > length,
        torch.ones_like(logits, dtype=logits.dtype) * -1e4,
        logits,
    )

def english_syllable_count(word):
    return len(
        re.findall('(?!e$)[aeiouy]+', word, re.I) +
        re.findall('^[^aeiouy]*e$', word, re.I)
    )




def count_syllable(word):
    cnt=0
    # print(bool(re.search("[ㄱ-ㅎ|ㅏ-ㅣ|가-힣]", word)))
    # print(word)
    # print(bool(re.search("[ㄱ-ㅎㅏ-ㅣ가-힣]", word)))
    if bool(re.search("[ㄱ-ㅎㅏ-ㅣ가-힣]", word)):
        for i,w in enumerate(word):
            if bool(re.search("[ㄱ-ㅎㅏ-ㅣ가-힣]",w)):
                cnt+=1
    else:
        cnt = english_syllable_count(word)
    # print(cnt)
    return cnt


def structured_sample(model, lengths, inp, top_w, temparature, experimental_loss, sampling_mode=0):
    model.eval()
    top_whatever = top_k_logits if isinstance(top_w, int) else top_p_logits
    cnt = 0
    gen_txt, gen_st = inp
    for _ in range(lengths):
        cnt+=1
        with torch.no_grad():
            inp = gen_txt, gen_st
            bs, l, inp = structured_input(inp)
            if experimental_loss:
                logits, target_st, new_txt_mem, new_st_mem = model.sampling(inp + (None, None, sampling_mode, top_w, temparature))
            else:
                logits, new_mem = model(inp + (None, None))
            # txt_mem = [torch.cat([txt_mem[i], new_txt_mem[i].to(txt_mem[i].dtype)[:,:-1]],1) for i in range(len(txt_mem))]
            # st_mem = [torch.cat([st_mem[i], new_st_mem[i].to(st_mem[i].dtype)[:,:-1]],1) for i in range(len(st_mem))]
            logits = top_whatever(logits, top_w)
            logits = logits.view(bs,l,-1)
            logits = logits[:,-1,:] / temparature
            sampled = torch.multinomial(torch.softmax(logits,-1),1)

            target_st = target_st.view(bs,l,-1)
            target_st = target_st[:,-1] / temparature
            sampled_st = torch.multinomial(torch.softmax(target_st.to(torch.float32),-1),1)

            gen_txt = torch.cat([gen_txt,sampled],1)
            gen_st = torch.cat([gen_st,sampled_st],1)
    return gen_txt.tolist(), gen_st.tolist()
