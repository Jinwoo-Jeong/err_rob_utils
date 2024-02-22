from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import sys
import torch.nn as nn
import tqdm
from datasets import load_dataset
from copy import deepcopy

import logging

logger = logging.getLogger('my')

import numpy as np

import torch
import scipy.sparse as sp
import numpy as np

def conv_bin2int(bin, bitwidth):
    if bitwidth == 32:
        dtype = torch.int32
    elif bitwidth == 16:
        dtype = torch.int16
    else:
        dtype = torch.int8
    
    bin = bin.view(-1, bitwidth).T.type(dtype)
    sign = bin[0]
    sum = torch.zeros(bin.size(-1), dtype=dtype)
    for i in range(1, len(bin)):
        bin[i] = sign ^ bin[i]
        sum += bin[i] * 2**((bitwidth-1)-i)
    
    mult_sign = torch.where(sign == 0, torch.tensor(1, dtype=dtype), torch.tensor(-1, dtype=dtype))

    sum = (mult_sign*sum) - sign.view(dtype)
    return sum
    

def error_gen(param, rate, seed):
    orig_size = param.size()
    bitwidth = param.data.element_size()*8
    
    bin_error = torch.tensor(sp.random(np.prod(orig_size), bitwidth, density=rate, dtype=bool, random_state=np.random.default_rng(seed)).toarray())
    error_matrix = conv_bin2int(bin_error, bitwidth)
    del bin_error
    return error_matrix.view(orig_size)

def error_injection(param, rate, seed, device="cuda"):
    err_mat = error_gen(param, rate, seed).to(device)
    int_form = err_mat.dtype
    param.data[:] = (param.data.view(int_form) ^ err_mat).view(param.dtype)
    del err_mat

def get_wikitext2(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    except:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    import random
    random.seed(seed)
    np.random.seed(0)
    torch.random.manual_seed(0)
    
    traindataset = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        attention_mask = torch.ones_like(inp)
        traindataset.append({'input_ids':inp,'attention_mask': attention_mask})
    return traindataset, testenc

def get_wiki(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    except:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    import random
    random.seed(seed)
    np.random.seed(0)
    torch.random.manual_seed(0)
    
    traindataset = []
    return testenc.input_ids

def eval_gpt2(model, encodings):
    max_length = model.config.n_positions
    stride = 1024
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(model.device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        #print(model)
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over input tokens.
            # Multiply it with trg_len to get the summation instead of average.
            # We will take average over all the tokens to get the true average
            # in the last step of this example.
            neg_log_likelihood = outputs.loss.type(torch.float) * trg_len

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
        
        
    #act_ber = (BER * 628) / (total_word_cnts * 32)
    debug_point1 = torch.stack(nlls).sum()
    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    #print(f'model: {model_id}, BER: {BER}, perplexity:{ppl}')
    
    return round(ppl.item(), 5)

@torch.no_grad()
def opt_eval(model, testenc, dev, BER, seqlen = 2048):
    print('Evaluating ...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):

        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in tqdm.tqdm(range(nsamples)):
        batch = testenc[:, (i * seqlen):((i + 1) * seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    for i in tqdm.tqdm(range(len(layers))):
        layer = layers[i].to(dev)

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.decoder.final_layer_norm is not None:
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
    if model.model.decoder.project_out is not None:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.decoder.final_layer_norm is not None:
            hidden_states = model.model.decoder.final_layer_norm(hidden_states)
        if model.model.decoder.project_out is not None:
            hidden_states = model.model.decoder.project_out(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * seqlen):((i + 1) * seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
    print(ppl.item())

    model.config.use_cache = use_cache
    return ppl

def evaluate_opt(model, testenc, device):
    model = model
    nsamples = 40
    model = model.eval()

    nlls = []
    for i in tqdm.tqdm(range(nsamples), desc="evaluating..."):
        batch = testenc[:, (i * 2048):((i + 1) * 2048)].to(model.device)
        with torch.no_grad():
            lm_logits = model(batch).logits
        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = testenc[:, (i * 2048):((i + 1) * 2048)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * 2048
        nlls.append(neg_log_likelihood)

    return torch.exp(torch.stack(nlls).sum() / (nsamples * 2048))

def lin_weight_identifier(name):
    linear_list = [
    #'k_proj',
    #'v_proj',
    #'q_proj',
    #'out_proj',
    #'fc1',
    'fc2',
    ]
    isWeight = name.split('.')[-1] == 'weight'
    isLinear = name.split('.')[-2] in linear_list
    return (isWeight and isLinear)

def lin_weight_identifier_gpt2(name):
    linear_list = [
    #'k_proj',
    #'v_proj',
    #'q_proj',
    #'out_proj',
    'c_fc',
    'c_proj',
    ]
    isWeight = name.split('.')[-1] == 'weight'
    isLinear = name.split('.')[-2] in linear_list
    isMLP = name.split('.')[-3] == 'mlp'
    return (isWeight and isLinear and isMLP)

def lin_weight_identifier_llama(name):
    linear_list = [
    #'k_proj',
    #'v_proj',
    #'q_proj',
    #'out_proj',
    'down_proj',
    'gate_proj',
    'up_proj'
    ]
    isWeight = name.split('.')[-1] == 'qweight'
    isLinear = name.split('.')[-2] in linear_list
    return (isWeight and isLinear)

def gptq_eval_opt(model_id, q_bit, rates, seed, loopcnt, device):
    print('gptq_eval')
    pretrained_model_dir = model_id
    _, testenc = get_wikitext2(128, 0, 2048, pretrained_model_dir)
    dev = device

    res = {}
    for i in rates:
        res[i]={}
        for j in range(loopcnt):
            quantized_model_dir = model_id.split('/')[-1]+'-'+ str(q_bit) + 'bit-128g'
            model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir, device=dev, use_triton=False)

            for name, param in model.named_parameters():
                if lin_weight_identifier(name) is True:
                    error_injection(param, i, seed+j, dev)

            res[i][j] = opt_eval(model.model, testenc, dev, seqlen = 2048)
            logger.info(f'{j}th loop, RBER: {i:.2e}, ppl: {res[i][j]:.2f}')

    for i in res.keys():
        print(f'RBER: {i}')
        for j in res[i].keys():
            print(f'{j}th loop: ppl={res[i][j]}')

def gptq_eval_opt_new(model_id, q_bit, rates, seed, loopcnt, device):
    print('gptq_eval')
    pretrained_model_dir = model_id
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    testenc = load_testenc(tokenizer, device)
    dev = device

    res = {}
    for i in rates:
        res[i]={}
        for j in range(loopcnt):
            quantized_model_dir = model_id
            model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir, device=dev, use_triton=False)

            #for name, param in model.named_parameters():
            #    if lin_weight_identifier(name) is True:
            #        error_injection(param, i, seed+j, dev)

            res[i][j] = evaluate_opt(model=model, testenc=testenc.to(device), device=model.device)
            print(res[i][j])
            logger.info(f'{j}th loop, RBER: {i:.2e}, ppl: {res[i][j]:.2f}')

    for i in res.keys():
        print(f'RBER: {i}')
        for j in res[i].keys():
            print(f'{j}th loop: ppl={res[i][j]}')
            
def gptq_eval_llama(model_id, q_bit, rates, seed, loopcnt, device, group_size):
    print('llama2_eval')
    pretrained_model_dir = model_id
    _, testenc = get_wikitext2(128, 0, 2048, pretrained_model_dir)
    dev_num = int(device.split(":")[-1])
    res = {}
    for i in rates:
        res[i]={}
        for j in range(loopcnt):
            quantized_model_dir = model_id
            model = AutoModelForCausalLM.from_pretrained(quantized_model_dir, device_map={"":dev_num}, trust_remote_code=True, revision='main')

            for key in model.state_dict().keys():
                if lin_weight_identifier_llama(key) is True:
                    param = model.state_dict()[key]
                    error_injection(param, i, seed+j, device)

            res[i][j] = evaluate_opt(model=model, testenc=testenc.to(device), device=model.device)
            logger.info(f'{j}th loop, RBER: {i:.2e}, ppl: {res[i][j]:.2f}')

    for i in res.keys():
        print(f'RBER: {i}')
        for j in res[i].keys():
            print(f'{j}th loop: ppl={res[i][j]}')

def gptq_eval_llama_quant(model_id, q_bit, rates, seed, loopcnt, device, group_size):
    print('llama2_eval')
    pretrained_model_dir = model_id
    testenc = get_wiki(128, 0, 2048, 'SparseLLM/ReluLLaMA-7B')
    dev_num = int(device.split(":")[-1])
    res = {}
    for i in rates:
        res[i]={}
        for j in range(loopcnt):
            quantized_model_dir = model_id
            model = AutoGPTQForCausalLM.from_quantized(pretrained_model_dir, device=device, use_safetensors=True, use_triton=False)

            for key in model.state_dict().keys():
                if lin_weight_identifier_llama(key) is True:
                    param = model.state_dict()[key]
                    error_injection(param, i, seed+j, device)

            res[i][j] = evaluate_opt(model=model, testenc=testenc.to(device), device=model.device)
            logger.info(f'{j}th loop, RBER: {i:.2e}, ppl: {res[i][j]:.2f}')

    for i in res.keys():
        print(f'RBER: {i}')
        for j in res[i].keys():
            print(f'{j}th loop: ppl={res[i][j]}')

def gptq_eval_gpt2(model_id, encodings, q_bit, rates, seed, loopcnt, device, group_size):
    print('gptq_eval_gpt2')
    dev = device
    repo_id = f"jwjeong/{model_id}-{q_bit}bit-gptq-{group_size}g"

    res = {}
    for i in rates:
        res[i]={}
        for j in range(loopcnt):
            model = AutoGPTQForCausalLM.from_quantized(repo_id, device=device, use_safetensors=True, use_triton=False)

            for name, param in model.named_parameters():
                if lin_weight_identifier_gpt2(name) is True:
                    error_injection(param, i, seed+j, dev)

            res[i][j] = eval_gpt2(model.model, encodings)
            logger.info(f'{j}th loop, RBER: {i:.2e}, ppl: {res[i][j]:.2f}')

    for i in res.keys():
        print(f'RBER: {i}')
        for j in res[i].keys():
            print(f'{j}th loop: ppl={res[i][j]}')

def bnb_eval(model_id, q_bit, rates, seed, loopcnt, device):
    print('bnb_eval')
    dev = device
    dev_num = int(device.split(":")[-1])
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    testenc = load_testenc(tokenizer, dev)
    res = {}

    if q_bit == 8:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        bnb_config = BitsAndBytesConfig(load_in_4bit=True)
    
    for i in rates:
        res[i] = {}
        for j in range(loopcnt):
            model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":dev_num})
            for name, param in model.named_parameters():
                if lin_weight_identifier(name) is True:
                    error_injection(param, i, seed+j, dev)
            
            res[i][j] = evaluate_opt(model, testenc, dev)
            logger.info(f'{j}th loop, RBER: {i:.2e}, ppl: {res[i][j]:.2f}')

    for i in res.keys():
        print(f'RBER: {i}')
        for j in res[i].keys():
            print(f'{j}th loop: ppl={res[i][j]}')


def noquant_eval(model_id, rates, seed, loopcnt, device):
    print('noquant_eval')
    dev = device
    dev_num = int(device.split(":")[-1])
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    testenc = load_testenc(tokenizer, dev)
    res = {}
    for i in rates:
        res[i] = {}
        for j in range(loopcnt):
            torch.cuda.empty_cache()
            model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
            for name, param in model.named_parameters():
                if lin_weight_identifier(name) is True:
                    error_injection(param, i, seed+j, dev)
            
            res[i][j] = evaluate_opt(model, testenc, dev)
            logger.info(f'{j}th loop, RBER: {i:.2e}, ppl: {res[i][j]:.2f}')
            del model

    for i in res.keys():
        print(f'RBER: {i}')
        for j in res[i].keys():
            print(f'{j}th loop: ppl={res[i][j]}')

def aiha_eval(model_id, rate, seed, loopcnt):
    pretrained_model_dir = model_id
    quantized_model_dir = 'opt-6.7b-4bit-128g'
    dev = "cuda:0"
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir)
    testenc = load_testenc(tokenizer, dev)

    res = {}
    for i in range(loopcnt):
        for j in rate:
            model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir, device=dev, use_triton=False)
            for name, param in model.model.named_parameters():
                if lin_weight_identifier(name) is True:
                    error_injection(param, j, seed+i, dev)

            res[i] = evaluate_opt(model.model, testenc, dev)
            print(res[i])

    for i in res.keys():
        print(f'{i}th loop: ppl={res[i]}')

def aiha_eval_8bit(rate):
    pretrained_model_dir = 'facebook/opt-6.7b'
    dev = "cuda:0"
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir)
    testenc = load_testenc(tokenizer, dev)

    res = {}
    for i in range(1):
        quantized_model_dir = 'opt-6.7b-8bit-128g'
        model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir, device=dev, use_triton=False)

        for name, param in model.model.named_parameters():
            if lin_weight_identifier(name) is True:
                error_injection(param, rate, 43+i, dev)

        res[i] = evaluate_opt(model.model, testenc, dev)

    for i in res.keys():
        print(f'{i}th loop: ppl={res[i]}')

def load_testenc(tokenizer, device):
    testenc = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    testenc = tokenizer("\n\n".join(testenc['text']), return_tensors='pt')
    return testenc.input_ids.to(device)

if __name__ == '__main__':
    rates = np.linspace(1, 5, 5) * 1e-4
    aiha_eval(rates)