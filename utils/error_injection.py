import scipy.sparse as sp
import torch
import numpy as np

def lin_weight_identifier(name):
    linear_list = [
    #'k_proj',
    #'v_proj',
    #'q_proj',
    #'out_proj',
    'fc1',
    'fc2',
    ]
    isWeight = name.split('.')[-1] == 'weight'
    isLinear = name.split('.')[-2] in linear_list
    return (isWeight and isLinear)

def lin_weight_identifier_gpt(name):
    linear_list = [
        'c_proj',
        'c_fc',
    ]
    isWeight = name.split('.')[-1] == 'weight'
    isLinear = name.split('.')[-2] in linear_list
    isMLP = name.split('.')[-3] == 'mlp'
    return (isWeight and isLinear and isMLP)

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