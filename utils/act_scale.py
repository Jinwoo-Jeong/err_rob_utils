import functools
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm
import torch
from utils.evaluate_opt import evaluate_opt
import gc

def get_actout(model, tokenizer, nsamples, seq_len, device, hooklist):
    model.eval()
    act = {}
    
    def stack_tensor(name, tensor):
        hidden_dim_in = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim_in).detach()
        if name in act:
            act[name] = torch.concat([act[name], tensor], dim=0)
        else:
            act[name] = tensor
        
    def hook(model, input, output, name):
        if isinstance(output, tuple):
            output = output[0]
        stack_tensor(name, output)
        
    hooks = []
    for name, m in model.named_modules():
        if name.split('.')[-1] in hooklist:
            hooks.append(
                m.register_forward_hook(
                    functools.partial(hook, name=name)
                )
            )
    
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    dataset = dataset.shuffle(seed=42)

    for i in tqdm(range(nsamples)):
        input_ids = tokenizer(dataset[i]["text"], return_tensors="pt",
                              max_length=seq_len, truncation=True).input_ids.to(device)
        model(input_ids)
    
    for h in hooks:
        h.remove()
    
    return act

def get_actin(model, tokenizer, nsamples, seq_len, device, hooklist):
    model.eval()
    act = {}
    
    def stack_tensor(name, tensor):
        hidden_dim_in = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim_in).detach()
        if name in act:
            act[name] = torch.concat([act[name], tensor], dim=0)
        else:
            act[name] = tensor
        
    def hook(model, input, output, name):
        if isinstance(input, tuple):
            input = input[0]
        stack_tensor(name, input)
        
    hooks = []
    for name, m in model.named_modules():
        if name.split('.')[-1] in hooklist:
            hooks.append(
                m.register_forward_hook(
                    functools.partial(hook, name=name)
                )
            )
    
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    dataset = dataset.shuffle(seed=42)

    for i in tqdm(range(nsamples)):
        input_ids = tokenizer(dataset[i]["text"], return_tensors="pt",
                              max_length=seq_len, truncation=True).input_ids.to(device)
        model(input_ids)
    
    for h in hooks:
        h.remove()
    
    return act

def get_actin_sparsity(model, dataset, tokenizer, nsamples, seq_len, device, hooklist):
    model.eval()
    act = {}
    
    def stack_tensor(name, tensor):
        hidden_dim_in = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim_in).detach()
        cnt_nonzero = torch.count_nonzero(tensor, dim=1) / hidden_dim_in
        if name in act:
            act[name] = torch.concat([act[name], cnt_nonzero], dim=0)
        else:
            act[name] = cnt_nonzero
        
    def hook(model, input, output, name):
        if isinstance(input, tuple):
            input = input[0]
        stack_tensor(name, input)
        
    hooks = []
    for name, m in model.named_modules():
        if name.split('.')[-1] in hooklist:
            hooks.append(
                m.register_forward_hook(
                    functools.partial(hook, name=name)
                )
            )
    
    for i in tqdm(range(nsamples)):
        input_ids = tokenizer(dataset[i]["text"], return_tensors="pt",
                              max_length=seq_len, truncation=True).input_ids.to(device)
        model(input_ids)
    
    for h in hooks:
        h.remove()
    
    return act

def get_activation(model, tokenizer, nsamples, seq_len, device):
    model.eval()
    hooking_list = [
        'fc1',
        'fc2'
    ]
    act_in = {}
    
    def stack_tensor(name, input):
        hidden_dim_in = input.shape[-1]
        input = input.view(-1, hidden_dim_in).detach()
        if name in act_in:
            act_in[name] = torch.concat([act_in[name], input], dim=0)
        else:
            act_in[name] = input
        
    
    def hook(model, input, output, name):
        if isinstance(input, tuple):
            input = input[0]
        if isinstance(output, tuple):
            output = output[0]
        stack_tensor(name, output)
        
    hooks = []
    for name, m in model.named_modules():
        if name.split('.')[-1] in hooking_list:
            hooks.append(
                m.register_forward_hook(
                    functools.partial(hook, name=name)
                )
            )
    
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    dataset = dataset.shuffle(seed=42)

    for i in tqdm(range(nsamples)):
        input_ids = tokenizer(dataset[i]["text"], return_tensors="pt",
                              max_length=seq_len, truncation=True).input_ids.to(device)
        model(input_ids)
    
    for h in hooks:
        h.remove()
    
    return act_in

def get_activation_gpt(model, tokenizer, nsamples, seq_len):
    model.eval()
    device = model.device
    hooking_list = [
                    'mlp.c_proj', 
                    #'mlp.c_fc'
                    ]

    act_in = {}
    stack_counter = 0
    
    def stack_tensor(name, input):
        hidden_dim_in = input.shape[-1]
        input = input.view(-1, hidden_dim_in).detach()
        if name in act_in:
            act_in[name] = torch.concat([act_in[name], input], dim=0)
        else:
            act_in[name] = input
    
    def hook(model, input, output, name):
        if isinstance(input, tuple):
            input = input[0]
        if isinstance(output, tuple):
            output = output[0]
        stack_tensor(name, output)
        
    hooks = []
    for name, m in model.named_modules():
        if len(name.split('.')) != 1:
            lin_id = name.split('.')[-1]
            mlp_id = name.split('.')[-2]
            if f'{mlp_id}.{lin_id}' in hooking_list:
                hooks.append(
                    m.register_forward_hook(
                        functools.partial(hook, name=name)
                    )
                )
    
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    dataset = dataset.shuffle(seed=42)
    testenc = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")

    for i in tqdm(range(nsamples)):
        input_ids = testenc.input_ids[:, seq_len*i:seq_len*(i+1)].to(model.device)
        model(input_ids)
    
    for h in hooks:
        h.remove()
    
    return act_in