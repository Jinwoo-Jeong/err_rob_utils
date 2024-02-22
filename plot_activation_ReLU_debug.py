# %%
#%load_ext autoreload
#%autoreload 2

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import torch
import numpy as np

# %%
model_id = "facebook/opt-125m"
device = "cuda" if torch.cuda.is_available() else "cpu"
#model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# %%
def get_wikitext2(nsamples, seed, seqlen, model):
    from datasets import load_dataset

    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

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
        traindataset.append({"input_ids": inp, "attention_mask": attention_mask})
    return traindataset, testenc

traindataset, testenc = get_wikitext2(128, 0, 2048, 'facebook/opt-125m')

quantize_config = BaseQuantizeConfig(
    bits=4, group_size=128, desc_act=False
)

model_quant = AutoGPTQForCausalLM.from_pretrained(model_id, quantize_config=quantize_config)
model_quant.quantize(traindataset, use_triton=False)

# %%
from utils.act_scale import get_actin, get_actout
fc1_actout = get_actout(model, tokenizer, 10, 512, device, ['fc1'])
fc2_actin = get_actin(model,tokenizer, 10, 512, device, ['fc2'])

# %%
fc2_actin[next(iter(fc2_actin))].numel() / 512

# %%
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
dataset = dataset.shuffle(seed=42)
input_ids = []
for i in range(10):
    input_ids.append(tokenizer(dataset[i]["text"], return_tensors="pt",
                              max_length=512, truncation=True).input_ids.to(device))

# %%
sum = 0
for i in range(10):
    sum += (input_ids[i].numel())

sum

# %%
for key in fc2_actin.keys():
    print(f'{fc2_actin[key].nonzero().size(0) / fc2_actin[key].numel() * 100:.2f}')


