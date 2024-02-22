import err_gptq_utils
import numpy as np
import sys
import argparse

import logging
from datetime import datetime

from log_conf import init_logger
import logging
from err_gptq_utils import *

from transformers import AutoTokenizer

init_logger()
logger = logging.getLogger('my')

def main():
    
    quant_method = 'gptq'

    loopcnt = 10
    seed = 44
    q_bit = 4

    num_ber = 1
    start_ber = 1e-2
    rates = np.linspace(1, num_ber, num_ber) * start_ber

    model_id = 'jwjeong/opt-125m-4bit-128g'
    device = "cuda:0"
    group_size = 128

    logger.info(f'quant method: {quant_method}')
    logger.info(f'loopcnt: {loopcnt}')
    logger.info(f'seed: {seed}')
    logger.info(f'q_bit: {q_bit}')
    logger.info(f'num_ber: {num_ber}')
    logger.info(f'start_ber: {start_ber}')
    logger.info(f'model_id: {model_id}')
    logger.info(f'device: {device}')
    logger.info(f'group_size: {group_size}')

    isOPT = False
    isGPT2 = False

    if model_id.split('/')[-1][:3] == 'opt':
        isOPT = True
    elif model_id.split('/')[-1][:4] == 'gpt2':
        isGPT2 = True

    #test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    #tokenizer = AutoTokenizer.from_pretrained(model_id)
    #encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")

    #err_gptq_utils.aiha_eval(model_id, rates, seed, loopcnt)
    if quant_method == 'gptq':
        err_gptq_utils.gptq_eval_opt_new(model_id, q_bit, rates, seed, loopcnt, device)
    elif quant_method == 'bnb':
        err_gptq_utils.bnb_eval(model_id, q_bit, rates, seed, loopcnt, device)
    else:
        err_gptq_utils.noquant_eval(model_id, rates, seed, loopcnt, device)

if __name__ == '__main__':
    main()