import torch
from tqdm import tqdm
from utils.evaluate_opt import get_wiki_test

def eval_gpt2(model, tokenizer):
    max_length = model.config.n_positions
    stride = 512
    encodings = get_wiki_test(tokenizer)
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
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