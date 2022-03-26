from more_itertools import sample
from transformers import GPT2LMHeadModel , GPT2Tokenizer
import torch
from torch import Tensor
from transformers import top_k_top_p_filtering
import torch.nn.functional as F

tokenizer = GPT2Tokenizer.from_pretrained('gpt2') 
model = GPT2LMHeadModel.from_pretrained('gpt2' , 
pad_token_id = tokenizer.eos_token_id)


print(tokenizer.decode(tokenizer.eos_token_id))

sentence = "I love you"
input_ids = tokenizer.encode(sentence , return_tensors = 'pt')
print(input_ids)
print(tokenizer.decode(input_ids[0]))

output = model(input_ids)
logits = output.logits
pred_ids = torch.argmax(logits, dim=-1)

def respond_to_batch(model, queries, txt_len=20, top_k=0, top_p=1.0):
    """Sample text from language model."""
    input_ids = queries
    for i in range(txt_len):
        # Get Logits
        output = model(input_ids)
        logits = output.logits
        next_token_logits = logits[:, -1, :]
        next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
        # Sample
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
        # next_token_score,next_token = torch.max(probs,dim = -1)
        input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
    return input_ids[:, -txt_len:]

for i in range(input_ids.shape[1]):
    current_id = tokenizer.decode(input_ids[:, i])
    next_id = tokenizer.decode(pred_ids[:, i])
    print(current_id, '-->', next_id)

return_outputs = respond_to_batch(model, input_ids, top_k = 50, txt_len = 50)

output = model.generate(input_ids, max_length = 50,no_repeat_ngram_size = 2, num_beams = 1)

# no_repeat_ngram_size

print(tokenizer.decode(return_outputs[0] , skip_special_tokens = True))
print("***************")
print(tokenizer.decode(output[0] , skip_special_tokens = True))