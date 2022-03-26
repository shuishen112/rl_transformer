import torch
from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        LogitsProcessorList,
        MinLengthLogitsProcessor,
        StoppingCriteriaList,
        MaxLengthCriteria,
        TopKLogitsWarper,
        TemperatureLogitsWarper,
        NoRepeatNGramLogitsProcessor
        
)
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
model = AutoModelForCausalLM.from_pretrained("facebook/bart-base")
no_repeat_ngram_size = 2
model.config.pad_token_id = model.config.eos_token_id
input_prompt = "I love you"
input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids
logits_processor = LogitsProcessorList([
                MinLengthLogitsProcessor(10, eos_token_id=model.config.eos_token_id),
                NoRepeatNGramLogitsProcessor(no_repeat_ngram_size)
            ]
        )
################  greed search ###################

stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=20)])
outputs = model.greedy_search(input_ids, logits_processor=logits_processor, stopping_criteria=stopping_criteria,
        output_scores = True,return_dict_in_generate = True,repetition_penalty = 2.5)

result = []
for score in outputs.scores:
    char_ind = torch.max(score, dim=1)[1].item()
    result.append(tokenizer.decode(char_ind))

print(outputs)
############### sample search ####################

############### sample ###########################

# logits_warper = LogitsProcessorList([TopKLogitsWarper(50),TemperatureLogitsWarper(0.7),])
# stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=20)])
# torch.manual_seed(0)

# outputs = model.sample(
#         input_ids,
#         logits_processor=logits_processor,
#         logits_warper=logits_warper,
#         stopping_criteria=stopping_criteria,
#         output_scores = True,
#         return_dict_in_generate = True)

################# sample  end #########################

################# bearm search ########################

# from transformers import (
#         AutoTokenizer,
#         AutoModelForSeq2SeqLM,
#         LogitsProcessorList,
#         MinLengthLogitsProcessor,
#         BeamSearchScorer,
        
#         )

# num_beams = 2

# beam_scorer = BeamSearchScorer(
#              batch_size=1,
#              num_beams=num_beams,
#              device=model.device,
#              num_beam_hyps_to_keep = 1
#         )

# no_repeat_ngram_size = 2

# logits_processor = LogitsProcessorList(
#            [
#                MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id),
#                 NoRepeatNGramLogitsProcessor(no_repeat_ngram_size)

#            ]
#         )
# stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=20)])
# expand_size = 2
# expanded_return_idx = (
#             torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(input_ids.device)
#         )
# input_ids = input_ids.index_select(0, expanded_return_idx)
        
# outputs = model.beam_search(input_ids, beam_scorer,
# logits_processor=logits_processor,stopping_criteria = stopping_criteria,
# output_scores = True,return_dict_in_generate = True,no_repeat_ngram_size = 2)

# print(tokenizer.batch_decode(outputs, skip_special_tokens=True))

# print(outputs)

# result = []
# for score in outputs.scores:
#     char_ind = torch.max(score, dim=1)[1].item()
#     result.append(tokenizer.decode(char_ind))

# print(result)
# print(outputs)

######################## beam search  end #############



print(outputs.scores[0].shape)
print(len(outputs.scores))
print(tokenizer.batch_decode(outputs.sequences))