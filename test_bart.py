import torch
import wandb
import time
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
tqdm.pandas()

from datasets import load_dataset


from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import BartTokenizer
from trl.bart import BartHeadWithValueModel, respond_to_batch
from trl.ppo import PPOTrainer
from trl.core import build_bert_batch_from_txt

config = {
    "lm_name": "facebook/bart-base",
    "ref_lm_name": "facebook/bart-base",
    "cls_model_name": "lvwerra/distilbert-imdb",
    "tk_name": "facebook/bart-base",
    "steps": 25600,
    "batch_size": 256,
    "forward_batch_size": 16,
    "ppo_epochs": 4,   
    "txt_in_len": 5,
    "txt_out_len": 10,
    "lr": 1.41e-5,
    "init_kl_coef":0.2,
    "target": 6,
    "horizon":10000,
    "gamma":1,
    "lam":0.95,
    "cliprange": .2,
    "cliprange_value":.2,
    "vf_coef":.1, 
}

# wandb.init(name='run-42', project='bart-sentiment', config=config, )

ds = load_dataset('imdb', split='train')
ds = ds.rename_columns({'text': 'review', 'label': 'sentiment'})
ds.set_format('pandas')
df = ds[:]

sentiment_model = AutoModelForSequenceClassification.from_pretrained(config["cls_model_name"])
sentiment_tokenizer = AutoTokenizer.from_pretrained(config["cls_model_name"])

# make sure the comments are long enough
df = df.loc[df['review'].str.len() > 500][:256]

# make sure comments are not too long
df['review'] = df['review'].apply(lambda x: x[:1000])

df.head()
print(df.head())


text = 'this movie was really bad!!'
output = sentiment_model.forward(**sentiment_tokenizer(text, return_tensors="pt"))
print(output)

bart_model = BartHeadWithValueModel.from_pretrained(config['lm_name'])
bart_model_ref = BartHeadWithValueModel.from_pretrained(config['ref_lm_name'])
bart_tokenizer = BartTokenizer.from_pretrained(config['tk_name'])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_ = bart_model.to(device)
_ = sentiment_model.to(device)
_ = bart_model_ref.to(device)

df['tokens'] = df['review'].progress_apply(lambda x: bart_tokenizer.encode(x, return_tensors="pt").to(device)[0, :config['txt_in_len']])

df['query'] = df['tokens'].progress_apply(lambda x: bart_tokenizer.decode(x))

print(df.head())

ppo_trainer = PPOTrainer(bart_model, bart_model_ref, **config)
fbs = config['forward_batch_size']

for epoch in tqdm(range(int(np.ceil(config["steps"]/config['batch_size'])))):
    torch.cuda.empty_cache()
    logs = dict()
    game_data = dict()
    timing = dict()
    t0 = time.time()
    
    #### get a batch from the dataset
    df_batch = df.sample(config['batch_size'])
    game_data['query'] = df_batch['query'].tolist()
    query_tensors = torch.stack(df_batch['tokens'].tolist())
    
    #### get response from bart
    t = time.time()
    total_length = config['txt_in_len']+config['txt_out_len']
    response_tensors = []
    for i in range(int(config['batch_size']/fbs)):
        response  = respond_to_batch(bart_model, query_tensors[i*fbs:(i+1)*fbs],
                                     txt_len=config['txt_out_len'],top_k=50)
        response_tensors.append(response)
    response_tensors = torch.cat(response_tensors)
    game_data['response'] = [bart_tokenizer.decode(response_tensors[i, :]) for i in range(config['batch_size'])]
    print(game_data['response'])
    timing['time/get_response'] = time.time()-t

    #### tokenize text for sentiment analysis
    t = time.time()
    texts = [q + r for q,r in zip(game_data['query'], game_data['response'])]
    sentiment_inputs, attention_masks = build_bert_batch_from_txt(texts, sentiment_tokenizer, device)    
    timing['time/build_input_sentiment'] = time.time()-t

    #### get sentiment score
    t = time.time()
    rewards = []
    for i in range(int(config['batch_size']/fbs)):
        res = sentiment_model.forward(sentiment_inputs[i*fbs:(i+1)*fbs],
                                      attention_masks[i*fbs:(i+1)*fbs])[0][:, 1].detach()
        rewards.append(res)
    rewards = torch.cat(rewards)
    timing['time/get_sentiment_preds'] = time.time()-t
    
    #### Run PPO training 
    t = time.time()
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    timing['time/optimization'] = time.time()-t
     
    #### Log everything
    timing['time/epoch'] = time.time()-t0
    table_rows = [list(r) for r in zip(game_data['query'], game_data['response'], rewards.cpu().tolist())]
    logs.update({'game_log':wandb.Table(
        columns=['query', 'response', 'reward'],
        rows=table_rows)})
    logs.update(timing)
    logs.update(stats)
    logs['env/reward_mean'] = torch.mean(rewards).cpu().numpy()
    logs['env/reward_std'] = torch.std(rewards).cpu().numpy()
    logs['env/reward_dist'] = rewards.cpu().numpy()
    # wandb.log(logs)