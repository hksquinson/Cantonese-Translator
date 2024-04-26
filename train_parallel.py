#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import time

import pandas as pd
import numpy as np

import torch
import nltk.translate.bleu_score as bleu


from modelscope import snapshot_download
from datasets import Dataset, load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import get_peft_model, LoraConfig, TaskType

from custom_tokenizers import YueTokenizer



# In[2]:


ABC_DICT_PATH = r'AIST4010-Cantonese-Translator-Data/ABC-Dict/abc_dict.csv'
KFCD_DICT_PATH = r'AIST4010-Cantonese-Translator-Data/kaifangcidian/kaifangcidian.csv'

def load_dataset(path):
    df = pd.read_csv(path)
    dataset = Dataset.from_pandas(df)
    return dataset



abc_set = pd.read_csv(ABC_DICT_PATH)
kfcd_set = pd.read_csv(KFCD_DICT_PATH)
# create new pd with 4 columns, src_lang, src_text, tgt_lang, tgt_text
train_df = pd.DataFrame(columns=['src_lang', 'src_text', 'tgt_lang', 'tgt_text'])
rows = []
for i, line in abc_set.iterrows():
    rows.append({'src_lang': 'en', 'src_text': line['en'], 'tgt_lang': 'yue', 'tgt_text': line['yue']})
    rows.append({'src_lang': 'yue', 'src_text': line['yue'], 'tgt_lang': 'en', 'tgt_text': line['en']})
for i, line in kfcd_set.iterrows():
    rows.append({'src_lang': 'chinese', 'src_text': line['chinese'], 'tgt_lang': 'yue', 'tgt_text': line['yue']})
    rows.append({'src_lang': 'yue', 'src_text': line['yue'], 'tgt_lang': 'chinese', 'tgt_text': line['chinese']})
train_df = pd.DataFrame(rows)
# to a dataset
train_dataset = Dataset.from_pandas(train_df) #.shuffle(seed=42)
#print samples from dataset
#iterate first 10 items of dataset
for i in range(20):
    print(train_dataset[i])
print(len(train_dataset))


# to a dataset




# In[3]:


# def count_dataset_tokens(dataset):
#     en_count = 0
#     yue_count = 0
#     for example in dataset:
#         en_count += len(example['en'])
#         yue_count += len(example['yue'])
#     return en_count, yue_count


# counts = np.array(count_dataset_tokens(abc_train_set))
# print(counts)
# print(counts/len(abc_train_set))

#print max length of the dataset
print('Max length of English sentence in ABC dataset:', max([len(x) for x in abc_set['en']]))
print('Max length of Cantonese sentence in ABC dataset:', max([len(x) for x in abc_set['yue']]))
print('Max length of Chinese sentence in KFCD dataset:', max([len(x) for x in kfcd_set['chinese']]))
print('Max length of Cantonese sentence in KFCD dataset:', max([len(x) for x in kfcd_set['yue']]))


# In[4]:


model_path=r'01ai/Yi-6B-Chat'

# model = Model.from_pretrained('01ai/Yi-6B')

# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     device_map="auto",
#     torch_dtype='auto'
# ).eval()


# tokenizer = AutoTokenizer.from_pretrained(model_name)


# In[5]:


model_dir = snapshot_download('01ai/Yi-6B-Chat', cache_dir='01ai/Yi-6B-Chat', revision='master')


# In[6]:


tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True, padding_side='right', max_length=512, return_tensors='pt')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# Since transformers 4.35.0, the GPT-Q/AWQ model can be loaded using AutoModelForCausalLM.
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map=device,
    torch_dtype='auto',
)



# In[7]:


# Prompt content: "hi"
messages = [
    {"role": "system", "content": "Translate the given English words into Chinese."},
    {"role": "user", "content": "Good morning, how are you?"},
]

print(tokenizer.apply_chat_template(conversation=messages, tokenize=False, add_generation_prompt=True, return_tensors='pt'))


input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')
output_ids = model.generate(input_ids.to('cuda'))
response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

# Model response: "Hello! How can I assist you today?"
print(response)


# In[8]:


# print(input_ids)
# print(output_ids)
# print(base_tokenizer.decode(input_ids[0]))
# print(base_tokenizer.decode(input_ids[0]))

# #get text of list of tokens in output_ids stored in array
# print([base_tokenizer.decode([token]) for token in output_ids[0]])


# In[9]:


def formatting_prompts_func(examples):
    output_texts = []
    langMap = {'en': 'English', 'yue': 'Cantonese', 'chinese': 'Mandarin'}
    # print(examples)
    for i in range(len(examples['src_text'])):
        src_lang_abbv = examples['src_lang'][i]
        tgt_lang_abbv = examples['tgt_lang'][i]
        src_lang = langMap[src_lang_abbv]
        tgt_lang = langMap[tgt_lang_abbv]
        src_text = examples['src_text'][i]
        tgt_text = examples['tgt_text'][i]
        system_prompt = f"<|im_start|>system\nTranslate the given {src_lang} words into {tgt_lang}.<|im_end|>\n"
        prompt = f"<|im_start|>user\n{src_text}<|im_end|>\n"
        response = f"<|im_start|>assistant\n{tgt_text}<|im_end|>\n"
        output_texts.append(system_prompt + prompt + response)
    # print(len(output_texts))
    return output_texts


# In[10]:


prompts = formatting_prompts_func(train_dataset[:10])
# for prompt in abc_prompts:
#     print(prompt)
for prompt in prompts:
    print(prompt)


# In[11]:


# for name, param in base_model.named_parameters():
#     print(f"Parameter name: {name}")
#     print(param)
#     print("-" * 50)


# In[12]:


# print(base_model.config)


# In[13]:


lora_config = LoraConfig(
    r=32, # Rank
    lora_alpha=32,
    target_modules = ["k_proj", "q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
peft_model = get_peft_model(model, 
                            lora_config)

peft_model.print_trainable_parameters()


# **Train Tokenizer**

# In[14]:


# def get_training_corpus(dataset):
#     for start_idx in range(0, len(dataset), 1000):
#         samples = dataset[start_idx : start_idx + 1000]
#         sample_en = samples["en"]
#         sample_yue = samples["yue"]
#         for i in range(len(sample_en)):
#             yield sample_en[i]
#             yield sample_yue[i]

# def get_yue_training_corpus(dataset):
#     for start_idx in range(0, len(dataset), 1000):
#         samples = dataset[start_idx : start_idx + 1000]
#         sample_yue = samples["yue"]
#         for i in range(len(sample_yue)):
#             yield sample_yue[i]

# training_corpus = get_yue_training_corpus(abc_train_set)


# curr_vocab = set(tokenizer.vocab)
# # print(curr_vocab)
# new_vocab = set()
# #iterate through all training_corpus lines
# for i, line in enumerate(training_corpus):
#     unique_chars = set(list(line))
#     print(unique_chars)
#     new_tokens = unique_chars - curr_vocab - new_vocab
#     new_vocab |= new_tokens


# for i in range(training_corpus):
#     next(training_corpus)
#     line = next(training_corpus)
#     for char in line:
#         if char not in tokenizer.vocab and char not in new_vocab:
#             new_vocab.add(char)
# print(new_vocab)
# print(len(new_vocab))
# tokenizer.add_tokens(list(new_vocab))
# tokenizer.save_pretrained("/root/tokenizer")


# In[15]:


# print(tokenizer("嗌呃畀啲嘢噃")['input_ids'])
# print(base_tokenizer("嗌呃畀啲嘢噃")['input_ids'])
# print(tokenizer.tokenize("嗌呃畀啲嘢噃"))
# print(base_tokenizer.tokenize("嗌呃畀啲嘢噃"))
# print(tokenizer("Good morning")['input_ids'])
# print(base_tokenizer("Good morning")['input_ids'])


# In[16]:


# bleu = evaluate.load('bleu')

# def compute_metrics(eval_preds):
#     logits, labels = eval_preds
#     predictions = np.argmax(logits, axis=-1)
#     print(predictions.shape, labels.shape)
#     return {"bleu": bleu(predictions, labels)}


# In[17]:


# peft_model.resize_token_embeddings(len(tokenizer))


# In[19]:

# get time stamp
timestr = time.strftime("%Y%m%d-%H%M%S")

# peft_model.resize_token_embeddings(len(tokenizer))
log_dir = f"tf-logs/{timestr}"


training_args = TrainingArguments(
    learning_rate=3e-4, 
    num_train_epochs=3,
    # max_steps = 200,
    logging_steps=10,
    output_dir="peft_model_sft_only_checkpoints",
    logging_dir=log_dir,
)

# response_template = "<|im_start|>assistant\n"
# collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

trainer = SFTTrainer(
    peft_model,
    args=training_args,
    train_dataset= train_dataset,
    tokenizer=tokenizer,
    formatting_func=formatting_prompts_func,
    # data_collator=collator,
)
trainer.train()


# In[ ]:


# In[ ]:


# #get random data from test dataset
# for i in range(5):
#     example = abc_test_set[i]
#     print(example)
#     text1 = f"""Translate the following words into Cantonese: 
#         {example['en']}
#         """
#     text2 = f"""Translate the following words into English:
#         {example['yue']}
#         """
#     texts = [text1, text2]
#     for text in texts:
#         messages = [
#             {"role": "user", "content": text}
#         ]
#         print(messages)
#         #print model outputs for base_model and peft_model
#         base_input_ids = base_tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')
#         peft_input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')
#         print("Base ID:", base_input_ids)
#         print("Base Input:", base_tokenizer.decode(base_input_ids[base_input_ids.shape[1]:], skip_special_tokens=True))
#         print("PEFT ID:", peft_input_ids)
#         print("PEFT Input:", tokenizer.decode(peft_input_ids[peft_input_ids.shape[1]:], skip_special_tokens=True))
#         print(peft_input_ids)
#         base_output_ids = base_model.generate(base_input_ids.to('cuda'), max_new_tokens=100)
#         peft_output_ids = peft_model.generate(peft_input_ids.to('cuda'), max_new_tokens=100)
#         print(base_output_ids.shape, peft_output_ids.shape)
#         print("Base model: ", base_tokenizer.decode(base_output_ids[0][base_input_ids.shape[1]:], skip_special_tokens=True))
#         print("Fine-tuned: ", tokenizer.decode(peft_output_ids[0][peft_input_ids.shape[1]:], skip_special_tokens=True))


# In[ ]:


# print(pd.DataFrame(trainer.state.log_history))
#save trainer log history
trainer.model.save_pretrained("peft_model_sft_only")

pd.DataFrame(trainer.state.log_history).to_csv("peft_model_sft_only/log_history.csv")

