# %%
import os

import pandas as pd
import numpy as np
import torch

import bitsandbytes
import accelerate
from datasets import Dataset

from modelscope import snapshot_download
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import Trainer, TrainingArguments
from peft import PeftModel


# %%
model_path=r'/root/autodl-tmp/01ai/Yi-6B-Chat'
model_dir = snapshot_download('01ai/Yi-6B-Chat', cache_dir='/root/autodl-tmp', revision='master')

# base_tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, padding_side='left', max_length=512, return_tensors='pt')

# base_model = AutoModelForCausalLM.from_pretrained(
# 	 model_path,
# 	 device_map='auto',
# 	 torch_dtype=torch.bfloat16,
# 	 trust_remote_code=True 
# ).eval()

# %%
messages = [
    {"role": "user", "content": "Translate the following words into English:\n你係邊個？"},
]

messages_plain = [
    """
    <|im_start|> user
    hi<|im_end|> 
    <|im_start|> assistant
    """
]



# input_ids = base_tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')
# output_ids = base_model.generate(input_ids.to('cuda'))
# # response = base_tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True, max_length=100)
# response = base_tokenizer.decode(output_ids[0], skip_special_tokens=False, max_length=100)

# # Model response: "Hello! How can I assist you today?"
# print(response)

# %%
tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/01ai/Yi-6B-Chat', use_fast=True, padding_side='left', max_length=512, return_tensors='pt')
base_tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/01ai/Yi-6B-Chat', use_fast=True, padding_side='left', max_length=512, return_tensors='pt')
# print(len(tokenizer.vocab))
# tokenizer = AutoTokenizer.from_pretrained('/root/AIST4010-Cantonese-Translator/tokenizer', use_fast=True, padding_side='left', max_length=512, return_tensors='pt')

# %%
# model = PeftModel.from_pretrained(
#     model,
#    '/root/peft_model',
#     is_trainable=False
# )

# %%
REPO_DIRECTORY = r'/root/'
ABC_DICT_PATH = r'autodl-tmp/AIST4010-Cantonese-Translator-Data/ABC-Dict/abc_dict.csv'
FLORES_PATH = r'/root/autodl-tmp/AIST4010-Cantonese-Translator-Data/flores+'

#print all files in the flores+ directory
print(os.listdir(FLORES_PATH))

def load_flores_dataset():
    files = os.listdir(FLORES_PATH)
    column_names = ['cmn_Hans', 'cmn_Hant', 'eng_Latn', 'yue_Hant']
    data_dict = {column: [] for column in column_names}
    for file in files:
        if file.startswith('.'):
            continue
        data = []
        with open(os.path.join(FLORES_PATH, file), 'r') as f:
            data = f.readlines()
            data = [line.strip() for line in data]
            lang = file.split('.')[1]
            print(data)
            print(lang)
            #append data to column
            data_dict[lang] += data
    df = pd.DataFrame(data_dict)
    return df

flores_df = load_flores_dataset()  
print(flores_df)
flores_dataset = Dataset.from_pandas(flores_df)

    # return flores_train, flores_val, flores_test

# %%
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
base_model = AutoModelForCausalLM.from_pretrained(
	 '/root/autodl-tmp/01ai/Yi-6B-Chat',
	 device_map=device,
	 torch_dtype=torch.bfloat16,
     quantization_config=BitsAndBytesConfig(load_in_8bit=True),
	 trust_remote_code=True 
).eval()
model = AutoModelForCausalLM.from_pretrained(
	 model_path,
	 device_map=device,
	 torch_dtype=torch.bfloat16,
	 quantization_config=BitsAndBytesConfig(load_in_8bit=True),
	 trust_remote_code=True 
).eval()
model.resize_token_embeddings(len(tokenizer))
model.load_adapter('/root/autodl-tmp/peft_model_sft')



# def load_flores_dataset():
    

# %%

def get_messages(sample, src, tgt):
    lang_map = {'eng_Latn': 'English', 'yue_Hant': 'Cantonese', 'cmn_Hans': 'Mandarin', 'cmn_Hant': 'Mandarin'}
    def get_prompt(src, tgt, sample):
        src_name = lang_map[src]
        tgt_name = lang_map[tgt]
        system_prompt = f"Translate the given {src_name} words to {tgt_name}."
        user_prompt = sample[src]
        return system_prompt, user_prompt
    system, user = get_prompt(src, tgt, sample)
    return [[
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]]

# train_samples = abc_train_set.shuffle(seed=10).select(range(20))
# test_samples = abc_test_set.shuffle(seed=10).select(range(20))
lang_names = ['eng_Latn', 'cmn_Hans', 'cmn_Hant']
cantonese_name = 'yue_Hant'

for src in lang_names:
    print(get_messages(flores_df.iloc[0], src, cantonese_name))
    print(get_messages(flores_df.iloc[1], cantonese_name, src))
     



# %%


# %%
def model_output(model, tokenizer, messages, name=None):
        responses = []
        for prompt in messages:
                input_ids = tokenizer.apply_chat_template(conversation=prompt, tokenize=True, add_generation_prompt=True, return_tensors='pt')
                with torch.cuda.amp.autocast():
                        output_ids = model.generate(input_ids.to('cuda'), max_new_tokens=500)
                response = base_tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True, max_length=100)
                # response = tokenizer.decode(output_ids[0], skip_special_tokens=False, max_length=100)
                # print(output_ids)
                # print(f"{name}:\n{response}\n")
                responses.append(response)
        return responses

for src in lang_names:
    print(model_output(base_model, tokenizer, get_messages(flores_df.iloc[0], src, cantonese_name), 'Base model'))
    print(model_output(model, tokenizer, get_messages(flores_df.iloc[0], src, cantonese_name), 'SFT model'))

# %%
def compare_outputs(samples):
    for sample in samples:
        print(sample)
        prompts = get_messages(sample)
        for prompt in prompts:
            print(f"Prompt:\n{prompt[1]['content']}")
            print()
            model_output(base_model, base_tokenizer, [prompt], 'Base model')
            model_output(model, tokenizer, [prompt], 'Fine-tuned model')

# compare_outputs(train_samples)

# %%
model_output(model, tokenizer, get_messages(train_samples[0]), 'Fine-tuned model')

# %%
compare_outputs(train_samples)


# %%
compare_outputs(test_samples)

# %%
messages = [
    {"role": "user", "content": "Translate the following words into English:\n乜嘢都係波士決定嘅，打工仔啲人淨係得個知字。\n"},
]

# get 5 random samples from train and test dataset
train_sample = abc_train_set.shuffle(seed=42).select(range(5))
test_sample = abc_test_set.shuffle(seed=42).select(range(5))

en_train_messages = {get_translate_prompt('Cantonese', sentence) for sentence in train_sample['en']}
en_test_messages = {get_translate_prompt('Cantonese', sentence) for sentence in test_sample['en']}
yue_train_messages = {get_translate_prompt('English', sentence) for sentence in train_sample['yue']}
yue_test_messages = {get_translate_prompt('English', sentence) for sentence in test_sample['yue']}

for messages in [en_train_messages, en_test_messages, yue_train_messages, yue_test_messages]:
    for message in messages:
        print(message)


# %%


# %%
input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')
output_ids = model.generate(input_ids.to('cuda'))
# response = base_tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True, max_length=100)
response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True, max_length=100)

# Model response: "Hello! How can I assist you today?"
print("Tuned model:", response)

input_ids = base_tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')
output_ids = base_model.generate(input_ids.to('cuda'))
# response = base_tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True, max_length=100)
response = base_tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True, max_length=100)

print("Base model:", response)


