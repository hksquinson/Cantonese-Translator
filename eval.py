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

from tqdm import tqdm


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
# messages = [
#     {"role": "user", "content": "Translate the following words into English:\n你係邊個？"},
# ]

# messages_plain = [
#     """
#     <|im_start|> user
#     hi<|im_end|> 
#     <|im_start|> assistant
#     """
# ]



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
# print(os.listdir(FLORES_PATH))

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
            # print(data)
            # print(lang)
            #append data to column
            data_dict[lang] += data
    df = pd.DataFrame(data_dict)
    return df

flores_df = load_flores_dataset()  
# print(flores_df)
flores_dataset = Dataset.from_pandas(flores_df)

    # return flores_train, flores_val, flores_test

# %%
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# model = AutoModelForCausalLM.from_pretrained(
# 	 '/root/autodl-tmp/01ai/Yi-6B-Chat',
# 	 device_map=device,
# 	 torch_dtype=torch.bfloat16,
#      quantization_config=BitsAndBytesConfig(load_in_8bit=True),
# 	 trust_remote_code=True 
# ).eval()
# Since transformers 4.35.0, the GPT-Q/AWQ model can be loaded using AutoModelForCausalLM.

base_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map=device,
    torch_dtype='auto',
)
model = base_model
# model.load_adapter('/root/autodl-tmp/peft_model_pretrained')
model.eval()

# pretrained_model = PeftModel.from_pretrained(base_model, '/root/autodl-tmp/peft_model_pretrained', is_trainable=False)
# model = pretrained_model.merge_and_unload().eval()
# sft_model = PeftModel.from_pretrained(model, '/root/autodl-tmp/peft_model_sft', is_trainable=False)
# model = sft_model.merge_and_unload().eval()
# model.eval()



# def load_flores_dataset():



# %%

YUE_CODE = 'yue_Hant'
LANG_CODES = ['eng_Latn', 'cmn_Hans', 'cmn_Hant']
lang_map = {'eng_Latn': 'English', 'cmn_Hans': 'Mandarin', 'cmn_Hant': 'Mandarin', 'yue_Hant': 'Cantonese'}

def get_prompt(src, tgt, sample):
    src_name = lang_map[src]
    tgt_name = lang_map[tgt]
    system_prompt = f"Translate the given {src_name} words to {tgt_name}."
    user_prompt = sample[src]
    message = F"""<|im_start|> system
    {system_prompt}<|im_end|>
    <|im_start|> user
    {user_prompt}<|im_end|>
    <|im_start|> assistant
    """
    return message

def get_all_prompts(src, tgt, samples):
    conversations = []
    # print(samples)
    for i in range(len(samples)):
        sample = samples.iloc[i]
        conversation = get_prompt(src, tgt, sample)
        conversations.append(conversation)
    return conversations

# train_samples = abc_train_set.shuffle(seed=10).select(range(20))
# test_samples = abc_test_set.shuffle(seed=10).select(range(20))

# for src in lang_names:
#     conversation_list = []
#     conversation_list.append(get_messages(flores_df.iloc[:5], src, cantonese_name))
#     conversation_list.append(get_messages(flores_df.iloc[:5], cantonese_name, src))
#     conversation_list = [item for sublist in conversation_list for item in sublist]
#     print(len(conversation_list))
#     for conversation in conversation_list:
#         print(conversation)
    

     



# %%
prompts = get_all_prompts('cmn_Hant', 'yue_Hant', flores_df.iloc[:3])
# for prompt in prompts:
#     print(prompt)


# %%

def model_output(model, tokenizer, messages, batch_size=16):
    responses = []
    for i in tqdm(range(0, len(messages), batch_size)):
        batch = messages[i:i+batch_size]
        input_ids = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.cuda.amp.autocast():
            output_ids = model.generate(**input_ids.to(device), max_new_tokens=512)
        output_ids = [output_ids[i][input_ids['input_ids'].shape[1]:] for i in range(len(output_ids))]
        response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        responses.append(response)
    responses = [item for sublist in responses for item in sublist]
    return responses

# prompts = get_all_prompts('cmn_Hant', 'yue_Hant', flores_df.iloc[:20])

# print(model_output(model, tokenizer, prompts))

# %% 

#model output test
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


# %%
# model output test
model_output(model, tokenizer, [get_prompt('cmn_Hant', 'yue_Hant', flores_df.iloc[0])])


# %%

for src_lang in LANG_CODES:
    prompts = get_all_prompts(src_lang, YUE_CODE, flores_df)
    output_lines = model_output(model, tokenizer, prompts)
    # save to file
    with open(f'../model_outputs/base_{src_lang}_to_{YUE_CODE}.txt', 'w', encoding='utf-8') as f:
        for line in output_lines:
            f.write(f"{line}\n")
    prompts = get_all_prompts(YUE_CODE, src_lang, flores_df)
    output_lines = model_output(model, tokenizer, prompts)
    # save to file
    with open(f'../model_outputs/base_{YUE_CODE}_to_{src_lang}.txt', 'w', encoding='utf-8') as f:
        for line in output_lines:
            f.write(f"{line}\n")




# %%
