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
model_path=r'01ai/Yi-6B-Chat'
model_dir = snapshot_download('01ai/Yi-6B-Chat', cache_dir='01ai/Yi-6B-Chat', revision='master')

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
tokenizer = AutoTokenizer.from_pretrained(model_dir, cache_dir='01ai/Yi-6B-Chat', local_files_only=True, use_fast=True, padding_side='right', max_length=512, return_tensors='pt')
base_tokenizer = AutoTokenizer.from_pretrained(model_dir, cache_dir='01ai/Yi-6B-Chat', local_files_only=True, use_fast=True, padding_side='right', max_length=512, return_tensors='pt')

# %%
# model = PeftModel.from_pretrained(
#     model,
#    '/root/peft_model',
#     is_trainable=False
# )

# %%
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
base_model = AutoModelForCausalLM.from_pretrained(
	 model_dir,
	 device_map=device,
	 torch_dtype=torch.bfloat16,
     quantization_config=BitsAndBytesConfig(load_in_8bit=True),
	 trust_remote_code=True 
).eval()

model = AutoModelForCausalLM.from_pretrained(
	 model_dir,
	 device_map=device,
	 torch_dtype=torch.bfloat16,
	 quantization_config=BitsAndBytesConfig(load_in_8bit=True),
	 trust_remote_code=True 
)

pretrained_model = PeftModel.from_pretrained(base_model, '/root/autodl-tmp/peft_model_pretrained', is_trainable=False)
model = pretrained_model.merge_and_unload()
sft_model = PeftModel.from_pretrained(model, '/root/autodl-tmp/peft_model_sft', is_trainable=False)
model = sft_model.merge_and_unload()
model.eval()

# %%
REPO_DIRECTORY = r''
ABC_DICT_PATH = r'AIST4010-Cantonese-Translator-Data/ABC-Dict/abc_dict.csv'

def load_abc_dataset():
    abc_dict = pd.read_csv(REPO_DIRECTORY + ABC_DICT_PATH)
    abc_dataset = Dataset.from_pandas(abc_dict)
    return abc_dataset

abc_set = load_abc_dataset()
abc_shuffled_set = abc_set.train_test_split(seed=42, test_size=0.1)
abc_train_set = abc_shuffled_set['train']
abc_test_set = abc_shuffled_set['test']

# %%

def get_messages(sample):
    lang_map = {'English': 'en', 'Cantonese': 'yue'}
    def get_prompt(src, tgt, sample):
        system_prompt = f"Translate the given {src} words to {tgt}."
        user_prompt = sample[lang_map[src]]
        return system_prompt, user_prompt
    system1, user1 = get_prompt('English', 'Cantonese', sample)
    system2, user2 = get_prompt('Cantonese', 'English', sample)
    return [[
        {
            "role": "system",
            "content": system1
        },
        {
            "role": "user",
            "content": user1
        }],
        [
        {
            "role": "system",
            "content": system2
        },
        {
            "role": "user",
            "content": user2
        }]
    ]
    

train_samples = abc_train_set.shuffle(seed=10).select(range(20))
test_samples = abc_test_set.shuffle(seed=10).select(range(20))

get_messages(train_samples[0])

# %%


# %%
def model_output(model, tokenizer, messages, name=None):
        for prompt in messages:
                input_ids = tokenizer.apply_chat_template(conversation=prompt, tokenize=True, add_generation_prompt=True, return_tensors='pt')
                with torch.cuda.amp.autocast():
                        output_ids = model.generate(input_ids.to('cuda'), max_new_tokens=200)
                response = base_tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True, max_length=100)
                # response = tokenizer.decode(output_ids[0], skip_special_tokens=False, max_length=100)
                # print(output_ids)
                print(f"{name}:\n{response}\n")


# print([model_output(base_model, base_tokenizer, get_messages(train_samples[0]), 'Base model'),])

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
# model_output(model, tokenizer, get_messages(train_samples[0]), 'Fine-tuned model')

# %%
# compare_outputs(train_samples)


# # %%
# compare_outputs(test_samples)

# # %%
# messages = [
#     {"role": "user", "content": "Translate the following words into English:\n乜嘢都係波士決定嘅，打工仔啲人淨係得個知字。\n"},
# ]

# # get 5 random samples from train and test dataset
# train_sample = abc_train_set.shuffle(seed=42).select(range(5))
# test_sample = abc_test_set.shuffle(seed=42).select(range(5))

# en_train_messages = {get_translate_prompt('Cantonese', sentence) for sentence in train_sample['en']}
# en_test_messages = {get_translate_prompt('Cantonese', sentence) for sentence in test_sample['en']}
# yue_train_messages = {get_translate_prompt('English', sentence) for sentence in train_sample['yue']}
# yue_test_messages = {get_translate_prompt('English', sentence) for sentence in test_sample['yue']}

# for messages in [en_train_messages, en_test_messages, yue_train_messages, yue_test_messages]:
#     for message in messages:
#         print(message)


# %%


#change the user block to test different inputs
messages = [
    {"role": "system", "content": "Translate the given English words into Cantonese."},
    {"role": "user", "content": "Good morning! How are you？"},
]


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


