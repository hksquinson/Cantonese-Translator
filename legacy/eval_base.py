# %%
import os
from pathlib import Path

import pandas as pd
import numpy as np
import torch

import bitsandbytes
import accelerate
from datasets import Dataset

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import Trainer, TrainingArguments
from peft import PeftModel

from tqdm import tqdm





# %%
DATA_DIR = Path('../Cantonese-Translator-Data')
REPO_DIRECTORY = r''
FLORES_PATH = DATA_DIR / 'flores+'

#print all files in the flores+ directory
# print(os.listdir(FLORES_PATH))

def load_flores_dataset():
    files = os.listdir(FLORES_PATH)
    column_names = ['cmn_Hans', 'cmn_Hant', 'eng_Latn', 'yue_Hant']
    data_dict = {column: [] for column in column_names}
    for file in sorted(files):
        if not file.startswith("dev"):
            continue
        print(file)
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
print(device)
# %%
tokenizer = AutoTokenizer.from_pretrained("01-ai/Yi-6B-Chat")

base_model = AutoModelForCausalLM.from_pretrained(
    "01-ai/Yi-6B-Chat",
    device_map=device,
    torch_dtype='auto',
)

model = base_model
model.eval()



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

def model_output(model, tokenizer, messages, batch_size=1):
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
    responses = [line.strip().replace('\n', '') for line in responses]
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

# Model response: "早上好，你今天过得怎么样？"
print(response)

# %%

if os.path.exists('model_outputs') == False:
    os.makedirs('model_outputs')

for src_lang in LANG_CODES:
    prompts = get_all_prompts(src_lang, YUE_CODE, flores_df)
    prompts = prompts[:8]
    output_lines = model_output(model, tokenizer, prompts)
    # save to file
    with open(f'model_outputs/base_{src_lang}_to_{YUE_CODE}.txt', 'w', encoding='utf-8') as f:
        for line in output_lines:
            f.write(f"{line}\n")
    prompts = get_all_prompts(YUE_CODE, src_lang, flores_df)
    prompts = prompts[:8]
    output_lines = model_output(model, tokenizer, prompts)
    # save to file
    with open(f'model_outputs/base_{YUE_CODE}_to_{src_lang}.txt', 'w', encoding='utf-8') as f:
        for line in output_lines:
            f.write(f"{line}\n")




# %%
