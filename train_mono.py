import os
import gc

import pandas as pd
import numpy as np

import torch

import bitsandbytes
from modelscope import snapshot_download
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from transformers import BitsAndBytesConfig
from trl import SFTTrainer
from peft import get_peft_model, LoraConfig, TaskType

from tqdm import tqdm

from custom_tokenizers import YueTokenizer

gc.collect()
torch.cuda.empty_cache()
torch.cuda.is_available()

DATA_DIRECTORY = r'/root/autodl-tmp/AIST4010-Cantonese-Translator-Data/'

def load_cantonese_wiki():
    wiki_lines = []
    def load_cantonese_wiki_file(filename='wiki_00'):
        with open(os.path.join(DATA_DIRECTORY, 'Cantonese-Wiki/text', filename), 'r') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
            lines = [line for line in lines if len(line) > 0]
            lines = [[line[i:i+500] for i in range(0, len(line), 500)] for line in lines]
            lines = [line for sublist in lines for line in sublist]
            return lines
        
    for file in os.listdir(os.path.join(DATA_DIRECTORY, 'Cantonese-Wiki/text')):
        curr_lines = load_cantonese_wiki_file(file)
        wiki_lines.extend(curr_lines)
    
    return wiki_lines

def load_openrice_reviews():
    with open(os.path.join(DATA_DIRECTORY, 'openrice/openrice.txt'), 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        lines = [line for line in lines if len(line) > 0]
        lines = [[line[i:i+500] for i in range(0, len(line), 500)] for line in lines]
        lines = [line for sublist in lines for line in sublist]
        return lines

yue_wiki_lines = load_cantonese_wiki()
openrice_lines = load_openrice_reviews()

print(len(yue_wiki_lines))
print(len(openrice_lines))

mono_dataset = Dataset.from_dict({
    'text': yue_wiki_lines + openrice_lines
})

print(len(mono_dataset))

#print mean sentence length
sentence_lengths = [len(sentence) for sentence in mono_dataset['text']]
print(np.mean(sentence_lengths))
print(np.sum(sentence_lengths))
print(np.max(sentence_lengths))

model_path=r'/root/autodl-tmp/01ai/Yi-6B-Chat'
model_dir = snapshot_download('01ai/Yi-6B-Chat', cache_dir='/root/autodl-tmp', revision='master')

base_tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, padding_side='right', max_length=512, return_tensors='pt')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# Since transformers 4.35.0, the GPT-Q/AWQ model can be loaded using AutoModelForCausalLM.
base_model = AutoModelForCausalLM.from_pretrained(
	 '/root/autodl-tmp/01ai/Yi-6B-Chat',
	 device_map=device,
	 torch_dtype=torch.bfloat16,
    #  quantization_config=BitsAndBytesConfig(load_in_8bit=True),
	#  trust_remote_code=True 
)


# # Prompt content: "hi"
# messages = [
#     {"role": "user", "content": "hi"}
# ]


# input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')
# output_ids = model.generate(input_ids.to('cuda'))
# response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

# # Model response: "Hello! How can I assist you today?"
# print(response)

# Prompt content: "hi"
messages = [
    {"role": "user", "content": "hi"}
]


input_ids = base_tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')
output_ids = base_model.generate(input_ids.to('cuda'))
response = base_tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

# Model response: "Hello! How can I assist you today?"
print(response)

lora_config = LoraConfig(
    r=32, # Rank
    lora_alpha=32,
    target_modules = ["k_proj", "q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
peft_model = get_peft_model(base_model, 
                            lora_config)
peft_model = peft_model.to(device)

peft_model.print_trainable_parameters()

tokenizer = YueTokenizer.from_pretrained(model_path, use_fast=True, padding_side='right', max_length=512, return_tensors='pt')

print(len(tokenizer.get_vocab()))

print(tokenizer.tokenize("嗌 呃 畀 啲 嘢 噃"))
print(base_tokenizer.tokenize("嗌 呃 畀 啲 嘢 噃"))
print(tokenizer.tokenize("嗌呃畀啲嘢噃"))
print(base_tokenizer.tokenize("嗌呃畀啲嘢噃"))
print(tokenizer.tokenize("中文字"))
print(base_tokenizer.tokenize("中文字"))
print(tokenizer("Good morning")['input_ids'])
print(base_tokenizer("Good morning")['input_ids'])

def formatting_prompts_func(examples):
    output_texts = []
    for i, example in enumerate(examples['text']):
        if example.strip() == '' or len(example) <= 1:
            continue
        example_len = len(example)
        random_split = np.random.randint(0.3*example_len, 0.7*example_len)
        random_split = max(min(random_split, example_len-1), 1)
        split1 = example[:random_split]
        split2 = example[random_split:]
        text = f"""<|im_start|> user
        Complete the following text: {split1} <|im_end|> 
        <|im_start|> assistant
        {split2} <|im_end|>"""
        output_texts.append(text)
    return output_texts

prompts = formatting_prompts_func(mono_dataset[:10])
for prompt in prompts:
    print(prompt)

peft_model.resize_token_embeddings(len(tokenizer))

training_args = TrainingArguments(
    learning_rate=1e-3, # Higher learning rate than full fine-tuning.
    num_train_epochs=3,
    logging_steps=100,
    output_dir="/root/peft_model",
    per_device_train_batch_size=1
)

# data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = SFTTrainer(
    model=peft_model,
    args=training_args,
    train_dataset=mono_dataset,
    formatting_func=formatting_prompts_func,
    tokenizer=tokenizer,
    # data_collator=data_collator,
)
trainer.train()

trainer.model.save_pretrained("/root/peft_model")

print(pd.DataFrame(trainer.state.log_history))