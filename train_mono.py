import os
import gc
import time

import pandas as pd
import numpy as np

import torch

import bitsandbytes
from modelscope import snapshot_download
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from transformers import BitsAndBytesConfig, AdamW
from trl import SFTTrainer
from peft import get_peft_model, LoraConfig, TaskType

from tqdm import tqdm

gc.collect()
torch.cuda.empty_cache()

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

DATA_DIRECTORY = r'AIST4010-Cantonese-Translator-Data'

def load_cantonese_wiki():
    wiki_lines = []
    def load_cantonese_wiki_file(filename='wiki_00'):
        with open(os.path.join(DATA_DIRECTORY, 'Cantonese-Wiki/text', filename), 'r') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
            lines = [line for line in lines if len(line) > 0]
            lines = [[line[i:i+500] for i in range(0, len(line), 500)] for line in lines]
            lines = ["<|startoftext|>" + line + "<|endoftext|>" for sublist in lines for line in sublist]
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
        lines = ["<|startoftext|>" + line + "<|endoftext|>" for sublist in lines for line in sublist]
        return lines

yue_wiki_lines = load_cantonese_wiki()
openrice_lines = load_openrice_reviews()

print(len(yue_wiki_lines))
print(len(openrice_lines))

mono_dataset = Dataset.from_dict({
    'text': openrice_lines + yue_wiki_lines
})

mono_dataset = mono_dataset.shuffle(seed=42)

print(len(mono_dataset))

#print mean sentence length
sentence_lengths = [len(sentence) for sentence in mono_dataset['text']]
print(np.mean(sentence_lengths))
print(np.sum(sentence_lengths))
print(np.max(sentence_lengths))

model_path=r'01ai/Yi-6B-Chat'
model_dir = snapshot_download('01ai/Yi-6B-Chat', local_files_only=True, cache_dir='', revision='master')

base_tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, padding_side='right', max_length=512, return_tensors='pt')
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, padding_side='right', max_length=512, return_tensors='pt')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# #cuda, mps, cpu
# device = torch.device('cpu')
# if torch.cuda.is_available():
#     device = torch.device('cuda')
# elif torch.backends.mps.is_available():
#     device = torch.device('mps')

print("Is CUDA available: ", torch.cuda.is_available())
# print("Current device: ", torch.cuda.current_device())


# Since transformers 4.35.0, the GPT-Q/AWQ model can be loaded using AutoModelForCausalLM.
model = AutoModelForCausalLM.from_pretrained(
	 '01ai/Yi-6B-Chat',
	 device_map=device,
	 torch_dtype='auto',
    #  quantization_config=BitsAndBytesConfig(load_in_8bit=True),
	#  trust_remote_code=True 
)

# for param in model.parameters():
#   param.requires_grad = False  # freeze the model - train adapters later
#   if param.ndim == 1:
#     # cast the small parameters (e.g. layernorm) to fp32 for stability
#     param.data = param.data.to(torch.float32)

# model.gradient_checkpointing_enable()  # reduce number of stored activations
# model.enable_input_require_grads()

# model.resize_token_embeddings(len(tokenizer))

# class CastOutputToFloat(torch.nn.Sequential):
#   def forward(self, x): return super().forward(x).to(torch.float32)
# model.lm_head = CastOutputToFloat(model.lm_head)

# # Prompt content: "hi"
# messages = [
#     {"role": "user", "content": "hi"}
# ]


# input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')
# output_ids = model.generate(input_ids.to('cuda'))
# response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

# # Model response: "Hello! How can I assist you today?"
# print(response)

mono_dataset = mono_dataset.map(lambda samples: tokenizer(samples['text']), batched=True)

# Prompt content: "hi"
messages = [
    {"role": "user", "content": "hi"}
]


input_ids = base_tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True, return_tensors='pt').to(device)
# with torch.cuda.amp.autocast():
output_ids = model.generate(input_ids, max_length=100)
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
model = get_peft_model(model, 
                            lora_config)

# peft_model.resize_token_embeddings(len(tokenizer))

model.print_trainable_parameters()

# tokenizer = YueTokenizer.from_pretrained(model_path, use_fast=True, padding_side='right', max_length=512, return_tensors='pt')

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

# get time stamp
timestr = time.strftime("%Y%m%d-%H%M%S")


# peft_model.resize_token_embeddings(len(tokenizer))
log_dir = f"tf-logs/{timestr}"

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

training_args = TrainingArguments(
    learning_rate=3e-4,
    num_train_epochs=1.5,
    # max_steps=100,
    logging_steps=100,
    output_dir="peft_model",
    logging_dir=log_dir,
    save_steps=5000,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    max_grad_norm = 0.5,
    report_to="tensorboard"
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

model = model.to(device)

print(model.vocab_size)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=mono_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# with torch.cuda.amp.autocast():
try:
    trainer.train()
except torch.cuda.OutOfMemoryError:
    print("Out of memory error occurred, stopping training...")


trainer.model.save_pretrained("peft_model_pretrained")

#save log history
log_history = pd.DataFrame(trainer.state.log_history)
log_history.to_csv(f"peft_model_pretrained/log_history.csv", index=False)