#%%
import os
import gc
import time
from pathlib import Path

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

from cantonese_translator import CantoneseTranslator
from cantonese_translator.dataset import MonolingualDataset

gc.collect()
torch.cuda.empty_cache()

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

DATA_DIRECTORY = Path('data')

# def load_cantonese_wiki():
#     wiki_lines = []
#     def load_cantonese_wiki_file(filename='wiki_00'):
#         cantonese_wiki_file = DATA_DIRECTORY / 'Cantonese-Wiki/text' / filename
#         with open(cantonese_wiki_file, 'r', encoding='utf-8') as f:
#             lines = f.readlines()
#             lines = [line.strip() for line in lines]
#             lines = [line for line in lines if len(line) > 0]
#             # lines = [[line[i:i+500] for i in range(0, len(line), 500)] for line in lines]
#             lines = ["<|startoftext|>" + line + "<|endoftext|>" for line in lines]
#             return lines
        
#     for file in os.listdir(os.path.join(DATA_DIRECTORY, 'Cantonese-Wiki/text')):
#         curr_lines = load_cantonese_wiki_file(file)
#         wiki_lines.extend(curr_lines)
    
#     return wiki_lines

# def load_openrice_reviews():
#     openrice_file = DATA_DIRECTORY / 'openrice/openrice.txt'
#     with open(openrice_file, 'r', encoding='utf-8') as f:
#         lines = f.readlines()
#         lines = [line.strip() for line in lines]
#         lines = [line for line in lines if len(line) > 0]
#         lines = ["<|startoftext|>" + line + "<|endoftext|>" for line in lines]
#         return lines

# yue_wiki_lines = load_cantonese_wiki()
# openrice_lines = load_openrice_reviews()

# print(len(yue_wiki_lines))
# print(len(openrice_lines))



mono_dataset = MonolingualDataset.load_from_files([DATA_DIRECTORY / 'Cantonese-Wiki/text', DATA_DIRECTORY / 'openrice/openrice.txt'])

# mono_dataset = mono_dataset.shuffle(seed=42)

# print(len(mono_dataset))

# #print mean sentence length
# sentence_lengths = [len(sentence) for sentence in mono_dataset['text']]
# print(np.mean(sentence_lengths))
# print(np.sum(sentence_lengths))
# print(np.max(sentence_lengths))

#%%

def get_models(base_model: str, quant: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if quant not in ["4bit", "8bit", "none"]:
        raise ValueError("Invalid quantization option")

    translator = CantoneseTranslator(
        base_model=base_model,
        adapter=None,
        eval=False,
        quantization=quant
    )

    lora_config = LoraConfig(
        r=32, # Rank
        lora_alpha=32,
        target_modules = ["k_proj", "q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    peft_model = get_peft_model(translator.model, 
                                lora_config)

    peft_model.print_trainable_parameters()

    return translator, peft_model

translator, peft_model = get_models("models/Yi-6B-Chat", "8bit")

#%%

# # Prompt content: "hi"
# messages = [
#     {"role": "user", "content": "hi"}
# ]


# input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')
# output_ids = model.generate(input_ids.to('cuda'))
# response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

# # Model response: "Hello! How can I assist you today?"
# print(response)

# mono_dataset = mono_dataset.map(lambda samples: tokenizer(samples['text']), batched=True)

# # Prompt content: "hi"
# messages = [
#     {"role": "user", "content": "hi"}
# ]

translator.translate(src_lang="English", tgt_lang="Cantonese", text="hi")

#%%

# get time stamp
timestr = time.strftime("%Y%m%d-%H%M%S")

log_dir = Path(f"logs/peft_model_pretrained_{timestr}")
adapters_dir = Path(f"adapters/peft_model_pretrained_{timestr}")

training_args = TrainingArguments(
    learning_rate=3e-4,
    num_train_epochs=1.5,
    max_steps=200,
    logging_steps=100,
    output_dir="peft_model",
    logging_dir=log_dir,
    save_steps=5000,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    max_grad_norm = 0.5,
    report_to="tensorboard"
)

data_collator = DataCollatorForLanguageModeling(tokenizer=translator.tokenizer, mlm=False)



trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=mono_dataset,
    tokenizer=translator.tokenizer,
    data_collator=data_collator
)

try:
    trainer.train()
except torch.cuda.OutOfMemoryError:
    print("Out of memory error occurred, stopping training...")


trainer.model.save_pretrained("peft_model_pretrained")

#save log history
log_history = pd.DataFrame(trainer.state.log_history)
log_history.to_csv(f"results/peft_model_pretrained/log_history.csv", index=False)
# %%
