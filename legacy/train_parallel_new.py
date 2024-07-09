#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import time

import pandas as pd
import numpy as np
from pathlib import Path

import torch
import nltk.translate.bleu_score as bleu

from datasets import Dataset, load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

from cantonese_translator import CantoneseTranslator
from cantonese_translator.dataset import ParallelDataset



# In[2]:

dataset_paths = [
    "data/ABC-Dict/abc_dict.csv",
    "data/kaifangcidian/kaifangcidian.csv"
]

train_dataset = ParallelDataset.load_from_csv(dataset_paths)

# for i in range(20):
#     print(train_dataset[i])

# print(len(train_dataset))

# In[3]:

device = "cuda" if torch.cuda.is_available() else "cpu"

translator = CantoneseTranslator(
    base_model="01-ai/Yi-6B-Chat",
    adapter=None,
    eval=False,
    quantization="8bit"
)


# %%
test_message = "Good morning, how are you?"
test_result = translator.translate(
    src_lang="English",
    tgt_lang="Cantonese",
    text=test_message
)

print(test_result)

#%%

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

# In[9]:

def formatting_prompts_func(examples):
    output_texts = []
    languages = list(examples.keys())
    # print(examples)
    for i in range(len(examples[languages[0]])):
        record = {lang: examples[lang][i] for lang in languages}
        for src_lang in languages:
            for tgt_lang in languages:
                if src_lang == tgt_lang:
                    continue
                src_text = record[src_lang]
                tgt_text = record[tgt_lang]
                if src_text is not None and tgt_text is not None:
                    system_prompt = f"<|im_start|>system Translate the given {src_lang} words into {tgt_lang}. <|im_end|>"
                    prompt = f"<|im_start|>user {src_text} <|im_end|>"
                    response = f"<|im_start|>assistant {tgt_text} <|im_end|>"
                    output_texts.append(system_prompt + prompt + response)
    # print(len(output_texts))
    return output_texts


# %% 

# # test formatting_prompts_func

# print(train_dataset[:10])

# prompts = formatting_prompts_func(train_dataset[:10])
# for prompt in prompts:
#     print(prompt)


# In[19]:

# get time stamp
timestr = time.strftime("%Y%m%d-%H%M%S")

# peft_model.resize_token_embeddings(len(tokenizer))
log_dir = Path(f"logs/peft_model_sft_only_{timestr}")
adapters_dir = Path(f"adapters/peft_model_sft_only_{timestr}")

training_args = TrainingArguments(
    learning_rate=3e-4, 
    num_train_epochs=3,
    max_steps=200,
    logging_steps=10,
    output_dir=adapters_dir,
    logging_dir=log_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    save_strategy="steps",
    save_steps=0.2,
)

trainer = SFTTrainer(
    peft_model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=translator.tokenizer,
    formatting_func=formatting_prompts_func,
    max_seq_length=512
    # data_collator=collator,
)

trainer.train()


# In[ ]:

#save trainer log history
trainer.model.save_pretrained(adapters_dir)

pd.DataFrame(trainer.state.log_history).to_csv(adapters_dir / Path("trainer_log.csv"), index=False)

# %%