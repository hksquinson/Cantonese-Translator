#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
import argparse

from pathlib import Path

import torch
import pandas as pd
from transformers import TrainingArguments
from trl import SFTTrainer
from peft import get_peft_model, LoraConfig, TaskType, PeftModel

from cantonese_translator import CantoneseTranslator
from cantonese_translator.dataset import ParallelDataset

# In[2]:


# dataset_paths = [
#     "data/ABC-Dict/abc_dict.csv",
#     "data/kaifangcidian/kaifangcidian.csv"
# ]

# train_dataset = ParallelDataset.load_from_csv(dataset_paths)

# for i in range(20):
#     print(train_dataset[i])

# print(len(train_dataset))

# In[3]:

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

def train_sft(train_dataset: ParallelDataset, translator: CantoneseTranslator, peft_model: PeftModel, max_steps: int = 0):
    # get time stamp
    timestr = time.strftime("%Y%m%d-%H%M%S")

    # peft_model.resize_token_embeddings(len(tokenizer))
    log_dir = Path(f"logs/peft_model_sft_only_{timestr}")
    adapters_dir = Path(f"adapters/peft_model_sft_only_{timestr}")

    training_args = TrainingArguments(
        learning_rate=3e-4, 
        num_train_epochs=3,
        # max_steps=200,
        logging_steps=10,
        output_dir=adapters_dir,
        logging_dir=log_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        save_strategy="steps",
        save_steps=0.1,
    )

    if max_steps > 0:
        training_args.max_steps = max_steps

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
    trainer.model.save_pretrained(adapters_dir)
    pd.DataFrame(trainer.state.log_history).to_csv(adapters_dir / Path("trainer_log.csv"), index=False)

def load_data(data_paths):
    dataset_paths = []

    with open(data_paths, 'r') as f:
        dataset_paths = f.readlines()
        dataset_paths = [line.strip() for line in dataset_paths]

    if len(dataset_paths) == 0:
        raise ValueError("No dataset paths found in the file")

    train_dataset = ParallelDataset.load_from_csv(dataset_paths)
    return train_dataset

# %%

def main():
    parser = argparse.ArgumentParser(description="Train on parallel datasets")
    parser.add_argument("--base_model", required=True, help="Path to the base model")
    parser.add_argument("--data_paths", required=True, help="File containing paths of training parallel corpuses")
    parser.add_argument("--max_steps", help="Maximum training steps") 
    parser.add_argument("--quant", choices=["4bit", "8bit", "none"], default="none", help="Quantization option")

    args = parser.parse_args()

    train_dataset = load_data(args.data_paths)

    translator, peft_model = get_models(args.base_model, args.quant)

    test_message = train_dataset["Cantonese"][0]
    test_result = translator.translate(
        src_lang="Cantonese",
        tgt_lang="English",
        text=test_message
    )

    print(test_result)

    max_steps = args.max_steps if args.max_steps else 0

    train_sft(train_dataset, translator, peft_model, max_steps)

    print("Training complete")





if __name__ == "__main__":
    main()