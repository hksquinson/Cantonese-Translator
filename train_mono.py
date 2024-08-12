#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
import argparse

from pathlib import Path

import torch
import pandas as pd
from transformers import TrainingArguments, DataCollatorForLanguageModeling, Trainer
from peft import get_peft_model, LoraConfig, TaskType, PeftModel

from cantonese_translator import CantoneseTranslator
from cantonese_translator.dataset import MonolingualDataset

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

# In[19]:

def train_mono(train_dataset: MonolingualDataset, translator: CantoneseTranslator, peft_model: PeftModel, max_steps: int = 0):
    # get time stamp
    timestr = time.strftime("%Y%m%d-%H%M%S")

    # peft_model.resize_token_embeddings(len(tokenizer))
    log_dir = Path(f"logs/peft_model_mono_data_{timestr}")
    adapters_dir = Path(f"adapters/peft_model_mono_data_{timestr}")

    train_examples = train_dataset.map(lambda samples: translator.tokenizer(samples['text']), batched=True)

    training_args = TrainingArguments(
        learning_rate=3e-4, 
        num_train_epochs=3,
        # max_steps=200,
        logging_steps=10,
        output_dir=adapters_dir,
        logging_dir=log_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        save_strategy="steps",
        save_steps=0.1,
    )

    max_steps = int(max_steps)

    data_collator = DataCollatorForLanguageModeling(tokenizer=translator.tokenizer, mlm=False)

    if max_steps > 0:
        training_args.max_steps = max_steps

    trainer = Trainer(
        peft_model,
        args=training_args,
        train_dataset=train_examples,
        tokenizer=translator.tokenizer,
        data_collator=data_collator
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

    train_dataset = MonolingualDataset.load_from_files(dataset_paths)
    return train_dataset

def main():
    parser = argparse.ArgumentParser(description="Train on parallel datasets")
    parser.add_argument("--base_model", required=True, help="Path to the base model")
    parser.add_argument("--data_paths", required=True, help="File containing paths of training parallel corpuses")
    parser.add_argument("--max_steps", help="Maximum training steps") 
    parser.add_argument("--quant", choices=["4bit", "8bit", "none"], default="none", help="Quantization option")

    args = parser.parse_args()

    train_dataset = load_data(args.data_paths)

    translator, peft_model = get_models(args.base_model, args.quant)

    test_message = train_dataset["text"][0]
    test_result = translator.translate(
        src_lang="Cantonese",
        tgt_lang="English",
        text=test_message
    )

    print(test_result)

    max_steps = args.max_steps if args.max_steps else 0

    train_mono(train_dataset, translator, peft_model, max_steps)

    print("Training complete")





if __name__ == "__main__":
    main()