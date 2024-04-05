import os

import pandas as pd
import numpy as np

import torch


from modelscope import snapshot_download
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from peft import get_peft_model, LoraConfig, TaskType

REPO_DIRECTORY = r'/root/'
ABC_DICT_PATH = r'autodl-tmp/AIST4010-Cantonese-Translator-Data/ABC-Dict/abc_dict.csv'

def load_abc_dataset():
    abc_dict = pd.read_csv(REPO_DIRECTORY + ABC_DICT_PATH)
    abc_dataset = Dataset.from_pandas(abc_dict)
    return abc_dataset

abc_set = load_abc_dataset()
abc_shuffled_set = abc_set.shuffle(seed=42).train_test_split(test_size=0.1)
abc_train_set = abc_shuffled_set['train']
abc_test_set = abc_shuffled_set['test']
for (i, example) in enumerate(abc_train_set):
    print(example)
    if i == 5:
        break

