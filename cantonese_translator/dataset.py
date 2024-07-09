import os
import pandas as pd
from pathlib import Path

from datasets import Dataset, load_dataset, concatenate_datasets

LANG_MAP = {
    'cmn_Hans': 'Simplified Chinese',
    'cmn_Hant': 'Traditional Chinese',
    'eng_Latn': 'English',
    'yue_Hant': 'Cantonese'
}

class ParallelDataset(Dataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def load_from_csv(cls, dataset_paths):
        dataset_list = [load_dataset("csv", data_files=dataset_path) for dataset_path in dataset_paths]
        merged_columns = set()
        for dataset in dataset_list:
            merged_columns |= set(dataset['train'].column_names)

        for column_name in merged_columns:
            for dataset in dataset_list:
                if column_name not in dataset['train'].column_names:
                    dataset['train'] = dataset['train'].add_column(column_name, [None] * len(dataset['train']))

        dataset_list = [dataset['train'] for dataset in dataset_list]
        combined_dataset = concatenate_datasets(dataset_list).shuffle(seed=42)

        df = pd.DataFrame(combined_dataset)
        return cls.from_pandas(df)

    def get_name(self):
        return self.__class__.__name__
    
    def get_languages(self):
        return self.column_names
    
    def get_language_data(self, language: str):
        return self[language]
    
    def get_parallel_data(self, start_index: int, end_index: int = None):
        if end_index is None:
            end_index = start_index + 1
        return self[start_index:end_index]
    
    def get_prompt(self, src_lang, tgt_lang, index):
        src_name = src_lang
        tgt_name = tgt_lang
        system_prompt = f"Translate the given {src_name} words to {tgt_name}."
        user_prompt = self[src_lang][index]
        return f"<|im_start|> system {system_prompt} <|im_end|> <|im_start|> user {user_prompt} <|im_end|>"
    
    def get_prompts(self, src_lang, tgt_lang, indices):
        return [self.get_prompt(src_lang, tgt_lang, index) for index in indices]


class FloresDataset(ParallelDataset):

    @classmethod
    def load_flores_dataset(cls, path: str) -> 'FloresDataset':  
        files = os.listdir(path)
        files.sort()
        # column_names = ['cmn_Hans', 'cmn_Hant', 'eng_Latn', 'yue_Hant']
        column_names = LANG_MAP
        data_dict = {column: [] for column in column_names.values()}
        for file in files:
            if not file.startswith("dev"):
                continue
            data = []
            with open(os.path.join(path, file), 'r') as f:
                data = f.readlines()
                data = [line.strip() for line in data]
                lang = column_names[file.split('.')[1]]
                # print(data)
                # print(lang)
                #append data to column
                data_dict[lang] += data
        
        return cls.from_dict(data_dict)
    

if __name__ == "__main__":
    flores_path = Path('data/flores+')
    flores_dataset = FloresDataset.load_flores_dataset(flores_path)
    print(flores_dataset.get_languages())
    print(flores_dataset.get_language_data('English')[:10])
    print(flores_dataset.get_parallel_data(1, 5))
    print(flores_dataset.get_prompt('English', 'Cantonese', 0))
    print(flores_dataset.get_prompts('English', 'Cantonese', [0, 1, 2]))