#%%
import datetime
import os

from tqdm import tqdm

from cantonese_translator.translator import CantoneseTranslator, ModelConfig, TranslationInput
from cantonese_translator.dataset import FloresDataset, LANG_MAP

#%%
def main():
    translator = CantoneseTranslator(
        base_model="models/Yi-6B-Chat",
        adapter="adapters/peft_model_sft",
        eval=True, 
        quantization="8bit"
    )
    print(translator.device)
    print(translator.translate_to_cantonese(src_lang="English", text="Hello, how are you?"))
    print(translator.translate_from_cantonese(tgt_lang="English", text="食咗飯未？​"))

    flores_path = "data/flores+"
    flores_dataset = FloresDataset.load_flores_dataset(flores_path)

    languages = flores_dataset.get_languages()
    languages.remove('Cantonese')

    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    os.makedirs(f"results/{current_time}", exist_ok=True)


    # translate from English/Mandarin/Taiwan Mandarin to Cantonese
    for src_lang in languages:
        print(f"Translating from {src_lang} to Cantonese")
        language_data = flores_dataset.get_language_data(src_lang)
        language_data = language_data[:5]
        results = [translator.translate_to_cantonese(src_lang=src_lang, text=line) for line in tqdm(language_data)]
        path = f"results/{current_time}/{src_lang}_to_Cantonese.txt"
        with open(path, 'w') as f:
            for result in results:
                f.write(result + '\n')

    
    # translate from Cantonese to English/Mandarin/Taiwan Mandarin
    for tgt_lang in languages:
        print(f"Translating from {src_lang} to Cantonese")
        language_data = flores_dataset.get_language_data('Cantonese')
        language_data = language_data[:5]
        results = [translator.translate_from_cantonese(tgt_lang=tgt_lang, text=line) for line in tqdm(language_data)]
        path = f"results/{current_time}/Cantonese_to_{tgt_lang}.txt"
        with open(path, 'w') as f:
            for result in results:
                f.write(result + '\n')
        # for line in language_data:
        #     print(translator.translate_from_cantonese(tgt_lang=tgt_lang, text=line))


if __name__ == "__main__":
    main()