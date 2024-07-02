import argparse
import datetime
import os
from tqdm import tqdm
from cantonese_translator.translator import CantoneseTranslator
from cantonese_translator.dataset import FloresDataset

def translate_dataset(translator, dataset, lang, current_time, from_cantonese, sample_size=None):
    if from_cantonese:
        src_lang, tgt_lang = 'Cantonese', lang
    else:
        src_lang, tgt_lang = lang, 'Cantonese'

    print(f"Translating from {src_lang} to {tgt_lang}")
    language_data = dataset.get_language_data(src_lang)
    
    if sample_size:
        language_data = language_data[:sample_size]

    results = [translator.translate(src_lang=src_lang, tgt_lang=tgt_lang, text=line) for line in tqdm(language_data)]
    
    if from_cantonese:
        path = f"results/{current_time}/Cantonese_to_{tgt_lang}.txt"
    else:
        path = f"results/{current_time}/{src_lang}_to_Cantonese.txt"
    
    with open(path, 'w') as f:
        for result in results:
            f.write(result + '\n')

def main():
    parser = argparse.ArgumentParser(description="Full dataset translation")
    parser.add_argument("--base_model", required=True, help="Path to the base model")
    parser.add_argument("--adapter", help="Path to the adapter model")
    parser.add_argument("--dataset_type", choices=["flores"], default="flores", help="Type of dataset")
    parser.add_argument("--dataset", required=True, help="Path to the dataset")
    parser.add_argument("--sample_size", type=int, help="Number of samples to translate")
    parser.add_argument("--quant", choices=["4bit", "8bit", "none"], default="8bit", help="Quantization option")

    args = parser.parse_args()

    translator = CantoneseTranslator(
        base_model=args.base_model,
        adapter=args.adapter,
        eval=True, 
        quantization=args.quant if args.quant != "none" else None
    )

    if args.dataset_type == "flores":
        dataset = FloresDataset.load_flores_dataset(args.dataset)
    else:
        raise ValueError(f"Unsupported dataset type: {args.dataset_type}")

    languages = dataset.get_languages()
    languages.remove('Cantonese')

    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    os.makedirs(f"results/{current_time}", exist_ok=True)

    for lang in languages:
        translate_dataset(translator, dataset, lang, current_time, from_cantonese=False, sample_size=args.sample_size)
        translate_dataset(translator, dataset, lang, current_time, from_cantonese=True, sample_size=args.sample_size)

if __name__ == "__main__":
    main()