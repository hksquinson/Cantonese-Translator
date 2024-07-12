import argparse
import datetime
from cantonese_translator.translator import CantoneseTranslator
from cantonese_translator.dataset import ParallelDataset, FloresDataset

# def translate_dataset_column(translator, dataset, lang, from_cantonese, sample_size=None):
#     if from_cantonese:
#         src_lang, tgt_lang = 'Cantonese', lang 
#     else:
#         src_lang, tgt_lang = lang, 'Cantonese'

#     print(f"Translating from {src_lang} to {tgt_lang}")
#     language_data = dataset.get_language_data(src_lang)
    
#     if sample_size:
#         language_data = language_data[:sample_size]

#     results = translator.batch_translate(src_lang=src_lang, tgt_lang=tgt_lang, batch_text=language_data, batch_size=8)
    
#     current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

#     src_lang = src_lang.replace(' ', '_')
    
#     if from_cantonese:
#         path = f"results/{current_time}_Cantonese_to_{tgt_lang}.txt"
#     else:
#         path = f"results/{current_time}_{src_lang}_to_Cantonese.txt"

#     path = path.replace(' ', '_')
    
#     with open(path, 'w') as f:
#         for result in results:
#             # remove line breaks
#             result = result.replace('\n', ' ')
#             f.write(result + '\n')

def translate_dataset_file(translator, file, lang, from_cantonese, sample_size=None, batch_size=8):
    if from_cantonese:
        src_lang, tgt_lang = 'Cantonese', lang 
    else:
        src_lang, tgt_lang = lang, 'Cantonese'

    print(f"Translating from {src_lang} to {tgt_lang}")
    
    with open(file, 'r') as f:
        language_data = f.readlines()
    
    if sample_size:
        language_data = language_data[:sample_size]

    results = translator.batch_translate(src_lang=src_lang, tgt_lang=tgt_lang, batch_text=language_data, batch_size=batch_size)
    
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    src_lang = src_lang.replace(' ', '_')
    
    if from_cantonese:
        path = f"results/{current_time}_Cantonese_to_{tgt_lang}.txt"
    else:
        path = f"results/{current_time}_{src_lang}_to_Cantonese.txt"

    path = path.replace(' ', '_')
    
    with open(path, 'w') as f:
        for result in results:
            # remove line breaks
            result = result.replace('\n', ' ')
            f.write(result + '\n')

def main():
    parser = argparse.ArgumentParser(description="Dataset column translation")
    parser.add_argument("--base_model", required=True, help="Path to the base model")
    parser.add_argument("--adapter", help="Path to the adapter model")
    parser.add_argument("--file", help="Path to the text file")
    parser.add_argument("--lang", choices=["English", "Simplified Chinese", "Traditional Chinese"], help="Source or target language")
    parser.add_argument("--sample_size", type=int, help="Number of samples to translate")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--from_cantonese", action="store_true", help="Translate from Cantonese")
    parser.add_argument("--quant", choices=["4bit", "8bit", "none"], default="none", help="Quantization option")

    args = parser.parse_args()

    translator = CantoneseTranslator(
        base_model=args.base_model,
        adapter=args.adapter,
        eval=True, 
        quantization=args.quant
    )

    translate_dataset_file(translator, args.file, args.lang, args.from_cantonese, sample_size=args.sample_size, batch_size=args.batch_size)

if __name__ == "__main__":
    main()