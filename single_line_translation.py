import argparse
from cantonese_translator.translator import CantoneseTranslator

def main():
    parser = argparse.ArgumentParser(description="Single line translation")
    parser.add_argument("--base_model", required=True, help="Path to the base model")
    parser.add_argument("--adapter", help="Path to the adapter model")
    parser.add_argument("--lang", help="Source or target language")
    parser.add_argument("--text", help="Text to translate")
    parser.add_argument("--from_cantonese", action="store_true", help="Translate from Cantonese")
    parser.add_argument("--quant", choices=["4bit", "8bit", "none"], default="none", help="Quantization option")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")

    args = parser.parse_args()

    print(args.adapter)

    translator = CantoneseTranslator(
        base_model=args.base_model,
        adapter=args.adapter,
        eval=True, 
        quantization=args.quant
    )

    if args.interactive:
        while True:
            print(f"Enter text to translate to {args.lang if args.from_cantonese else 'Cantonese'} (or 'q' to quit): ")
            text = input()
            if text.lower() == "q":
                break
            if args.from_cantonese:
                result = translator.translate(tgt_lang=args.lang, text=text)
            else:
                result = translator.translate(src_lang=args.lang, text=text)
            print(f"Translated Text:")
            print(result)
    elif args.text:
        text = args.text
        if args.from_cantonese:
            result = translator.translate(tgt_lang=args.lang, text=text)
        else:
            result = translator.translate(src_lang=args.lang, text=text)
        print(result)
    else:
        print("Error: In non-interactive mode, --text argument is required.")

if __name__ == "__main__":
    main()