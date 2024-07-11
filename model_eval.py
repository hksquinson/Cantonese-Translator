import argparse
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm
from torch import cuda
from sacrebleu.metrics import BLEU
from comet import download_model, load_from_checkpoint
from evaluate import load

def get_bleu_score(src_lang, tgt_lang, truth_dir, model_dir):
    bleu = BLEU(tokenize='zh')
    truth_file = Path(truth_dir, f'{tgt_lang}.txt')
    pred_file = Path(model_dir, f'{src_lang}_to_{tgt_lang}.txt')
    
    with open(truth_file, 'r') as file:
        truth = [line.strip() for line in file]
    
    with open(pred_file, 'r') as file:
        pred = [line.strip() for line in file]
    
    scores = bleu.corpus_score(pred, [truth])
    return scores

def get_bertscore(src_lang, tgt_lang, truth_dir, model_dir):
    bertscore_metric_en = load('bertscore', lang='en')
    bertscore_metric_zh = load('bertscore', lang='zh')

    truth_file = Path(truth_dir, f'{tgt_lang}.txt')
    pred_file = Path(model_dir, f'{src_lang}_to_{tgt_lang}.txt')

    with open(truth_file, 'r') as file:
        truth_lines = [line.strip() for line in file]
    with open(pred_file, 'r') as file:
        pred_lines = [line.strip() for line in file]

    bertscore_metric = bertscore_metric_en if tgt_lang == 'English' else bertscore_metric_zh
    lang = 'en' if tgt_lang == 'English' else 'zh'
    
    precisions, recalls, f1s = [], [], []
    for i in tqdm(range(0, len(pred_lines), 10)):
        results = bertscore_metric.compute(predictions=pred_lines[i:i+10], references=truth_lines[i:i+10], lang=lang)
        precisions.extend(results['precision'])
        recalls.extend(results['recall'])
        f1s.extend(results['f1'])

    return np.mean(precisions), np.mean(recalls), np.mean(f1s)

def get_comet_score(src_lang, tgt_lang, truth_dir, model_dir):
    comet_model_path = download_model("Unbabel/wmt22-comet-da")
    comet_model = load_from_checkpoint(comet_model_path)

    src_file = Path(truth_dir, f'{src_lang}.txt')
    truth_file = Path(truth_dir, f'{tgt_lang}.txt')
    pred_file = Path(model_dir, f'{src_lang}_to_{tgt_lang}.txt')

    with open(src_file, 'r') as file:
        src_lines = [line.strip() for line in file]
    with open(truth_file, 'r') as file:
        truth_lines = [line.strip() for line in file]
    with open(pred_file, 'r') as file:
        pred_lines = [line.strip() for line in file]

    data = [
        {"src": src, "mt": mt, "ref": ref} for src, mt, ref in zip(src_lines, pred_lines, truth_lines)
    ]
    comet_output = comet_model.predict(data, batch_size=8, gpus=1)
    return comet_output.system_score

def get_record(src, tgt, truth_dir, model_dir):
    if not Path(model_dir, f"{src}_to_{tgt}.txt").exists():
        return None
    print(f"Evaluating {model_dir} on {src} to {tgt}")
    bleu_scores = get_bleu_score(src, tgt, truth_dir, model_dir)
    bert_scores = get_bertscore(src, tgt, truth_dir, model_dir)
    comet_scores = get_comet_score(src, tgt, truth_dir, model_dir)
    
    return {
        "model": model_dir,
        "src_lang": src,
        "tgt_lang": tgt,
        "bleu_score": round(bleu_scores.score, 2),
        "bert_precision": round(bert_scores[0], 2),
        "bert_recall": round(bert_scores[1], 2),
        "bert_f1": round(bert_scores[2], 2),
        "comet_score": comet_scores,
    }

def main():
    parser = argparse.ArgumentParser(description='Evaluate translation model')
    parser.add_argument('--truth_dir', type=str, required=True, help='Directory containing ground truth files')
    parser.add_argument('--model_output_dir', type=str, required=True, help='Directory containing model output files')
    parser.add_argument('--src_lang', type=str, required=True, help='Source language')
    parser.add_argument('--tgt_lang', type=str, required=True, help='Target language')

    args = parser.parse_args()

    if ' ' in args.src_lang:
        args.src_lang = args.src_lang.replace(' ', '_')
    if ' ' in args.tgt_lang:
        args.tgt_lang = args.tgt_lang.replace(' ', '_')

    record = get_record(args.src_lang, args.tgt_lang, args.truth_dir, args.model_output_dir)

    if record:
        print("Evaluation Results:")
        for key, value in record.items():
            print(f"{key}: {value}")
    else:
        print(f"No results found for {args.src_lang} to {args.tgt_lang} in {args.model_output_dir}")

if __name__ == "__main__":
    main()