#%%
import time

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

from sacrebleu.metrics import BLEU, CHRF, TER


from tqdm import tqdm
from torch import cuda
from comet import download_model, load_from_checkpoint

from evaluate import load

#%%

#change directories here to evaluate different models and datasets
truth_dir = Path('data/flores+/evaluation/ground_truth')
results_dir = Path('results')
output_dirs = [
    'results/base_2024-07-09_21-59-54',
    'results/peft_2024-07-11_09-41-09'
]

#%%
cantonese = 'Cantonese'
langauges = ['Simplified_Chinese', 'Traditional_Chinese', 'English']


#%%
        
bleu = BLEU(tokenize='zh')
        
def get_bleu_score(src_lang, tgt_lang, model_dir):
    truth_file = Path(truth_dir, f'{tgt_lang}.txt')
    pred_file = Path(model_dir, f'{src_lang}_to_{tgt_lang}.txt')
    
    truth = []
    with open(truth_file, 'r') as file:
        truth = file.readlines()
        truth = [line.strip() for line in truth]
    
    pred = []
    with open(pred_file, 'r') as file:
        pred = file.readlines()
        pred = [line.strip() for line in pred]
    
    scores = bleu.corpus_score(pred, [truth])

    print(scores)
    
    return scores

#%%
lang_pairs = [(cantonese, lang) for lang in langauges] + [(lang, cantonese) for lang in langauges]


#%%
bertscore_metric_en = load('bertscore', lang='en')
bertscore_metric_zh = load('bertscore', lang='zh')

#%%
def get_bertscore(src_lang, tgt_lang, model_dir):
    # src_file = os.path.join(truth_dir, f'{src_lang}.txt')
    truth_file = Path(truth_dir, f'{tgt_lang}.txt')
    pred_file = Path(model_dir, f'{src_lang}_to_{tgt_lang}.txt')
    src_lines, truth_lines, pred_lines = [], [], []
    # with open(src_file, 'r') as file:
    #     src_lines = file.readlines()
    #     src_lines = [line.strip() for line in src_lines]
    with open(truth_file, 'r') as file:
        truth_lines = file.readlines()
        truth_lines = [line.strip() for line in truth_lines]
    with open(pred_file, 'r') as file:
        pred_lines = file.readlines()
        pred_lines = [line.strip() for line in pred_lines]
    bertscore_metric = bertscore_metric_en if tgt_lang == 'eng_Latn' else bertscore_metric_zh
    lang = 'en' if tgt_lang == 'eng_Latn' else 'zh'
    precisions = np.array([])
    recalls = np.array([])
    f1s = np.array([])
    for i in tqdm(range(0, len(pred_lines), 10)):
        results = bertscore_metric.compute(predictions=pred_lines[i:i+10], references=truth_lines[i:i+10], lang=lang)
        precisions = np.append(precisions, results['precision'])
        recalls = np.append(recalls, results['recall'])
        f1s = np.append(f1s, results['f1'])
    precision_mean = precisions.mean().item()
    recall_mean = recalls.mean().item()
    f1_mean = f1s.mean().item()
    print(f'Precision: {precision_mean}, Recall: {recall_mean}, F1: {f1_mean}')
    return precision_mean, recall_mean, f1_mean

# #%%

comet_model_path = download_model("Unbabel/wmt22-comet-da")
device = 'cuda' if cuda.is_available() else 'cpu'
comet_model = load_from_checkpoint(comet_model_path)

def get_comet_score(src_lang, tgt_lang, model_dir):
    # src_file = os.path.join(truth_dir, f'{src_lang}.txt')
    # truth_file = os.path.join(root_dir, truth_dir, f'{tgt_lang}.txt')
    # pred_file = os.path.join(root_dir, model_dir, f'{model_dir}_{src_lang}_to_{tgt_lang}.txt')

    src_file = Path(truth_dir, f'{src_lang}.txt')
    truth_file = Path(truth_dir, f'{tgt_lang}.txt')
    pred_file = Path(model_dir, f'{src_lang}_to_{tgt_lang}.txt')

    src_lines, truth_lines, pred_lines = [], [], []
    with open(src_file, 'r') as file:
        src_lines = file.readlines()
        src_lines = [line.strip() for line in src_lines]
    with open(truth_file, 'r') as file:
        truth_lines = file.readlines()
        truth_lines = [line.strip() for line in truth_lines]
    with open(pred_file, 'r') as file:
        pred_lines = file.readlines()
        pred_lines = [line.strip() for line in pred_lines]
    data = [
        {"src": src, "mt": mt, "ref": ref} for src, mt, ref in zip(src_lines, pred_lines, truth_lines)
    ]
    comet_output = comet_model.predict(data, batch_size=8, gpus=1)
    return comet_output.system_score

# #%%
# get_comet_score('cmn_Hans', 'yue_Hant', 'sft_only')

#%%
# create dataframe with rows representing models, columns representing bleu scores for each language pair

def get_record(src, tgt, model):
    # check if model exists
    if not Path(model, f"{src}_to_{tgt}.txt").exists():
        return None
    print(f"Evaluating {model} on {src} to {tgt}")
    bleu_scores = get_bleu_score(src, tgt, model)
    bert_scores = get_bertscore(src, tgt, model)
    comet_scores = get_comet_score(src, tgt, model)
    bleu_score = round(bleu_scores.score, 2)
    bert_precision = round(bert_scores[0], 2)
    bert_recall = round(bert_scores[1], 2)
    bert_f1 = round(bert_scores[2], 2)
    return {
        "model": model,
        "src_lang": src,
        "tgt_lang": tgt,
        "bleu_score": bleu_score,
        "bert_precision": bert_precision,
        "bert_recall": bert_recall,
        "bert_f1": bert_f1,
        "comet_score": comet_scores,
    }



records = []
for src, tgt in lang_pairs:
    for model in output_dirs:
        record = get_record(src, tgt, model)
        if record is not None:
            records.append(record)

eval_results = pd.DataFrame(records)

print(eval_results)

# plot bar graph on bleu scores




timestr = time.strftime("%Y%m%d-%H%M%S")


eval_results.to_csv(f'{results_dir}/{timestr}_evaluation_results.csv', index=False)

plt.plot(eval_results['model'], eval_results['bleu_score'])


#%%




