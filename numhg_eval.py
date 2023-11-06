import argparse, re, os
from typing import List, Union, Iterable
from itertools import zip_longest
from compare_mt.rouge.rouge_scorer import RougeScorer
from nltk import sent_tokenize, word_tokenize
from sklearn.metrics import accuracy_score
import numpy as np
from tqdm import tqdm

def cal_num_acc(predict_path, num_gt_path, num_path):
    pred_all, gt_all = [], []
    pred_copy, gt_copy = [], []
    pred_cal, gt_cal = [], []
    pattern = re.compile(r'\d{1,3}(?:,\d{3})+|\d+[\/\.]{0,1}\d+|\d+')

    with open(predict_path) as pred, open(num_gt_path) as target, open(num_path) as num_type:
        total_num, count_copy, count_cal = 0, 0, 0
        for (hyp, gt, num) in zip(pred, target, num_type):
            generated_num_list = pattern.findall(hyp)
            if(str(num).strip()=='0'): # Copy
                count_copy += 1
                if(str(gt).strip() in generated_num_list and len(generated_num_list)==1):
                    pred_copy.append(1)
                    pred_all.append(1)
                else:
                    pred_copy.append(0)
                    pred_all.append(0)
                gt_copy.append(1)
                gt_all.append(1)
            else:
                count_cal += 1
                if(str(gt).strip() in generated_num_list and len(generated_num_list)==1):
                    pred_cal.append(1)
                    pred_all.append(1)
                else:
                    pred_cal.append(0)
                    pred_all.append(0)
                gt_cal.append(1)
                gt_all.append(1)
            total_num += 1
        print("All Accuracy: %.6f, Copy Accuracy: %.6f, Cal Accuracy: %.6f"%(accuracy_score(gt_all, pred_all), accuracy_score(gt_copy, pred_copy), accuracy_score(gt_cal, pred_cal)))


def cal_rouge_score(target_path, predict_path):
    rouge1, rouge2, rougeLsum = 0, 0, 0
    rouge_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)

    def process(x):
        return sent_tokenize(" ".join(word_tokenize(x.strip())))

    with open(predict_path) as pred, open(target_path) as target:
        total_num = 0
        for (hyp, ref) in zip(pred, target):
            hyp = hyp.strip().strip("\"")
            ref = ref.strip().strip("\"")
            hyp = process(hyp)
            ref = process(ref)
            score = rouge_scorer.score("\n".join(ref), "\n".join(hyp))
            rouge1 += score["rouge1"].fmeasure
            rouge2 += score["rouge2"].fmeasure
            rougeLsum += score["rougeLsum"].fmeasure
            total_num += 1
        rouge1 = rouge1 / total_num
        rouge2 = rouge2 / total_num
        rougeLsum = rougeLsum / total_num
        print("evaluation rouge1: %.6f, rouge2: %.6f, rougeL: %.6f"%(rouge1, rouge2, rougeLsum))


def cal_mover_score(target_path, predict_path):
    from moverscore_v2 import word_mover_score, get_idf_dict
    from collections import defaultdict

    def sentence_score(hypothesis: str, references: List[str], trace=0):
        idf_dict_hyp = defaultdict(lambda: 1.)
        idf_dict_ref = defaultdict(lambda: 1.)
        hypothesis = [hypothesis] * len(references)
        sentence_score = 0 
        scores = word_mover_score(references, hypothesis, idf_dict_ref, idf_dict_hyp, stop_words=[], n_gram=2, remove_subwords=False)
        sentence_score = np.mean(scores)
        if trace > 0:
            print(hypothesis, references, sentence_score)
                
        return sentence_score

    def corpus_score(sys_stream: List[str], ref_streams:Union[str, List[Iterable[str]]], trace=0):
        if isinstance(sys_stream, str):
            sys_stream = [sys_stream]

        if isinstance(ref_streams, str):
            ref_streams = [[ref_streams]]

        fhs = [sys_stream] + ref_streams

        corpus_score = 0
        for lines in tqdm(zip_longest(*fhs)):
            if None in lines:
                raise EOFError("Source and reference streams have different lengths!")
                
            hypo, *refs = lines
            corpus_score += sentence_score(hypo, refs, trace=0)
            
        corpus_score /= len(sys_stream)

        return corpus_score

    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ['MOVERSCORE_MODEL'] = "roberta-large"
    with open(predict_path) as pred, open(target_path) as target:
        total_num, sentence_score = 0.0, 0.0
        hyp_list, ref_list = [], []
        for (hyp, ref) in tqdm(zip(pred, target)):
            ref_list.append(ref.strip().strip("\""))
            hyp_list.append(hyp.strip().strip("\""))
        idf_dict_hyp = get_idf_dict(hyp_list)
        idf_dict_ref = get_idf_dict(ref_list)
        mover_score = word_mover_score(ref_list, hyp_list, idf_dict_ref, idf_dict_hyp, stop_words=[], n_gram=1, remove_subwords=True)
        print("evaluation MoverScore: %.6f"%(np.mean(mover_score)))

    # with open(predict_path) as pred, open(target_path) as target:
    #     hyp_list, ref_list = [], []
    #     for (hyp, ref) in zip(pred, target):
    #         ref_list.append(ref.strip())
    #         hyp_list.append(hyp.strip())

    #     mover_score = corpus_score(hyp_list, [ref_list])
    #     print("evaluation MoverScore: %.6f"%(np.mean(mover_score)))


def cal_bert_score(target_path, predict_path):
    os.system("bert-score -r {} -c {} --lang en --rescale_with_baseline".format(target_path, predict_path))


def main(args):
    print('Calculating Rouge Score......')
    cal_rouge_score(args.tgt_path, args.pre_path)

    print('\nCalculating Numeral Accuracy......')
    cal_num_acc(args.pre_path, args.num_gt_path, args.num_type_path)

    print('\nCalculating Moverscore......')
    cal_mover_score(args.tgt_path, args.pre_path)

    print('\nCalculating BERTScore......')
    cal_bert_score(args.tgt_path, args.pre_path)


if __name__ ==  "__main__":
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument("--tgt_path", default="", type=str, help="target path")
    parser.add_argument("--pre_path", default="", type=str, help="predict path")
    parser.add_argument("--num_gt_path", default="", type=str, help="numerical ground truth path")
    parser.add_argument("--num_type_path", default="", type=str, help="type of each summary, 1:Reasoning, 0:Copy")
    args = parser.parse_args()
    main(args)