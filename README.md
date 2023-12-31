# NumHG

Here are the dataset and evaluation code from the paper:

*NumHG: A Dataset for Number-Focused Headline Generation*


## How to Install
- `Python version >= 3.7`
- `PyTorch version >= 1.10.0`

Install the package required:
```sh
pip install -r requirements.txt
```
```sh
git clone https://github.com/AIPHES/emnlp19-moverscore.git
cd emnlp19-moverscore/
python setup.py install
```


## How to Evaluate
Run the evaluation code:
```sh
python numhg_eval.py \
--tgt_path=target_path \
--pre_path=predict_result_path \
--num_gt_path=numeral_ground_truth_path \
--num_type_path=numeral_type_path
```
