# Improving Sequence Modeling Ability of Recurrent Neural Networks via Sememes
This repo is the official implementation of the paper "[Improving Sequence Modeling Ability of Recurrent Neural Networks via Sememes](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9149672)" that is published in IEEE/ACM Transactions on Audio, Speech, and Language Processing (TASLP). 

If you have any question about the code, feel free to contact: yujiaqin16@gmail.com & scouyang4354@gmail.com

## Sememe Dataset
`sememe.txt` comprises all 2186 sememes. `hownet_en.txt` is an English version of HowNet sememe knowledge dictionary, where each line of word is followed by a line of all its sememes. More details about HowNet and sememes can be found in paper.

## Language Modeling
First download PTB dataset and put it at `data/ptb`, then run:

```
cd LM
CUDA_VISIBLE_DEVICES=3 python main.py --cuda --emsize 1500 --nhid 1500  --epochs 40 --sememe_dim 1500 --model_type LSTM_cell --dropout 0.65
```

## Natural Language Inference

First download the pretrained glove embeddings through: https://nlp.stanford.edu/projects/glove/ and put it at `../glove`.
Also download SNLI dataset through:

```
bash dataset/get_data.bash
```
and put it at `./dataset/SNLI`.

Then train on SNLI Dataset by:

```
python3 train_nli.py --word_emb_path ../glove/glove.840B.300d.txt --encoder_type LSTM_cell --gpu_id 2
```

## Text Classification

Save the encoder trained on SNLI at `./savedir`, and test the downstream sentence enoding task using `SentEval/encoders/infersent.py` (you may need to modify the corresponding paths)

## Adversarial Attack

For adversarial attack, you could generate adversarial examples after you have trainning an NLI model. Please change your trained `.pkl` file path and corresponding encoder type.

```
python3 adv_attack.py
```
Then you should add these new example to the original NLI dataset and keep training the NLI model for more epochs.

## Citation

Please cite the following paper if the code or data help you:

```
@article{qin2020improving,
  author={Qin, Yujia and Qi, Fanchao and Ouyang, Sicong and Liu, Zhiyuan and Yang, Cheng and Wang, Yasheng and Liu, Qun and Sun, Maosong},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing}, 
  title={Improving Sequence Modeling Ability of Recurrent Neural Networks via Sememes}, 
  year={2020},
  volume={28},
  number={},
  pages={2364-2373},
 }
```