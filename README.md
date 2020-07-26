# Improving Sequence Modeling Ability of Recurrent Neural Networks via Sememes
If you have any question, feel free to contact us: yujiaqin16@gmail.com scouyang4354@gmail.com

## Sememe Dataset
sememe.txt include all 2186 sememe. hownet_en.txt is an English version of HowNet, where each line of word is followed by a line of all its sememes. More Details about HowNet could be found in paper.

## Language model
For language model, you could run by:

```
cd LM
CUDA_VISIBLE_DEVICES=3 python main.py --cuda --emsize 1500 --nhid 1500  --epochs 40 --sememe_dim 1500 --model_type LSTM_cell --dropout 0.7
```

## Sentence encoders

First please download the pretrained glove embeddings, which can be achieved through: https://nlp.stanford.edu/projects/glove/
and SNLI dataset through:
bash dataset/get_data.bash

Then train on SNLI Dataset by:

```
python3 train_nli.py --word_emb_path ../glove/glove.840B.300d.txt --encoder_type LSTM_cell --gpu_id 2
```

## Adversarial attack

For adversarial attack, you could generate adversarial examples after you have trainning an NLI model.

```
python3 adv_attack.py
```
Then you should add these new example to the original NLI dataset and keep trainning the NLI model for more epochs. Please keep encoder type all the same.
