# Enhancing Recurrent Neural Networks with Sememes
If you have any question, feel free to contact us: qinyj16@mails.tsinghua.edu.cn, scouyang4354@gmail.com

## Sememe Dataset
sememe.txt include all 2186 sememe. hownet_en.txt is an English version HowNet while each line of word is followed by a line of all its sememes. More Details about HowNet could be found in paper.

## Language model

For language model, you could run by:

```
cd LM
CUDA_VISIBLE_DEVICES=3 python main.py --cuda --emsize 1500 --nhid 1500  --epochs 40 --sememe_dim 1500 --model_type LSTM_cell --dropout 0.7
```
The default command is for LSTM+cell, if you want to test other models, you could replace corresponding command.

## Sentence encoders

First please download the pretrained glove embeddings, which can be achieved through: https://nlp.stanford.edu/projects/glove/ 

Then you could easily train on SNLI Dataset by:

```
python3 train_nli.py --word_emb_path ../glove/glove.840B.300d.txt --encoder_type LSTM_extra_cell --gpu_id 2
```

All the models mentioned in paper are included in models.py. Check the name of the encoder and notice that you have to change the parameter --encoder_type.

## Adversarial attack

For adversarial attack, you could generate adversarial examples after you have trainning an NLI model.

```
python3 adv_attack.py
```
Then you should add these new example to the original NLI dataset and keep trainning the NLI model for more epochs. Please keep encoder type all the same.
