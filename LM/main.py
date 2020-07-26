# coding: utf-8
import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx
import torch.optim as optim
import hashlib
import data
import model
from sememe import Sememe
import random
import numpy as np
# SEMEME DATASET
parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./data/ptb',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--model_type', type=str, default='LSTM_gate')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--sememe_dim', default=200, type=int,
                help='Size of input sememe vector')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')
parser.add_argument('--lr_decay', type=float, default=4,
                    help='learning rate decay')
parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
parser.add_argument('--drop_rate', type=float, default=1,
                    help='drop rate for sememes')
args = parser.parse_args()
print(args)
torch.cuda.set_device(args.gpu_id)
random_seed = random.randint(10, 10000)
print(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
# Set the random seed manually for reproducibility.
#torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
if 'ptb' in args.data:
    args.save = 'ptb_' + str(args.sememe_dim) + '_' + args.model_type + '_' + str(args.dropout) + '_' + args.save + '_' + str(args.drop_rate) + '_' + str(random_seed)
else:
    args.save = 'wiki_' + str(args.sememe_dim) + '_' + args.model_type + '_' + str(args.dropout) + '_' + args.save + '_' + str(args.drop_rate) + '_' + str(random_seed)
print('will save to %s'%(args.save))

#sememe.txt是共有哪些英文sememe
sememe_dir = 'data/sememe.txt'
#hownet_en.txt是每个英文词有哪些英文sememe
hownet_dir = 'data/hownet_en.txt'
#用来将每个词还原到原型
lemma_dir =  'data/lemmatization.txt'
# sememe是模仿vocab模块写的
sememe = Sememe(hownet_dir = hownet_dir, sememe_dir = sememe_dir, lemma_dir = lemma_dir, filename = hownet_dir, lower = True, drop_rate = args.drop_rate)
device = torch.device("cuda" if args.cuda else "cpu")
emb_s_file = os.path.join(args.data, 'snli_embed_sememe_' + str(args.sememe_dim) + '_' + str(args.dropout) + '.pth')
if os.path.isfile(emb_s_file):
    emb_s = torch.load(emb_s_file)
else:
    emb_s = torch.zeros(sememe.size(), args.sememe_dim, dtype=torch.float, device=device)
    emb_s.normal_(0, 0.05)
    torch.save(emb_s, emb_s_file)

###############################################################################
# Load data
###############################################################################
fn = 'corpus.{}.data'.format(hashlib.md5(args.data.encode()).hexdigest())
#if os.path.exists(fn):
if False:
    print('Loading cached dataset...')
    corpus = torch.load(fn)
    print('Loading end')
else:
    print('Producing dataset...')
    corpus = data.Corpus(args.data, sememe)
    torch.save(corpus, fn)
    print('Producing end')

# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)
def batchify_sememe(data, bsz):
    nbatch = len(data) // bsz
    print(nbatch)
    data = data[0:nbatch*bsz]
    print(len(data))
    new_data = torch.zeros([nbatch, bsz, sememe.size()], dtype = torch.uint8).cuda()
    for i in range(nbatch):
        for j in range(bsz):
            for item in data[j*nbatch+i]:
                new_data[i,j,item] = 1
    return new_data
eval_batch_size = 10
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)
train_sememes = batchify_sememe(corpus.train_sememes, args.batch_size)
valid_sememes = batchify_sememe(corpus.valid_sememes, eval_batch_size)
test_sememes = batchify_sememe(corpus.test_sememes, eval_batch_size)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.sememe_dim, sememe.size(), args.model_type, args.dropout, args.tied).to(device)
# plug these into embedding matrix inside model
model.emb_sememe.weight.data.copy_(emb_s)
criterion = nn.CrossEntropyLoss()

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def get_batch(source, i, source_s = None):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    data_s = source_s[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target, data_s


def evaluate(data_source,data_s_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets, data_s = get_batch(data_source, i, data_s_source)
            output, hidden = model(data, data_s, hidden)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
            hidden = repackage_hidden(hidden)
    return total_loss / (len(data_source) - 1)


def train():
    optimizer = optim.SGD(filter(lambda p: p.requires_grad,
                                      model.parameters()), lr=args.lr, weight_decay=0)
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        optimizer.zero_grad()
        data, targets, data_s = get_batch(train_data, i, train_sememes)

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data, data_s, hidden)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        for p in model.parameters():
            if(p.requires_grad):
                if p.grad is not None:
                    p.data.add_(-lr, p.grad.data)

        total_loss += loss.item()
        #optimizer.step()
        #optimizer.zero_grad()


        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


def export_onnx(path, batch_size, seq_len):
    print('The model is also exported in ONNX format at {}'.
          format(os.path.realpath(args.onnx_export)))
    model.eval()
    dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
    hidden = model.init_hidden(batch_size)
    torch.onnx.export(model, (dummy_input, hidden), path)


# Loop over epochs.
lr = args.lr
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(val_data, valid_sememes)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= args.lr_decay
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    model.rnn.flatten_parameters()

# Run on test data.
test_loss = evaluate(test_data,test_sememes)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)

if len(args.onnx_export) > 0:
    # Export the model in ONNX format.
    export_onnx(args.onnx_export, batch_size=1, seq_len=args.bptt)
