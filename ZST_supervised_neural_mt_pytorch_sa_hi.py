
# **```ZST Neural Machine Translation```**

This notebook runs Seq2Seq model to translate Sanskrit to Hindi.using ZST approach

NOTE: This requires GPU to run. With only CPU, processing can be painfully slow, with GPU also, training takes around 1.5 hrs
"""

!rm -rf sample_data
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from google.colab import drive
drive.mount('/content/drive')

"""This is the repo which contains the parallel data we are going to use."""

!unzip /content/drive/MyDrive/sa_hi_zst1_25gb-20240417T133941Z-001.zip -d /content/drive/MyDrive/

pwd

#%cd /content/drive/MyDrive/sa_hi_zst1_25gb/data/
sour_lines = []
tar_lines = []
with open('/content/drive/MyDrive/sa_hi_zst1_25gb/data/train_source1.txt-filtered.sa', 'r') as f:
  sour_lines.extend([x.replace('\n', '') for x in f.readlines()])

with open('/content/drive/MyDrive/sa_hi_zst1_25gb/data/train_source2.txt-filtered.en', 'r') as f:
  sour_lines.extend([x.replace('\n', '') for x in f.readlines()])

#with open('/content/drive/MyDrive/sa_hi_zst1_25gb/data/test/src-test.txt-filtered.sa', 'r') as f:
  #sour_lines.extend([x.replace('\n', '') for x in f.readlines()])

# with open('manu_english.txt', 'r') as f:
#   eng_lines.extend([x.replace('\n', '') for x in f.readlines()])

# with open('ramayan_english.txt', 'r') as f:
#   eng_lines.extend([x.replace('\n', '') for x in f.readlines()])

# with open('rigveda_english.txt', 'r') as f:
#   eng_lines.extend([x.replace('\n', '') for x in f.readlines()])



with open('/content/drive/MyDrive/sa_hi_zst1_25gb/data/train_target1.txt-filtered.en', 'r') as f:
  tar_lines.extend([x.replace('\n', '') for x in f.readlines()])

with open('/content/drive/MyDrive/sa_hi_zst1_25gb/data/train_target2.txt-filtered.hi', 'r') as f:
  tar_lines.extend([x.replace('\n', '') for x in f.readlines()])


#with open('/content/drive/MyDrive/sa_hi_zst1_25gb/data/test/tgt-test.txt-filtered.hi', 'r') as f:
  #tar_lines.extend([x.replace('\n', '') for x in f.readlines()])
# with open('manu_sanskrit.txt', 'r') as f:
#   sanskrit_lines.extend([x.replace('\n', '') for x in f.readlines()])

# with open('ramayan_sanskrit.txt', 'r') as f:
#   sanskrit_lines.extend([x.replace('\n', '') for x in f.readlines()])

# with open('rigveda_sanskrit.txt', 'r') as f:
#   sanskrit_lines.extend([x.replace('\n', '') for x in f.readlines()])

print(sour_lines[:100])
print(tar_lines[:100])

print(len(sour_lines))
print(len(tar_lines))

# Randomly shuffling the data into training and dev set, the test data is already provided on github
import random
c = list(zip(sour_lines, tar_lines))
random.shuffle(c)

sour_lines, tar_lines = zip(*c)

train_text_so = sour_lines[:-1374]
train_text_ta = tar_lines[:-1374]

dev_text_so = sour_lines[-1374:]
dev_text_ta = tar_lines[-1374:]

print(len(train_text_so))
print(len(dev_text_so))

# Start and End tokens
SOS_token = 0
EOS_token = 1


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

# Reading the data and forming pairs
def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Forming pairs of sentences
    pairs = [[train_text_so[i], train_text_ta[i]] for i in range(len(train_text_so))]
    #pairs_v=
    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

"""Filtering the data by only keeping sentences of length less than 100 words



"""

MAX_LENGTH = 100

def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


input_lang, output_lang, pairs = prepareData('source', 'target', False)
print(random.choice(pairs)) # a random pair of parallel sentences

"""Seq2Seq Model
=================

A seq2seq network is a model consisting of two RNNs called the encoder and decoder. The encoder reads an input sequence and outputs a single vector, and the decoder reads that vector to produce an output sequence.

Unlike sequence prediction with a single RNN, where every input
corresponds to an output, the seq2seq model frees us from sequence
length and order, which makes it ideal for translation between two
languages.

The Encoder
-----------

The encoder of a seq2seq network is a RNN that outputs some value for
every word from the input sentence. For every input word the encoder
outputs a vector and a hidden state, and uses the hidden state for the
next input word.
"""

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

"""The Decoder
-----------

The decoder is another RNN that takes the encoder output vector(s) and
outputs a sequence of words to create the translation.

Attention Decoder

If only the context vector is passed betweeen the encoder and decoder,
that single vector carries the burden of encoding the entire sentence.

Attention allows the decoder network to "focus" on a different part of
the encoder's outputs for every step of the decoder's own outputs. First
we calculate a set of *attention weights*. These will be multiplied by
the encoder output vectors to create a weighted combination. The result
(called ``attn_applied`` in the code) should contain information about
that specific part of the input sequence, and thus help the decoder
choose the right output words.

Calculating the attention weights is done with another feed-forward
layer ``attn``, using the decoder's input and hidden state as inputs.
Because there are sentences of all sizes in the training data, to
actually create and train this layer we have to choose a maximum
sentence length (input length, for encoder outputs) that it can apply
to. Sentences of the maximum length will use all the attention weights,
while shorter sentences will only use the first few.
"""

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

"""Training
========

Preparing Training Data
-----------------------

To train, for each pair we will need an input tensor (indexes of the
words in the input sentence) and target tensor (indexes of the words in
the target sentence). While creating these vectors we will append the
EOS token to both sequences.
"""

import random
def indexesFromSentence(lang, sentence):
    res = []
    for word in sentence.split(' '):
      if word not in lang.word2index.keys():
        res.append(random.choice(list(lang.word2index.values())))
      else:
        res.append(lang.word2index[word])
    return res

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

"""Training the Model
------------------

To train we run the input sentence through the encoder, and keep track
of every output and the latest hidden state. Then the decoder is given
the ``<SOS>`` token as its first input, and the last hidden state of the
encoder as its first hidden state.



"""

teacher_forcing_ratio = 0.5
def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]

    else:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

# Print time elapsed and estimated time remaining given the current time and progress %.
import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

# Each iteration of training
def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(random.choice(pairs))
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

"""Evaluation
==========

Evaluation is mostly the same as training, but there are no targets so
we simply feed the decoder's predictions back to itself for each step.
Every time it predicts a word we add it to the output string, and if it
predicts the EOS token we stop there.
"""

def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]

"""Google translate API to translate English to Hindi"""

$ pip install googletrans

!pip install googletrans
from googletrans import Translator
translator = Translator()



!pip install googletrans

def evaluate_test(encoder, decoder):
  test = []
  with open('/content/drive/MyDrive/sa_hi_zst1_25gb/data/test/src-test.txt-filtered.sa', 'r') as f:
  #with open('/content/drive/MyDrive/sa_hi_zst1_25gb/data/test/test_sanskrit.sa', 'r') as f:
    #test_text = [x.replace('\n', '') for x in f.readlines()]
  #for sa_text in test_text:
   for sa_text in test_text:
    # print('>', pair[0])
    # print('=', pair[1])
    output_words, attentions = evaluate(encoder, decoder, sa_text)
    output_sentence = ' '.join(output_words)
    #print('<', output_sentence)
    # print('')
    test.append(output_sentence)
    #print(test)

  # translated = []
  # print(len(test))

  # #for line in test:
  # ans = translator.translate(test, dest='hi',src='auto')
  # print(ans)

  #translated.append(ans.text)

  with open('/content/drive/MyDrive/sa_hi_zst1_25gb/data/test/translated_hi1.hi', 'w') as f:
    for line in test:
      f.write(line + '\n')

  # calculate bleu score
  import nltk
  with open('/content/drive/MyDrive/sa_hi_zst1_25gb/data/test/tgt-test.txt-filtered.hi', 'r') as f: # ground truth file
    reference = [x.replace('\n', '') for x in f.readlines()]

  with open('/content/drive/MyDrive/sa_hi_zst1_25gb/data/test/translated_hi1.hi', 'r') as f: # translated hindi file (our output)
    candidate = [x.replace('\n', '') for x in f.readlines()]

  from nltk.translate.bleu_score import SmoothingFunction
  smoothie = SmoothingFunction().method4
  #print(test_text,reference,canditate)
  #print("bleu", nltk.translate.bleu_score.corpus_bleu(reference, candidate, smoothing_function=smoothie))
  print("bleu", nltk.translate.bleu_score.corpus_bleu(reference, candidate, smoothing_function=smoothie))
  print("bleu1", nltk.translate.bleu_score.corpus_bleu(reference, candidate,weights=(1,0,0,0), smoothing_function=smoothie))
  print("bleu2", nltk.translate.bleu_score.corpus_bleu(reference, candidate,weights=(0,1,0,0), smoothing_function=smoothie))
  print("bleu3", nltk.translate.bleu_score.corpus_bleu(reference, candidate,weights=(0,0,1,0), smoothing_function=smoothie))
  print("bleu4", nltk.translate.bleu_score.corpus_bleu(reference, candidate,weights=(1,0,0,1), smoothing_function=smoothie))
  print("chrf",nltk.translate.chrf_score.corpus_chrf(reference, candidate))

####for validation data set BLEU scores###
#dev_text_so = sour_lines[-1374:]
#dev_text_ta = tar_lines[-1374:]

def evaluate_test(encoder, decoder):
  test = []
 # with open('/content/drive/MyDrive/sa_hi_zst1_25gb/data/test/src-test.txt-filtered.sa', 'r') as f:
  #with open('/content/drive/MyDrive/sa_hi_zst1_25gb/data/test/test_sanskrit.sa', 'r') as f:
    #test_text = [x.replace('\n', '') for x in f.readlines()]
  #for sa_text in test_text:
  for sa_text in dev_text_so:
    # print('>', pair[0])
    # print('=', pair[1])
    output_words, attentions = evaluate(encoder, decoder, sa_text)
    output_sentence = ' '.join(output_words)
    #print('<', output_sentence)
    # print('')
    test.append(output_sentence)
    #print(test)

  # translated = []
  # print(len(test))

  # #for line in test:
  # ans = translator.translate(test, dest='hi',src='auto')
  # print(ans)

  #translated.append(ans.text)

  with open('/content/drive/MyDrive/sa_hi_zst1_25gb/data/test/translated_val.tar', 'w') as f:
    for line in test:
      f.write(line + '\n')

  # calculate bleu score
  import nltk
  #with open('/content/drive/MyDrive/sa_hi_zst1_25gb/data/test/tgt-test.txt-filtered.hi', 'r') as f: # ground truth file
  reference = dev_text_ta

  with open('/content/drive/MyDrive/sa_hi_zst1_25gb/data/test/translated_val.tar', 'r') as f: # translated hindi file (our output)
    candidate = [x.replace('\n', '') for x in f.readlines()]

  from nltk.translate.bleu_score import SmoothingFunction
  smoothie = SmoothingFunction().method4
  #print(test_text,reference,canditate)
  #print("bleu", nltk.translate.bleu_score.corpus_bleu(reference, candidate, smoothing_function=smoothie))
  print("bleu", nltk.translate.bleu_score.corpus_bleu(reference, candidate, smoothing_function=smoothie))
  print("bleu1", nltk.translate.bleu_score.corpus_bleu(reference, candidate,weights=(1,0,0,0), smoothing_function=smoothie))
  print("bleu2", nltk.translate.bleu_score.corpus_bleu(reference, candidate,weights=(0,1,0,0), smoothing_function=smoothie))
  print("bleu3", nltk.translate.bleu_score.corpus_bleu(reference, candidate,weights=(0,0,1,0), smoothing_function=smoothie))
  print("bleu4", nltk.translate.bleu_score.corpus_bleu(reference, candidate,weights=(1,0,0,1), smoothing_function=smoothie))
  print("chrf",nltk.translate.chrf_score.corpus_chrf(reference, candidate))

# defining hidden layer size and the encoder and decoder networks
hidden_size = 256

encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)

# train
trainIters(encoder1, attn_decoder1, 80000, print_every=5000)

## save model for

# saving the trained models
torch.save(encoder1.state_dict(), '/content/drive/MyDrive/sa_hi_zst1_25gb/trained_encoder.pth')
torch.save(attn_decoder1.state_dict(), '/content/drive/MyDrive/sa_hi_zst1_25gb/trained_decoder.pth')





###validation data set###

evaluate_test(encoder1, attn_decoder1)

###for validation data output print###

# calculate bleu score
import nltk
  #with open('/content/drive/MyDrive/sa_hi_zst1_25gb/data/test/tgt-test.txt-filtered.hi', 'r') as f: # ground truth file
reference = dev_text_ta

with open('/content/drive/MyDrive/sa_hi_zst1_25gb/data/test/translated_val.tar', 'r') as f: # translated hindi file (our output)
    candidate = [x.replace('\n', '') for x in f.readlines()]
for i in range(len(reference)):
  #print("source sanskrit:",test_text[i])
  print("source text:",dev_text_so[i])
  print("actual target:",reference[i])
  print("predicted target:",candidate[i])

with open('/content/drive/MyDrive/sa_hi_zst1_25gb/data/test/src-test.txt-filtered.sa', 'r') as f:
    test_text = [x.replace('\n', '') for x in f.readlines()]
with open('/content/drive/MyDrive/sa_hi_zst1_25gb/data/test/tgt-test.txt-filtered.hi', 'r') as f: # ground truth file
    reference = [x.replace('\n', '') for x in f.readlines()]

with open('/content/drive/MyDrive/sa_hi_zst1_25gb/data/test/translated_hi.hi', 'r') as f: # translated hindi file (our output)
    candidate = [x.replace('\n', '') for x in f.readlines()]
for i in range(len(test_text)):
  print("source sanskrit:",test_text[i])
  print("actual hindi:",reference[i])
  print("predicted hindi:",candidate[i])

hidden_size = 256

encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)

#encoder1 = TheModelClass(*args, **kwargs)
encoder1.load_state_dict(torch.load('/content/drive/MyDrive/sa_hi_zst1_25gb/trained_encoder.pth'))
#attn_decoder1=TheModelClass(*args, **kwargs)
attn_decoder1.load_state_dict(torch.load('/content/drive/MyDrive/sa_hi_zst1_25gb/trained_decoder.pth'))
#model.eval()

trainIters(encoder1, attn_decoder1, 40000, print_every=5000)

# saving the trained models
torch.save(encoder1.state_dict(), '/content/drive/MyDrive/sa_hi_zst1_25gb/trained_encoder.pth')
torch.save(attn_decoder1.state_dict(), '/content/drive/MyDrive/sa_hi_zst1_25gb/trained_decoder.pth')

encoder1.load_state_dict(torch.load('/content/drive/MyDrive/sa_hi_zst1_25gb/trained_encoder.pth',map_location=torch.device('cpu')))
#attn_decoder1=TheModelClass(*args, **kwargs)
attn_decoder1.load_state_dict(torch.load('/content/drive/MyDrive/sa_hi_zst1_25gb/trained_decoder.pth',map_location=torch.device('cpu')))

evaluate_test(encoder1, attn_decoder1)

with open('/content/drive/MyDrive/sa_hi_zst1_25gb/data/test/test_sanskrit.sa', 'r') as f:
    test_text = [x.replace('\n', '') for x in f.readlines()]
with open('/content/drive/MyDrive/sa_hi_zst1_25gb/data/test/test_hindi.hi', 'r') as f: # ground truth file
    reference = [x.replace('\n', '') for x in f.readlines()]

with open('/content/drive/MyDrive/sa_hi_zst1_25gb/data/test/translated_hi1.hi', 'r') as f: # translated hindi file (our output)
    candidate = [x.replace('\n', '') for x in f.readlines()]
for i in range(len(test_text)):
  print("source sanskrit:",test_text[i])
  print("actual hindi:",reference[i])
  print("predicted hindi:",candidate[i])

with open('/content/drive/MyDrive/sa_hi_zst1_25gb/data/test/src-test.txt-filtered.sa', 'r') as f:
    test_text = [x.replace('\n', '') for x in f.readlines()]
with open('/content/drive/MyDrive/sa_hi_zst1_25gb/data/test/tgt-test.txt-filtered.hi', 'r') as f: # ground truth file
    reference = [x.replace('\n', '') for x in f.readlines()]

with open('/content/drive/MyDrive/sa_hi_zst1_25gb/data/test/translated_hi.hi', 'r') as f: # translated hindi file (our output)
    candidate = [x.replace('\n', '') for x in f.readlines()]
for i in range(len(test_text)):
  print("source sanskrit:",test_text[i])
  print("actual hindi:",reference[i])
  print("predicted hindi:",candidate[i])

# defining hidden layer size and the encoder and decoder networks
hidden_size = 256

encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)

# train
trainIters(encoder1, attn_decoder1, 1000, print_every=500)

ref1 = str('It is a guide to action that ensures that the military will forever heed Party commands').split()
print(ref1)

import nltk
ref1 = 'It is a guide to action that ensures that the military will forever heed Party commands'
ref2 = 'It is the guiding principle which guarantees the military forces always being under the command of the Party'
hyp1 = 'It is a guide to action which ensures that the military always obeys the commands of the party'
hyp2 ='It is to insure the troops forever hearing the activity guidebook that party direct'
#corpus_chrf([ref1, ref2], [hyp1, hyp2])
nltk.translate.chrf_score.corpus_chrf([ref1, ref2], [hyp1, hyp2])

ref1 = str('It is a guide to action that ensures that the military '
            'will forever heed Party commands').split()
print(ref1)
ref2 = str('It is the guiding principle which guarantees the military '
           'forces always being under the command of the Party').split()

hyp1 = str('It is a guide to action which ensures that the military '
           'always obeys the commands of the party').split()
hyp2 = str('It is to insure the troops forever hearing the activity '
            'guidebook that party direct')
nltk.translate.chrf_score.corpus_chrf([ref1, ref2], [hyp1, hyp2])

evaluate_test(encoder1, attn_decoder1)

with open('/content/drive/MyDrive/sa_hi_zst1_25gb/data/test/src-test.txt-filtered.sa', 'r') as f:
    test_text = [x.replace('\n', '') for x in f.readlines()]
with open('/content/drive/MyDrive/sa_hi_zst1_25gb/data/test/tgt-test.txt-filtered.hi', 'r') as f: # ground truth file
    reference = [x.replace('\n', '') for x in f.readlines()]

with open('/content/drive/MyDrive/sa_hi_zst1_25gb/data/test/translated_hi.hi', 'r') as f: # translated hindi file (our output)
    candidate = [x.replace('\n', '') for x in f.readlines()]
for i in range(len(test_text)):
  print("source sanskrit:",test_text[i])
  print("actual hindi:",reference[i])
  print("predicted hindi:",candidate[i])

evaluate_test(encoder1, attn_decoder1)

with open('/content/drive/MyDrive/sa_hi_zst1_25gb/data/test/src-test.txt-filtered.sa', 'r') as f:
    test_text = [x.replace('\n', '') for x in f.readlines()]
with open('/content/drive/MyDrive/sa_hi_zst1_25gb/data/test/tgt-test.txt-filtered.hi', 'r') as f: # ground truth file
    reference = [x.replace('\n', '') for x in f.readlines()]

with open('/content/drive/MyDrive/sa_hi_zst1_25gb/data/test/translated_hi.hi', 'r') as f: # translated hindi file (our output)
    candidate = [x.replace('\n', '') for x in f.readlines()]
for i in range(len(test_text)):
  print("source sanskrit:",test_text[i])
  print("actual hindi:",reference[i])
  print("predicted hindi:",candidate[i])

# with open('/content/drive/MyDrive/sa_hi_zst1_25gb/data/test/tgt-test.txt-filtered.hi', 'r') as f: # ground truth file
#     reference = [x.replace('\n', '') for x in f.readlines()]
# with open('/content/drive/MyDrive/sa_hi_zst1_25gb/data/test/translated_hi.hi', 'r') as f: # translated hindi file (our output)
#     candidate = [x.replace('\n', '') for x in f.readlines()]
import nltk
from nltk.translate.bleu_score import SmoothingFunction
smoothie = SmoothingFunction().method4
#print(test_text,reference,canditate)
#print("bleu", nltk.translate.bleu_score.corpus_bleu(reference, candidate, smoothing_function=smoothie))
print("bleu", nltk.translate.bleu_score.corpus_bleu(reference, candidate, smoothing_function=smoothie))
print("bleu1", nltk.translate.bleu_score.corpus_bleu(reference, candidate,weights=(1,0,0,0), smoothing_function=smoothie))
print("bleu2", nltk.translate.bleu_score.corpus_bleu(reference, candidate,weights=(0,1,0,0), smoothing_function=smoothie))
print("bleu3", nltk.translate.bleu_score.corpus_bleu(reference, candidate,weights=(0,0,1,0), smoothing_function=smoothie))
print("bleu4", nltk.translate.bleu_score.corpus_bleu(reference, candidate,weights=(1,0,0,1), smoothing_function=smoothie))
print("chrf",nltk.translate.chrf_score.corpus_chrf(reference, candidate))

evaluate_test(encoder1, attn_decoder1)

with open('/content/drive/MyDrive/sa_hi_zst1_25gb/data/test/src-test.txt-filtered.sa', 'r') as f:
    test_text = [x.replace('\n', '') for x in f.readlines()]
with open('/content/drive/MyDrive/sa_hi_zst1_25gb/data/test/tgt-test.txt-filtered.hi', 'r') as f: # ground truth file
    reference = [x.replace('\n', '') for x in f.readlines()]

with open('/content/drive/MyDrive/sa_hi_zst1_25gb/data/test/translated_hi.hi', 'r') as f: # translated hindi file (our output)
    candidate = [x.replace('\n', '') for x in f.readlines()]
for i in range(len(test_text)):
  print("source sanskrit:",test_text[i])
  print("actual hindi:",reference[i])
  print("predicted hindi:",candidate[i])

# with open('/content/drive/MyDrive/sa_hi_zst1_25gb/data/test/tgt-test.txt-filtered.hi', 'r') as f: # ground truth file
#     reference = [x.replace('\n', '') for x in f.readlines()]
# with open('/content/drive/MyDrive/sa_hi_zst1_25gb/data/test/translated_hi.hi', 'r') as f: # translated hindi file (our output)
#     candidate = [x.replace('\n', '') for x in f.readlines()]
import nltk
from nltk.translate.bleu_score import SmoothingFunction
smoothie = SmoothingFunction().method4
#print(test_text,reference,canditate)
#print("bleu", nltk.translate.bleu_score.corpus_bleu(reference, candidate, smoothing_function=smoothie))
print("bleu", nltk.translate.bleu_score.corpus_bleu(reference, candidate, smoothing_function=smoothie))
print("bleu1", nltk.translate.bleu_score.corpus_bleu(reference, candidate,weights=(1,0,0,0), smoothing_function=smoothie))
print("bleu2", nltk.translate.bleu_score.corpus_bleu(reference, candidate,weights=(0,1,0,0), smoothing_function=smoothie))
print("bleu3", nltk.translate.bleu_score.corpus_bleu(reference, candidate,weights=(0,0,1,0), smoothing_function=smoothie))
print("bleu4", nltk.translate.bleu_score.corpus_bleu(reference, candidate,weights=(1,0,0,1), smoothing_function=smoothie))
print("chrf",nltk.translate.chrf_score.corpus_chrf(reference, candidate))

### for spearman correlation between hum eval and BLEU1,2,3,4 and chrf####

import nltk
from nltk.translate.bleu_score import SmoothingFunction
smoothie = SmoothingFunction().method4
b1=[]
b2=[]
b3=[]
b4=[]
chRf=[]
with open('/content/drive/MyDrive/sa_hi_zst1_25gb/data/test/src-test.txt-filtered.sa', 'r') as f:
    test_text = [x.replace('\n', '') for x in f.readlines()]
with open('/content/drive/MyDrive/sa_hi_zst1_25gb/data/test/tgt-test.txt-filtered.hi', 'r') as f: # ground truth file
    reference = [x.replace('\n', '') for x in f.readlines()]

with open('/content/drive/MyDrive/sa_hi_zst1_25gb/data/test/translated_hi.hi', 'r') as f: # translated hindi file (our output)
    candidate = [x.replace('\n', '') for x in f.readlines()]
for i in range(0,50):
  b1.append(nltk.translate.bleu_score.sentence_bleu(reference[i], candidate[i],weights=(1,0,0,0), smoothing_function=smoothie))
  b2.append(nltk.translate.bleu_score.sentence_bleu(reference[i], candidate[i],weights=(0,1,0,0), smoothing_function=smoothie))
  b3.append(nltk.translate.bleu_score.sentence_bleu(reference[i], candidate[i],weights=(0,0,1,0), smoothing_function=smoothie))
  b4.append(nltk.translate.bleu_score.sentence_bleu(reference[i], candidate[i],weights=(0,0,0,1), smoothing_function=smoothie))
  chRf.append(nltk.translate.chrf_score.sentence_chrf(reference[i], candidate[i]))

print(b1)
print(b2)
print(b3)
print(b4)
print(chRf)


humval=[1, 2, 1, 1, 4, 1, 1, 1, 2, 1, 2, 2, 1, 2, 4, 4, 1, 1, 1, 3, 1, 2, 2, 4, 1, 2, 2, 1, 1, 2, 3, 2, 1, 4, 1, 3, 2, 2, 4, 1, 1, 1, 1, 2, 1, 1, 1, 2, 3, 3]

b1=[0.32142857142857145, 0.24242424242424243, 0.18181818181818182, 0.2295081967213115, 0.3125, 0.37037037037037035, 0.23611111111111113, 0.42857142857142855, 0.36, 0.2121212121212121, 0.52, 0.16901408450704225, 0.22666666666666666, 0.2, 0.3695652173913043, 0.34146341463414637, 0.42857142857142855, 0.19047619047619047, 0.4666666666666667, 0.2692307692307693, 0.3125, 0.42857142857142855, 0.3043478260869566, 0.325, 0.4117647058823529, 0.25, 0.20289855072463772, 0.32692307692307687, 0.25, 0.2602739726027397, 0.2857142857142857, 0.2857142857142857, 0.2, 0.3548387096774194, 0.43478260869565216, 0.391304347826087, 0.3137254901960784, 0.030303030303030304, 0.3, 0.4, 0.24489795918367346, 0.3, 0.375, 0.2857142857142857, 0.2857142857142857, 0.3478260869565218, 0.3333333333333333, 0.25, 0.26666666666666666, 0.5]
b2=[0.01234149818583409, 0.01092658612958275, 0.014719249777896742, 0.006851456440288851, 0.011179793234837825, 0.012676295638478188, 0.0060234734070648685, 0.01234149818583409, 0.0134119826036175, 0.01092658612958275, 0.0134119826036175, 0.006089542681487594, 0.005834443396670688, 0.007420987380060135, 0.008508091992197986, 0.00928393016676077, 0.010456906063204157, 0.005338333492582306, 0.011728266833317774, 0.013032386152085925, 0.011179793234837825, 0.01522261218861711, 0.01425224643604159, 0.009458665266958812, 0.017707583400351348, 0.0055468691578150365, 0.006226627212643027, 0.007747536703100838, 0.011179793234837825, 0.005958971446039433, 0.01522261218861711, 0.01234149818583409, 0.006939567054613731, 0.011446624014950488, 0.01425224643604159, 0.01425224643604159, 0.007863651265448652, 0.01092658612958275, 0.011728266833317774, 0.009458665266958812, 0.0047267705965676, 0.007983720419241114, 0.009458665266958812, 0.007318821255882091, 0.01234149818583409, 0.01425224643604159, 0.008651505658568908, 0.011179793234837825, 0.019343215722158646, 0.01068594098368534]
b3=[0.006408085596490774, 0.005639528324945938, 0.007727606133395791, 0.003483791410316366, 0.005776226504666211, 0.006591673732008659, 0.0030547615135828963, 0.006408085596490774, 0.00699755614101783, 0.005639528324945938, 0.00699755614101783, 0.0030888984616241413, 0.0029571836394084324, 0.003780503004936293, 0.004350728859646699, 0.004760989829108089, 0.005386891002256689, 0.002701717560270314, 0.006073566752968134, 0.006787701120878089, 0.005776226504666211, 0.008011901151903745, 0.0074654624188789254, 0.004853788755413074, 0.009444044480187383, 0.0028089914324832582, 0.0031597809735800448, 0.003951243718581427, 0.005776226504666211, 0.003021450310667881, 0.008011901151903745, 0.006408085596490774, 0.0035296073812259477, 0.005920667593939907, 0.0074654624188789254, 0.0074654624188789254, 0.00401206697216768, 0.005639528324945938, 0.006073566752968134, 0.004853788755413074, 0.002388003895140923, 0.004075023963987653, 0.004853788755413074, 0.0037271774914214356, 0.006408085596490774, 0.0074654624188789254, 0.004426351732291069, 0.005776226504666211, 0.010415577696546965, 0.005509938319712753]
b4=[0.0033322045101752034, 0.002913756301222066, 0.004067161122839888, 0.001771928389729875, 0.0029877033644825227, 0.0034331634020878417, 0.0015495167097884256, 0.0033322045101752034, 0.0036578134373502283, 0.002913756301222066, 0.0036578134373502283, 0.0015671617195004836, 0.0014991278172001078, 0.001926602492900226, 0.0022259543002843573, 0.0024431395175686242, 0.0027776156730386055, 0.001367536049025714, 0.003149256834872366, 0.003541409280458132, 0.0029877033644825227, 0.004228503385726976, 0.003919367769911436, 0.0024924861176445524, 0.005059309542957528, 0.0014227359203486626, 0.001603828221438356, 0.002015940672745626, 0.0029877033644825227, 0.001532306943267282, 0.004228503385726976, 0.0033322045101752034, 0.0017957651588693425, 0.003066060004004595, 0.003919367769911436, 0.003919367769911436, 0.002047825850377254, 0.002913756301222066, 0.003149256834872366, 0.0024924861176445524, 0.0012065703891238346, 0.0020808633007596525, 0.0024924861176445524, 0.0018987507975165793, 0.0033322045101752034, 0.003919367769911436, 0.0022658705296251893, 0.0029877033644825227, 0.005641771252296272, 0.00284383913275497]
chRf=[0.0405577807446967, 0.06641449553545449, 0.02306805074971174, 0.047419059892890386, 0.09089185719620509, 0.04174054373522466, 0.04661473355862317, 0.05726764268861345, 0.03471298868949559, 0.051085568326947724, 0.051443857603402234, 0.03684665374627262, 0.055990884035138515, 0.036241995303035776, 0.10471180054513395, 0.09239708761572231, 0.08091687679332239, 0.0582355135283455, 0.058069068634339686, 0.045819014891179934, 0.07234410631438239, 0.06981900994845985, 0.03422151653449491, 0.11688226683724912, 0.04848484848484858, 0.051068756722087105, 0.07743366310727189, 0.04586014645548315, 0.0634996124850279, 0.07175319608415949, 0.07414230848870655, 0.03359595631003787, 0.034374486796379375, 0.08928438799762335, 0.07161210402139616, 0.043231750531537994, 0.037393779662202524, 1.0000000000000001e-16, 0.08811942270605329, 0.04495908099908294, 0.04721302727733167, 0.03741771007678261, 0.043256155140500165, 0.0401552310889032, 0.05571731310325185, 0.027952480782669552, 0.07573464988257672, 0.08289321652906458, 0.042735042735042826, 0.020675430364961306]

# Python3 Program to find correlation coefficient


# Utility Function to print
# a Vector
def printVector(X):
	print(*X)

# Function returns the rank vector
# of the set of observations


def rankify(X):

	N = len(X)

	# Rank Vector
	Rank_X = [None for _ in range(N)]

	for i in range(N):

		r = 1
		s = 1

		# Count no of smaller elements
		# in 0 to i-1
		for j in range(i):
			if (X[j] < X[i]):
				r += 1
			if (X[j] == X[i]):
				s += 1

		# Count no of smaller elements
		# in i+1 to N-1
		for j in range(i+1, N):
			if (X[j] < X[i]):
				r += 1
			if (X[j] == X[i]):
				s += 1

		# Use Fractional Rank formula
		# fractional_rank = r + (n-1)/2
		Rank_X[i] = r + (s-1) * 0.5

	# Return Rank Vector
	return Rank_X


# function that returns
# Pearson correlation coefficient.
def correlationCoefficient(X, Y):
	n = len(X)
	sum_X = 0
	sum_Y = 0
	sum_XY = 0
	squareSum_X = 0
	squareSum_Y = 0

	for i in range(n):

		# sum of elements of array X.
		sum_X = sum_X + X[i]

		# sum of elements of array Y.
		sum_Y = sum_Y + Y[i]

		# sum of X[i] * Y[i].
		sum_XY = sum_XY + X[i] * Y[i]

		# sum of square of array elements.
		squareSum_X = squareSum_X + X[i] * X[i]
		squareSum_Y = squareSum_Y + Y[i] * Y[i]

	# use formula for calculating
	# correlation coefficient.
	corr = (n * sum_XY - sum_X * sum_Y) / ((n * squareSum_X -
											sum_X * sum_X) * (n * squareSum_Y - sum_Y * sum_Y)) ** 0.5

	return corr


# Driver function
X = b1
Y =humval

# Get ranks of vector X
rank_x = rankify(X)

# Get ranks of vector y
rank_y = rankify(Y)

print("Vector X")
printVector(X)

# Print rank vector of X
print("Rankings of X")
printVector(rank_x)

# Print Vector Y
print("Vector Y")
printVector(Y)

# Print rank vector of Y
print("Rankings of Y")
printVector(rank_y)

# Print Spearmans coefficient
print("Spearman's Rank correlation: ")
print(correlationCoefficient(rank_x, rank_y))


# This code is contributed by phasing17

def correlationCoefficient(X, Y):
	n = len(X)
	sum_X = 0
	sum_Y = 0
	sum_XY = 0
	squareSum_X = 0
	squareSum_Y = 0

	for i in range(n):

		# sum of elements of array X.
		sum_X = sum_X + X[i]

		# sum of elements of array Y.
		sum_Y = sum_Y + Y[i]

		# sum of X[i] * Y[i].
		sum_XY = sum_XY + X[i] * Y[i]

		# sum of square of array elements.
		squareSum_X = squareSum_X + X[i] * X[i]
		squareSum_Y = squareSum_Y + Y[i] * Y[i]

	# use formula for calculating
	# correlation coefficient.
	corr = (n * sum_XY - sum_X * sum_Y) / ((n * squareSum_X -
											sum_X * sum_X) * (n * squareSum_Y - sum_Y * sum_Y)) ** 0.5

	return corr


# Driver function
X = b2
Y =humval

# Get ranks of vector X
rank_x = rankify(X)

# Get ranks of vector y
rank_y = rankify(Y)

print("Vector X")
printVector(X)

# Print rank vector of X
print("Rankings of X")
printVector(rank_x)

# Print Vector Y
print("Vector Y")
printVector(Y)

# Print rank vector of Y
print("Rankings of Y")
printVector(rank_y)

# Print Spearmans coefficient
print("Spearman's Rank correlation: ")
print(correlationCoefficient(rank_x, rank_y))



def correlationCoefficient(X, Y):
	n = len(X)
	sum_X = 0
	sum_Y = 0
	sum_XY = 0
	squareSum_X = 0
	squareSum_Y = 0

	for i in range(n):

		# sum of elements of array X.
		sum_X = sum_X + X[i]

		# sum of elements of array Y.
		sum_Y = sum_Y + Y[i]

		# sum of X[i] * Y[i].
		sum_XY = sum_XY + X[i] * Y[i]

		# sum of square of array elements.
		squareSum_X = squareSum_X + X[i] * X[i]
		squareSum_Y = squareSum_Y + Y[i] * Y[i]

	# use formula for calculating
	# correlation coefficient.
	corr = (n * sum_XY - sum_X * sum_Y) / ((n * squareSum_X -
											sum_X * sum_X) * (n * squareSum_Y - sum_Y * sum_Y)) ** 0.5

	return corr


# Driver function
X = b3
Y =humval

# Get ranks of vector X
rank_x = rankify(X)

# Get ranks of vector y
rank_y = rankify(Y)

print("Vector X")
printVector(X)

# Print rank vector of X
print("Rankings of X")
printVector(rank_x)

# Print Vector Y
print("Vector Y")
printVector(Y)

# Print rank vector of Y
print("Rankings of Y")
printVector(rank_y)

# Print Spearmans coefficient
print("Spearman's Rank correlation: ")
print(correlationCoefficient(rank_x, rank_y))



def correlationCoefficient(X, Y):
	n = len(X)
	sum_X = 0
	sum_Y = 0
	sum_XY = 0
	squareSum_X = 0
	squareSum_Y = 0

	for i in range(n):

		# sum of elements of array X.
		sum_X = sum_X + X[i]

		# sum of elements of array Y.
		sum_Y = sum_Y + Y[i]

		# sum of X[i] * Y[i].
		sum_XY = sum_XY + X[i] * Y[i]

		# sum of square of array elements.
		squareSum_X = squareSum_X + X[i] * X[i]
		squareSum_Y = squareSum_Y + Y[i] * Y[i]

	# use formula for calculating
	# correlation coefficient.
	corr = (n * sum_XY - sum_X * sum_Y) / ((n * squareSum_X -
											sum_X * sum_X) * (n * squareSum_Y - sum_Y * sum_Y)) ** 0.5

	return corr


# Driver function
X = b4
Y =humval

# Get ranks of vector X
rank_x = rankify(X)

# Get ranks of vector y
rank_y = rankify(Y)

print("Vector X")
printVector(X)

# Print rank vector of X
print("Rankings of X")
printVector(rank_x)

# Print Vector Y
print("Vector Y")
printVector(Y)

# Print rank vector of Y
print("Rankings of Y")
printVector(rank_y)

# Print Spearmans coefficient
print("Spearman's Rank correlation: ")
print(correlationCoefficient(rank_x, rank_y))



def correlationCoefficient(X, Y):
	n = len(X)
	sum_X = 0
	sum_Y = 0
	sum_XY = 0
	squareSum_X = 0
	squareSum_Y = 0

	for i in range(n):

		# sum of elements of array X.
		sum_X = sum_X + X[i]

		# sum of elements of array Y.
		sum_Y = sum_Y + Y[i]

		# sum of X[i] * Y[i].
		sum_XY = sum_XY + X[i] * Y[i]

		# sum of square of array elements.
		squareSum_X = squareSum_X + X[i] * X[i]
		squareSum_Y = squareSum_Y + Y[i] * Y[i]

	# use formula for calculating
	# correlation coefficient.
	corr = (n * sum_XY - sum_X * sum_Y) / ((n * squareSum_X -
											sum_X * sum_X) * (n * squareSum_Y - sum_Y * sum_Y)) ** 0.5

	return corr


# Driver function
X = chRf
Y =humval

# Get ranks of vector X
rank_x = rankify(X)

# Get ranks of vector y
rank_y = rankify(Y)

print("Vector X")
printVector(X)

# Print rank vector of X
print("Rankings of X")
printVector(rank_x)

# Print Vector Y
print("Vector Y")
printVector(Y)

# Print rank vector of Y
print("Rankings of Y")
printVector(rank_y)

# Print Spearmans coefficient
print("Spearman's Rank correlation: ")
print(correlationCoefficient(rank_x, rank_y))