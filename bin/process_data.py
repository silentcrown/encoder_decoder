import unicodedata
import re
import random
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


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
            
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s



MAX_LENGTH = 50
'''
eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re ")
'''

def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH #and \ p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def readLangs(dir_, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open(dir_, encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_seq = Lang('input')
        output_seq = Lang('output')
    else:
        input_seq = Lang('input')
        output_seq = Lang('output')

    return input_seq, output_seq, pairs

def prepareData(dir_, reverse=False):
    input_seq, output_seq, pairs = readLangs(dir_, reverse)
    print("Read %s sentence pairs" % len(pairs))
    #pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_seq.addSentence(pair[0])
        output_seq.addSentence(pair[1])
    print("Counted words:")
    print(input_seq.name, input_seq.n_words)
    print(output_seq.name, output_seq.n_words)
    return input_seq, output_seq, pairs

def indexesFromSentence(seq, sentence):
    return [seq.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(seq, sentence):
    indexes = indexesFromSentence(seq, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long).view(-1, 1)#, device=device).view(-1, 1)


def tensorsFromPair(input_seq, output_seq,pair):
    input_tensor = tensorFromSentence(input_seq, pair[0])
    target_tensor = tensorFromSentence(output_seq, pair[1])
    return (input_tensor, target_tensor)


if __name__ == '__main__':
    input_seq, output_seq, pairs = prepareData('../data/tatoeba/eng-fra.txt', True)
    print(random.choice(pairs))