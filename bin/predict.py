import time
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import random
from process_data import *
from my_decoder import *
from my_encoder import *
from utils import *
from tqdm import tqdm


MAX_LENGTH = 50
def evaluate(input_seq, output_seq, encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_seq, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size)#, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]])#, device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        #decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length): #, decoder_attention
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)#, encoder_outputs)
            #decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_seq.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words#, decoder_attentions[:di + 1]
    
def evaluateRandomly(input_seq, output_seq, test_pairs, encoder, decoder, n=10, max_length=MAX_LENGTH):
    for i in range(n):
        pair = random.choice(test_pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words = evaluate(input_seq,output_seq, encoder, decoder, pair[0])#output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')
        
def compute_test_accuracy(input_seq, output_seq, test_pairs, encoder, decoder, max_length=MAX_LENGTH):
    correct, total = 0, 0
    for i in tqdm(range(len(test_pairs))):
        pair = random.choice(test_pairs)
        seq, target = pair[0], pair[1]
        target_words = target.split(' ')
        output_words = evaluate(input_seq, output_seq, encoder, decoder, seq)
        total += max(len(output_words), len(target_words))
        if max(len(output_words), len(target_words)) == len(target_words):
            total -= 1
        for j in range(min(len(output_words), len(target_words))):
            if output_words[j] == target_words[j]:
                correct += 1
    return 'test size of  ' + str(len(test_pairs)) + '   accuracy is ::   ' + str(100 * (1.0 * correct / total)) + '%'