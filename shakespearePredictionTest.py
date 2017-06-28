#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 vishalapr <vishalapr@vishal-Lenovo-G50-70>
#
# Distributed under terms of the MIT license.

"""

"""

import os
import torch
import random
import string

from torch.autograd import Variable

from textUtils import inputVector
from processData import loadShakespeare, characterSetSize, characterSet
from vanillaLargeRNN import vanillaLargeRNN

uses_cuda = True

length_lim = 150
all_names = loadShakespeare()
save_file = 'shakespeare-model-checkpoint.dat'
if uses_cuda:
    save_file = 'shakespeare-model-checkpoint-cuda.dat'

rnn = vanillaLargeRNN(characterSetSize(), 
    256, 
    characterSetSize())
rnn.eval()

if uses_cuda:
    rnn = rnn.cuda()

if os.path.isfile(save_file):
    rnn.load_state_dict(torch.load(save_file))

# Paragraphs start with a capital letter
def generateName(start_char='M'):
    input = Variable(inputVector(start_char))
    hidden1, hidden2, hidden3 = rnn.initHidden()
    if uses_cuda:
        input = input.cuda()
        hidden1 = hidden1.cuda()
        hidden2 = hidden2.cuda()
        hidden3 = hidden3.cuda()
    final_text = start_char

    for i in range(length_lim):
        output, hidden1, hidden2, hidden3 = rnn(input[0], hidden1, hidden2, hidden3)
        topvec, topidx = output.data.topk(1)
        topidx = topidx[0][0]
        if topidx == characterSetSize() - 1:
            break
        else:
            letter = characterSet()[topidx]
            final_text += letter
        input = Variable(inputVector(letter))
        if uses_cuda:
            input = input.cuda()

    return final_text

# Generate 3 text segments randomly
for i in range(3):
    # Start with a capital letter
    start_char = random.choice(string.letters)
    print "GENERATION", i, ", starting with", start_char
    print generateName(start_char)