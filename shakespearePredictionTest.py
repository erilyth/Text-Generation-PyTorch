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

length_lim = 150
all_names = loadShakespeare()
save_file = 'shakespeare-model-checkpoint.dat'

rnn = vanillaLargeRNN(characterSetSize(), 
    256, 
    characterSetSize())

if os.path.isfile(save_file):
    rnn.load_state_dict(torch.load(save_file))

# Paragraphs start with a capital letter
def generateName(start_char='M'):
    input = Variable(inputVector(start_char))
    hidden = rnn.initHidden()
    final_text = start_char

    for i in range(length_lim):
        output, hidden = rnn(input[0], hidden)
        topvec, topidx = output.data.topk(1)
        topidx = topidx[0][0]
        if topidx == characterSetSize() - 1:
            break
        else:
            letter = characterSet()[topidx]
            final_text += letter
        input = Variable(inputVector(letter))

    return final_text

# Generate 20 text segments randomly
for i in range(20):
    # Start with a capital letter
    start_char = random.choice(string.letters)
    print "GENERATION", i, ", starting with", start_char
    print generateName(start_char)