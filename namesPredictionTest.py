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

from textUtils import class2Vector, inputVector
from processData import loadNames, characterSetSize, characterSet
from categoryRNN import categoryRNN

length_lim = 15
all_names = loadNames()
all_types = all_names.keys()
save_file = 'names-model-checkpoint.dat'

rnn = categoryRNN(characterSetSize(), 
    128, 
    characterSetSize(), 
    len(all_types))
rnn.eval()

if os.path.isfile(save_file):
    rnn.load_state_dict(torch.load(save_file))

# Names start with a capital letter
def generateName(category, start_char='A'):
    category_vec = Variable(class2Vector(category))
    input = Variable(inputVector(start_char))
    hidden = rnn.initHidden()
    final_name = start_char

    for i in range(length_lim):
        output, hidden = rnn(category_vec, input[0], hidden)
        topvec, topidx = output.data.topk(1)
        topidx = topidx[0][0]
        if topidx == characterSetSize() - 1:
            break
        else:
            letter = characterSet()[topidx]
            final_name += letter
        input = Variable(inputVector(letter))

    return final_name

# Generate 100 names randomly
for i in range(100):
    class_idx = random.randint(0, len(all_types)-1)
    start_char = random.choice(string.letters[26:])
    print all_types[class_idx], start_char, generateName(all_types[class_idx], start_char)