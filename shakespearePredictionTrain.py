#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 vishalapr <vishalapr@vishal-Lenovo-G50-70>
#
# Distributed under terms of the MIT license.

"""

"""

import random
import os
import math

import torch
import torch.nn as nn
from torch.autograd import Variable

from textUtils import inputVector, targetVector
from processData import loadShakespeare, characterSetSize, characterSet
from vanillaLargeRNN import vanillaLargeRNN
from plotLosses import plotLosses

uses_cuda = True

all_shakespeare = loadShakespeare()
all_chars = characterSet()

# Use negative log likelyhood for the loss
criterion = nn.CrossEntropyLoss()
learning_rate = 0.0005

rnn = vanillaLargeRNN(characterSetSize(), 256, characterSetSize())
if uses_cuda:
    rnn = rnn.cuda()

iters = 100000
save_iter = 10000
plot_iter = 250
total_losses = []
total_loss = 0
save_file = 'shakespeare-model-checkpoint.dat'
if uses_cuda:
    save_file = 'shakespeare-model-checkpoint-cuda.dat'

if os.path.isfile(save_file):
    rnn.load_state_dict(torch.load(save_file))

def train(input_line_vec, target_line_vec):
    hidden1, hidden2, hidden3 = rnn.initHidden()
    if uses_cuda:
        hidden1 = hidden1.cuda()
        hidden2 = hidden2.cuda()
        hidden3 = hidden3.cuda()
        input_line_vec = input_line_vec.cuda()
        target_line_vec = target_line_vec.cuda()
    rnn.zero_grad()
    loss = 0
    for i in range(min(input_line_vec.size()[0], 150)):
        output, hidden1, hidden2, hidden3 = rnn(input_line_vec[i], hidden1, hidden2, hidden3)
        loss += criterion(output, target_line_vec[i])
    loss.backward()

    for param in rnn.parameters():
        param.data.add_(-learning_rate, param.grad.data)

    # Return the final predicted output vector and the per char loss
    return output, loss.data[0] / input_line_vec.size()[0]


# Category and class represent the same and are used interchangeably 
for iter_idx in range(iters):
    line = all_shakespeare[random.randint(0, len(all_shakespeare)-1)]
    input_line_vec = Variable(inputVector(line))
    target_line_vec = Variable(targetVector(line))
    output, loss = train(input_line_vec, target_line_vec)
    if not math.isnan(loss):
        total_loss += loss

    if iter_idx % save_iter == 0:
        print "Saving model"
        torch.save(rnn.state_dict(), save_file)

    if iter_idx % plot_iter == plot_iter-1:
        total_losses.append(total_loss / plot_iter)
        print "Iter idx:", iter_idx, "Current loss:", (total_loss / plot_iter)
        total_loss = 0
        #plotLosses(total_losses)
