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
import torch
import os
import torch.nn as nn

from processData import loadNames, characterSetSize, characterSet
from categoryRNN import categoryRNN
from plotLosses import plotLosses

all_names = loadNames()
all_types = all_names.keys()
all_chars = characterSet()

def class2Vector(class_name):
	class_idx = all_types.index(class_name)
	class_vec = torch.zeros(1, len(all_types))
	class_vec[0][class_idx] = 1
	return class_vec

def inputVector(line):
	input_vec = torch.zeros(len(line), 1, characterSetSize())
	for char_idx in range(len(line)):
		char_cur = line[char_idx]
		input_vec[char_idx][0][all_chars.find(char_cur)] = 1
	return input_vec

def targetVector(line):
	target_vec = torch.zeros(len(line), 1, characterSetSize())
	for char_idx in range(1,len(line)):
		char_cur = line[char_idx]
		target_vec[char_idx-1][0][all_chars.find(char_cur)] = 1
	# Add an EOF as the final target	
	target_vec[len(line)-1][0][characterSetSize()-1]	

# Use negative log likelyhood for the loss
criterion = nn.NLLLoss()
learning_rate = 0.0005

rnn = categoryRNN(characterSetSize(), 128, characterSetSize(), len(all_types))

def train(category_vec, input_line_vec, target_line_vec):
	hidden = rnn.initHidden()
	rnn.zero_grad()
	loss = 0
	for i in range(input_line_vec.size()[0]):
		output, hidden = rnn(category_vec, input_line_vec[i], hidden)
		loss += criterion(output, target_line_vec[i])
	loss.backward()

	for param in rnn.parameters():
		param.data.add_(-learning_rate, param.grad.data)

	# Return the final predicted output vector and the per char loss
	return output, loss.data[0] / input_line_vec.size()[0]

iters = 100000
print_iter = 5000
save_iter = 10000
plot_iter = 500
total_losses = []
total_loss = 0
save_file = 'names-model-checkpoint.dat'

if os.path.isfile(save_file):
	rnn.load_state_dict(torch.load(save_file))

# Category and class represent the same and are used interchangeably 
for iter_idx in range(iters):
	category_name = all_types[random.randint(0, len(all_types)-1)]
	line = all_names[category_name][random.randint(0, len(all_names[category_name])-1)]
	category_vec = Variable(class2Vector(category_name))
	input_line_vec = Variable(inputVector(line))
	target_line_vec = Variable(targetVector(line))
	output, loss = train(category_vec, input_line_vec, target_line_vec)
	total_loss += loss

	if iter_idx % save_iter == 0:
		print "Saving model"
		torch.save(rnn.state_dict(), save_file)

	if iter_idx % print_iter == 0:
		print "Iter idx:", iter_idx, "Current loss:", loss

	if iter_idx % plot_iter == 0:
		total_losses.append(total_loss / plot_iter)
		total_loss = 0
		plotLosses(total_losses)
