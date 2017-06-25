#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 vishalapr <vishalapr@vishal-Lenovo-G50-70>
#
# Distributed under terms of the MIT license.

"""

"""

import torch
import torch.nn as nn
from torch.autograd import Variable

class categoryRNN(nn.Module):
    def __init__(self, input_shape, hidden_shape, output_shape, n_classes):
        super(categoryRNN, self).__init__()

        self.hidden_shape = hidden_shape
        self.input2hidden = nn.Linear(n_classes + input_shape + hidden_shape, hidden_shape)
        self.input2output = nn.Linear(n_classes + input_shape + hidden_shape, output_shape)
        self.output2output = nn.Linear(output_shape + hidden_shape, output_shape)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax()

    # Predicts a character given the previous character and the hidden state as inputs
    def forward(self, class_vec, input, hidden):
        input_comb = torch.cat([class_vec, input, hidden], 1)
        hidden = self.input2hidden(input_comb)
        output_temp = self.input2output(input_comb)
        output_comb = torch.cat([output_temp, hidden], 1)
        output = self.output2output(output_comb)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_shape))
