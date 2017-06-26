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

class vanillaLargeRNN(nn.Module):
    def __init__(self, input_shape, hidden_shape, output_shape):
        super(vanillaLargeRNN, self).__init__()

        self.hidden_shape = hidden_shape
        self.input2hidden = nn.Linear(input_shape + hidden_shape, hidden_shape)
        self.input2output = nn.Linear(input_shape + hidden_shape, output_shape)
        self.hidden2hidden = nn.Linear(hidden_shape, hidden_shape)
        self.output2output1 = nn.Linear(output_shape, output_shape)
        self.output2output2 = nn.Linear(output_shape + hidden_shape, output_shape)
        self.output2output3 = nn.Linear(output_shape, output_shape)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.1)
        self.dropout4 = nn.Dropout(0.1)

    # Predicts a character given the previous character and the hidden state as inputs
    # The forward pass needs to be defined, backward gradients are automatically calculated
    def forward(self, input, hidden):
        input_comb = torch.cat([input, hidden], 1)
        hidden = self.input2hidden(input_comb)
        hidden = self.hidden2hidden(hidden)
        hidden = self.dropout1(hidden)
        output_temp = self.input2output(input_comb)
        output_temp = self.output2output1(output_temp)
        output_temp = self.dropout2(output_temp)
        output_comb = torch.cat([output_temp, hidden], 1)
        output = self.output2output2(output_comb)
        output = self.dropout3(output)
        output = self.output2output3(output)
        output = self.dropout4(output)
        return output, hidden

    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_shape))
