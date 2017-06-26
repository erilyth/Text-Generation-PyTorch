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
        self.input2hidden1 = nn.Linear(input_shape + hidden_shape, hidden_shape)
        self.input2output1 = nn.Linear(input_shape + hidden_shape, output_shape)
        self.output2output1 = nn.Linear(output_shape + hidden_shape, output_shape)
        self.dropout1 = nn.Dropout(0.3)
        self.input2hidden2 = nn.Linear(input_shape + hidden_shape, hidden_shape)
        self.input2output2 = nn.Linear(input_shape + hidden_shape, output_shape)
        self.output2output2 = nn.Linear(output_shape + hidden_shape, output_shape)
        self.dropout2 = nn.Dropout(0.3)
        self.input2hidden3 = nn.Linear(input_shape + hidden_shape, hidden_shape)
        self.input2output3 = nn.Linear(input_shape + hidden_shape, output_shape)
        self.output2output3 = nn.Linear(output_shape + hidden_shape, output_shape)
        self.dropout3 = nn.Dropout(0.3)

    # Predicts a character given the previous character and the hidden state as inputs
    # The forward pass needs to be defined, backward gradients are automatically calculated
    def forward(self, input, hidden1, hidden2, hidden3):
        input_comb1 = torch.cat([input, hidden1], 1)
        hidden1 = self.input2hidden1(input_comb1)
        output_temp1 = self.input2output1(input_comb1)
        output_comb1 = torch.cat([output_temp1, hidden1], 1)
        output1 = self.output2output1(output_comb1)
        output1 = self.dropout1(output1)
        input_comb2 = torch.cat([output1, hidden2], 1)
        hidden2 = self.input2hidden2(input_comb2)
        output_temp2 = self.input2output2(input_comb2)
        output_comb2 = torch.cat([output_temp2, hidden2], 1)
        output2 = self.output2output2(output_comb2)
        output2 = self.dropout2(output2)
        input_comb3 = torch.cat([output2, hidden3], 1)
        hidden3 = self.input2hidden3(input_comb3)
        output_temp3 = self.input2output3(input_comb3)
        output_comb3 = torch.cat([output_temp3, hidden3], 1)
        output3 = self.output2output3(output_comb3)
        output3 = self.dropout3(output3)
        return output3, hidden1, hidden2, hidden3

    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_shape)), Variable(torch.zeros(1, self.hidden_shape)), Variable(torch.zeros(1, self.hidden_shape))
