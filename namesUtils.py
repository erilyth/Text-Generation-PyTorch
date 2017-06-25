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
from processData import loadNames, characterSetSize, characterSet

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
    # Target has to be a longTensor of indices for NLLLoss
    target_vec = []
    for char_idx in range(1,len(line)):
        char_cur = line[char_idx]
        target_vec.append(all_chars.find(char_cur))
    # Add an EOF as the final target    
    target_vec.append(characterSetSize()-1)
    return torch.LongTensor(target_vec)