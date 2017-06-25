#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 vishalapr <vishalapr@vishal-Lenovo-G50-70>
#
# Distributed under terms of the MIT license.

"""

"""

from __future__ import unicode_literals
from io import open
import string
import unicodedata
import glob

letters = string.ascii_letters + " .;,'-"
letters_size = len(letters) + 1

def characterSetSize():
	return letters_size

def characterSet():
	return letters

def unicodeToAscii(cur_line):
	return ''.join(
		c for c in unicodedata.normalize('NFD', cur_line)
		if unicodedata.category(c) != 'Mn'
		and c in letters)

def readLines(file):
	lines = open(file, encoding='utf-8').read().strip()
	lines = lines.split('\n')
	lines = [unicodeToAscii(line) for line in lines]
	return lines

"""
Load the names dataset to generate new names
Data stored in the `names` folder with a text file for each language
"""
def loadNames():
	all_names = {}
	for filename in glob.glob('data/names/*.txt'): 
		category = filename.split('/')[-1].split('.')[0]
		lines = readLines(filename)
		all_names[category] = lines
	return all_names
