# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 19:05:33 2023

@author: linhai
"""

import os

os.system("python train_AT.py --cuda_id 0 --epsilon 2 --tag AT2")
os.system("python train_AT.py --cuda_id 0 --epsilon 4 --tag AT4")
os.system("python train_AT.py --cuda_id 0 --epsilon 6 --tag AT6")
