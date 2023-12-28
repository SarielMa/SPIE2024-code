# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 22:33:56 2023

@author: linhai
"""

import os

os.system("python train_TRADES.py  --device 1 --epsilon_train 2")
os.system("python train_TRADES.py  --device 1 --epsilon_train 4")
os.system("python train_TRADES.py  --device 1 --epsilon_train 6")