# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 19:50:19 2023

@author: linhai
"""

import os

os.system("python train_AT.py  --device 0 --epsilon_train 2")
os.system("python train_AT.py  --device 0 --epsilon_train 4")
os.system("python train_AT.py  --device 0 --epsilon_train 6")
