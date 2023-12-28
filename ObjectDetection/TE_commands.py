# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 01:08:37 2023

@author: linhai
"""

import os

os.system("python train_TE.py  --device 2 --epsilon_train 2")
os.system("python train_TE.py  --device 2 --epsilon_train 4")
os.system("python train_TE.py  --device 2 --epsilon_train 6")