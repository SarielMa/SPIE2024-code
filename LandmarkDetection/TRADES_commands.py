# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 19:05:33 2023

@author: linhai
"""

import os

os.system("python train_TRADES_L2.py --cuda_id 2 --epsilon 2 --tag TRADES2")
os.system("python train_TRADES_L2.py --cuda_id 2 --epsilon 4 --tag TRADES4")
os.system("python train_TRADES_L2.py --cuda_id 2 --epsilon 6 --tag TRADES6")
