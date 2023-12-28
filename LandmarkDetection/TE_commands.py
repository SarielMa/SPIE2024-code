# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 19:05:33 2023

@author: linhai
"""

import os

os.system("python train_TE.py --cuda_id 3 --epsilon 2 --tag TE2")
os.system("python train_TE.py --cuda_id 3 --epsilon 4 --tag TE4")
os.system("python train_TE.py --cuda_id 3 --epsilon 6 --tag TE6")
