#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 17:21:30 2018

@author: ninalopatina
"""

import os
import pandas as pd
pd.set_option("display.max_columns",30)


os.chdir('/Users/ninalopatina/ray_results/default/DQN_CartPole-v0_0_2018-06-20_12-48-43rltozctp')
df = pd.read_csv('progress.csv')
df2 = pd.read_json('result.json',lines=True)