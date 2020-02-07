import pandas as pd
import numpy as np
import itertools
from datetime import datetime, timedelta
import re
import sys
import os
from .envs.spy_envs import DailySpyEnv, IntradaySpyEnv

default_data_file = os.path.join(os.path.dirname(__file__),'data/filtered_spy_2017_2019_all.csv')

def make(envName, data_file=default_data_file):
    if envName == 'SPY-Daily-v0':
        return DailySpyEnv()
    if envName == 'SPY-Minute-v0':
        return IntradaySpyEnv()