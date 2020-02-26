# Copyright 2020 The fingym Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd
import numpy as np
import itertools
from datetime import datetime, timedelta
import re
import sys
import os
from .envs.spy_envs import DailySpyEnv, IntradaySpyEnv
from .envs.tsla_envs import TslaDailyEnv
from .envs.googl_envs import GooglDailyEnv

default_data_file = os.path.join(os.path.dirname(__file__),'data/filtered_spy_2017_2019_all.csv')

def make(envName, data_file=default_data_file):
    if envName == 'SPY-Daily-v0':
        return DailySpyEnv()
    if envName == 'SPY-Minute-v0':
        return IntradaySpyEnv()
    if envName == 'TSLA-Daily-v0':
        return TslaDailyEnv()
    if envName == 'Googl-Daily-v0':
        return GooglDailyEnv()
    else:
        raise NotImplementedError