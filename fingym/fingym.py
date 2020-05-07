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

from fingym.logger import create_logger
logger = create_logger("fingym")

import traceback

from fingym.envs.spy_envs import DailySpyEnv, IntradaySpyEnv, SpyDailyRandomWalkEnv
from fingym.envs.tsla_envs import TslaDailyEnv, TslaDailyRandomWalkEnv
from fingym.envs.googl_envs import GooglDailyEnv, GooglDailyRandomWalkEnv
from fingym.envs.cgc_envs import CgcDailyEnv, CgcDailyRandomWalkEnv
from fingym.envs.cron_envs import CronDailyEnv, CronDailyRandomWalkEnv
from fingym.envs.ba_envs import BaDailyEnv, BaDailyRandomWalkEnv
from fingym.envs.amzn_envs import AmznDailyEnv, AmznDailyRandomWalkEnv
from fingym.envs.amd_envs import AmdDailyEnv, AmdDailyRandomWalkEnv
from fingym.envs.abbv_envs import AbbvDailyEnv, AbbvDailyRandomWalkEnv
from fingym.envs.aapl_envs import AaplDailyEnv, ApplDailyRandomWalkEnv

alphavantage_exc_info = None
try:
    from fingym.envs.alphavantage_envs import AlphavantageDailyEnv, AlphavantageDailyRandomWalkEnv
except:
    alphavantage_exc_info = sys.exc_info()
    pass

iex_exc_info = None
try:
    from fingym.envs.iex_cloud_envs import IEXCloudDailyEnv
except:
    iex_exc_info = sys.exc_info()
    pass

def make(envName, stock_symbol = None, alphavantage_key = None, no_days_to_random_walk=222, iex_token = None, iex_start = None, iex_end = None):
    if envName == 'SPY-Daily-v0':
        return DailySpyEnv()
    if envName == 'SPY-Minute-v0':
        return IntradaySpyEnv()
    if envName == 'SPY-Daily-Random-Walk':
        return SpyDailyRandomWalkEnv(no_days_to_random_walk)
    if envName == 'TSLA-Daily-v0':
        return TslaDailyEnv()
    if envName == 'TSLA-Daily-Random-Walk':
        return TslaDailyRandomWalkEnv(no_days_to_random_walk)
    if envName == 'GOOGL-Daily-v0':
        return GooglDailyEnv()
    if envName == 'GOOGL-Daily-Random-Walk':
        return GooglDailyRandomWalkEnv(no_days_to_random_walk)
    if envName == 'CGC-Daily-v0':
        return CgcDailyEnv()
    if envName == 'CGC-Daily-Random-Walk':
        return CgcDailyRandomWalkEnv(no_days_to_random_walk)
    if envName == 'CRON-Daily-v0':
        return CronDailyEnv()
    if envName == 'CRON-Daily-Random-Walk':
        return CronDailyRandomWalkEnv(no_days_to_random_walk)
    if envName == 'BA-Daily-v0':
        return BaDailyEnv()
    if envName == 'BA-Daily-Random-Walk':
        return BaDailyRandomWalkEnv(no_days_to_random_walk)
    if envName == 'AMZN-Daily-v0':
        return AmznDailyEnv()
    if envName == 'AMZN-Daily-Random-Walk':
        return AmznDailyRandomWalkEnv(no_days_to_random_walk)
    if envName == 'AMD-Daily-v0':
        return AmdDailyEnv()
    if envName == 'AMD-Daily-Random-Walk':
        return AmdDailyRandomWalkEnv(no_days_to_random_walk)
    if envName == 'ABBV-Daily-v0':
        return AbbvDailyEnv()
    if envName == 'ABBV-Daily-Random-Walk':
        return AbbvDailyRandomWalkEnv(no_days_to_random_walk)
    if envName == 'AAPL-Daily-v0':
        return AaplDailyEnv()
    if envName == 'AAPL-Daily-Random-Walk':
        return ApplDailyRandomWalkEnv(no_days_to_random_walk)

    try:
        if envName == 'Alphavantage-Daily-v0':
            return AlphavantageDailyEnv(stock_symbol, alphavantage_key)
        if envName == 'Alphavantage-Daily-Random-Walk':
            return AlphavantageDailyRandomWalkEnv(stock_symbol, alphavantage_key, no_days_to_random_walk)
    except:
        if alphavantage_exc_info:
            traceback.print_exception(*alphavantage_exc_info)
        else:
            traceback.print_exc()

    if envName == 'IEXCloud-Daily-v0':
        try:
            return IEXCloudDailyEnv(stock_symbol, iex_start, iex_end, iex_token)
        except:
            if iex_exc_info:
                traceback.print_exception(*iex_exc_info)
            else:
                traceback.print_exc()
    else:
        raise NotImplementedError