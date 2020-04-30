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
from fingym.envs.spy_envs import DailySpyEnv, IntradaySpyEnv, SpyDailyRandomWalkEnv
from fingym.envs.tsla_envs import TslaDailyEnv
from fingym.envs.googl_envs import GooglDailyEnv
from fingym.envs.cgc_envs import CgcDailyEnv
from fingym.envs.cron_envs import CronDailyEnv
from fingym.envs.cgc_envs import CgcDailyEnv
from fingym.envs.ba_envs import BaDailyEnv
from fingym.envs.amzn_envs import AmznDailyEnv
from fingym.envs.amd_envs import AmdDailyEnv
from fingym.envs.abbv_envs import AbbvDailyEnv
from fingym.envs.aapl_envs import AaplDailyEnv

def make(envName):
    if envName == 'SPY-Daily-v0':
        return DailySpyEnv()
    if envName == 'SPY-Minute-v0':
        return IntradaySpyEnv()
    if envName == 'SPY-Daily-Random-Walk':
        return SpyDailyRandomWalkEnv()
    if envName == 'TSLA-Daily-v0':
        return TslaDailyEnv()
    if envName == 'GOOGL-Daily-v0':
        return GooglDailyEnv()
    if envName == 'CGC-Daily-v0':
        return CgcDailyEnv()
    if envName == 'CRON-Daily-v0':
        return CronDailyEnv()
    if envName == 'BA-Daily-v0':
        return BaDailyEnv()
    if envName == 'AMZN-Daily-v0':
        return AmznDailyEnv()
    if envName == 'AMD-Daily-v0':
        return AmdDailyEnv()
    if envName == 'ABBV-Daily-v0':
        return AbbvDailyEnv()
    if envName == 'AAPL-Daily-v0':
        return AaplDailyEnv()
    else:
        raise NotImplementedError