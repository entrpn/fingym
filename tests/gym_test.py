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

from fingym.fingym import make, DailySpyEnv, IntradaySpyEnv, GooglDailyEnv, CronDailyEnv, CgcDailyEnv, BaDailyEnv, AmznDailyEnv, AmdDailyEnv, AbbvDailyEnv, AaplDailyEnv, SpyDailyRandomWalkEnv

import pytest

@pytest.fixture
def spy_daily_v0_env():
    return make('SPY-Daily-v0')

@pytest.fixture
def spy_intraday_v0_env():
    return make('SPY-Minute-v0')

@pytest.fixture
def spy_daily_random_walk_env():
    return make('SPY-Daily-Random-Walk')

@pytest.fixture
def googl_daily_v0_env():
    return make('GOOGL-Daily-v0')

@pytest.fixture
def cron_daily_v0_env():
    return make('CRON-Daily-v0')

@pytest.fixture
def cgc_daily_v0_env():
    return make('CGC-Daily-v0')

@pytest.fixture
def ba_daily_v0_env():
    return make('BA-Daily-v0')

@pytest.fixture
def amzn_daily_v0_env():
    return make('AMZN-Daily-v0')

@pytest.fixture
def amd_daily_v0_env():
    return make('AMD-Daily-v0')

@pytest.fixture
def abbv_daily_v0_env():
    return make('ABBV-Daily-v0')

@pytest.fixture
def aapl_daily_v0_env():
    return make('AAPL-Daily-v0')

def test_make_spy_daily_v0(spy_daily_v0_env):
    assert type(spy_daily_v0_env) == DailySpyEnv

def test_make_spy_intraday_v0(spy_intraday_v0_env):
    assert type(spy_intraday_v0_env) == IntradaySpyEnv

def test_make_spy_daily_random_walk(spy_daily_random_walk_env):
    assert type(spy_daily_random_walk_env) == SpyDailyRandomWalkEnv

def test_make_googl_daily_v0(googl_daily_v0_env):
    assert type(googl_daily_v0_env) == GooglDailyEnv

def test_make_cron_daily_v0(cron_daily_v0_env):
    assert type(cron_daily_v0_env) == CronDailyEnv

def test_make_cgc_daily_v0(cgc_daily_v0_env):
    assert type(cgc_daily_v0_env) == CgcDailyEnv

def test_make_ba_daily_v0(ba_daily_v0_env):
    assert type(ba_daily_v0_env) == BaDailyEnv

def test_make_amzn_daily_v0(amzn_daily_v0_env):
    assert type(amzn_daily_v0_env) == AmznDailyEnv

def test_make_amd_daily_v0(amd_daily_v0_env):
    assert type(amd_daily_v0_env) == AmdDailyEnv

def test_make_abbv_daily_v0(abbv_daily_v0_env):
    assert type(abbv_daily_v0_env) == AbbvDailyEnv

def test_make_aapl_daily_v0(aapl_daily_v0_env):
    assert type(aapl_daily_v0_env) == AaplDailyEnv