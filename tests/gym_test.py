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

from gym.gym import make, DailySpyEnv, IntradaySpyEnv

import pytest

@pytest.fixture
def spy_daily_v0_env():
    return make('SPY-Daily-v0')

@pytest.fixture
def spy_intraday_v0_env():
    return make('SPY-Minute-v0')

def test_make_spy_daily_v0(spy_daily_v0_env):
    assert type(spy_daily_v0_env) == DailySpyEnv

def test_make_spy_intraday_v0(spy_intraday_v0_env):
    assert type(spy_intraday_v0_env) == IntradaySpyEnv