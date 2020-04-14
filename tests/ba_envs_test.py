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

from fingym.envs.ba_envs import BaDailyEnv

import pytest

@pytest.fixture
def ba_daily_v0_env():
    return BaDailyEnv()

def test_make_ba_daily_v0_env(ba_daily_v0_env):
    assert type(ba_daily_v0_env) == BaDailyEnv

def test_ba_daily_v0_file_location(ba_daily_v0_env):
    assert 'filtered_ba_data' in ba_daily_v0_env._get_data_file()