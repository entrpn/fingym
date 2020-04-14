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

from fingym.envs.amzn_envs import AmznDailyEnv

import pytest

@pytest.fixture
def amzn_daily_v0_env():
    return AmznDailyEnv()

def test_make_amzn_daily_v0_env(amzn_daily_v0_env):
    assert type(amzn_daily_v0_env) == AmznDailyEnv

def test_amzn_daily_v0_file_location(amzn_daily_v0_env):
    assert 'filtered_amzn_data' in amzn_daily_v0_env._get_data_file()