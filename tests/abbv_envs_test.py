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

from fingym.envs.abbv_envs import AbbvDailyEnv

import pytest

@pytest.fixture
def abbv_daily_v0_env():
    return AbbvDailyEnv()

def test_make_abbv_daily_v0_env(abbv_daily_v0_env):
    assert type(abbv_daily_v0_env) == AbbvDailyEnv

def test_amd_daily_v0_file_location(abbv_daily_v0_env):
    assert 'filtered_abbv_data' in abbv_daily_v0_env._get_data_file()