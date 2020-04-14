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

from fingym.envs.cron_envs import CronDailyEnv

import pytest

@pytest.fixture
def cron_daily_v0_env():
    return CronDailyEnv()

def test_make_cron_daily_v0_env(cron_daily_v0_env):
    assert type(cron_daily_v0_env) == CronDailyEnv

def test_cron_daily_v0_file_location(cron_daily_v0_env):
    assert 'filtered_cron_data' in cron_daily_v0_env._get_data_file()