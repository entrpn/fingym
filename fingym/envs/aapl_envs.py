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

from fingym.envs.spy_envs import DailySpyEnv, SpyDailyRandomWalkEnv
import os

class ApplDailyRandomWalkEnv(SpyDailyRandomWalkEnv):
    def _get_data_file(self):
        return os.path.join(os.path.dirname(__file__),'..','data/filtered_aapl_data_10_yrs.csv')

class AaplDailyEnv(DailySpyEnv):
    def __init__(self):
        super().__init__()

    def _get_data_file(self):
        return os.path.join(os.path.dirname(__file__),'..','data/filtered_aapl_data_10_yrs.csv')