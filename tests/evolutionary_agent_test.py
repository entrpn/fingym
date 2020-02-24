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

import pytest

from gym.examples.agents.evolutionary_agent import EvoAgent, create_random_agents

@pytest.fixture
def env():
    return 

@pytest.fixture
def agent():
    return EvoAgent()

def test_create_random_agents():
    num_agents = 500
    state_size = 5
    time_frame = 30
    agents = create_random_agents(num_agents,state_size,time_frame)
    assert len(agents) == num_agents
    agents[0].state_size == state_size
    agents[0].time_frame == time_frame
    agents[0].action_size == 7




