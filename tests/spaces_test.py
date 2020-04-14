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
from fingym.spaces.space import BuyHoldSellSpace

@pytest.fixture
def buy_hold_sell_space():
    return BuyHoldSellSpace()

def test_buy_and_hold_sell_shape(buy_hold_sell_space):
    assert buy_hold_sell_space.shape == 2

def test_buy_hold_sell_sample_hold(buy_hold_sell_space):
    sample_action = buy_hold_sell_space.sample(2)
    assert sample_action[0] == 0
    assert sample_action[1] == 15

def test_buy_hold_sell_sample_buy(buy_hold_sell_space):
    sample_action = buy_hold_sell_space.sample(30)
    assert sample_action[0] == 1
    assert sample_action[1] == 37

def test_buy_hold_sell_sample_sell(buy_hold_sell_space):
    sample_action = buy_hold_sell_space.sample(4)
    assert sample_action[0] == 2
    assert sample_action[1] == 46