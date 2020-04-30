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

from fingym.envs.spy_envs import DailySpyEnv, IntradaySpyEnv, SpyDailyRandomWalkEnv

import pytest

@pytest.fixture
def spy_daily_v0_env():
    return DailySpyEnv()

@pytest.fixture
def spy_intraday_minute_v0_env():
    return IntradaySpyEnv()

@pytest.fixture
def spy_daily_random_walk_env():
    return SpyDailyRandomWalkEnv()

def test_make_spy_daily_random_walk_env(spy_daily_random_walk_env):
    assert type(spy_daily_random_walk_env) == SpyDailyRandomWalkEnv

def test_spy_daily_random_walk_init_values(spy_daily_random_walk_env):
    assert spy_daily_random_walk_env.cur_step == 0
    assert spy_daily_random_walk_env.n_step == len(spy_daily_random_walk_env.data)
    assert spy_daily_random_walk_env.initial_investment == 25000
    assert spy_daily_random_walk_env.cash_in_hand == spy_daily_random_walk_env.initial_investment
    assert spy_daily_random_walk_env.stock_owned == 0
    assert len(spy_daily_random_walk_env.headers) == 2
    assert spy_daily_random_walk_env.state_dim == 4
    assert spy_daily_random_walk_env._close_idx == 1
    assert spy_daily_random_walk_env._open_idx == 1

def test_spy_daily_random_walk_reset(spy_daily_random_walk_env):
    observation = spy_daily_random_walk_env.reset()
    assert spy_daily_random_walk_env.cur_step == 0
    assert spy_daily_random_walk_env.cash_in_hand == spy_daily_random_walk_env.initial_investment
    assert spy_daily_random_walk_env.stock_owned == 0
    assert len(observation) == spy_daily_random_walk_env.state_dim
    assert observation[0] == 0
    assert observation[1] == 25000
    assert observation[3] == 86.16

def test_spy_daily_random_walk_step(spy_daily_random_walk_env):
    spy_daily_random_walk_env.reset()
    done = False
    while not done:
        obs, reward, done, info = spy_daily_random_walk_env.step(spy_daily_random_walk_env.action_space.sample())
    
    assert info['original_close'] == 321.22
    

def test_make_spy_intraday_v0_env(spy_intraday_minute_v0_env):
    assert type(spy_intraday_minute_v0_env) == IntradaySpyEnv

def test_spy_intraday_minute_v0_init_values(spy_intraday_minute_v0_env):
    assert spy_intraday_minute_v0_env.cur_step == 0
    assert spy_intraday_minute_v0_env.n_step == len(spy_intraday_minute_v0_env.data)
    assert spy_intraday_minute_v0_env.initial_investment == 25000
    assert spy_intraday_minute_v0_env.cash_in_hand == spy_intraday_minute_v0_env.initial_investment
    assert spy_intraday_minute_v0_env.stock_owned == 0
    assert len(spy_intraday_minute_v0_env.headers) == 6
    assert spy_intraday_minute_v0_env.state_dim == 8
    assert spy_intraday_minute_v0_env._close_idx == 4
    assert spy_intraday_minute_v0_env._open_idx == 1

def test_spy_intraday_minute_v0_reset(spy_intraday_minute_v0_env):
    observation = spy_intraday_minute_v0_env.reset()
    assert spy_intraday_minute_v0_env.cur_step == 0
    assert spy_intraday_minute_v0_env.cash_in_hand == spy_intraday_minute_v0_env.initial_investment
    assert spy_intraday_minute_v0_env.stock_owned == 0
    assert len(observation) == spy_intraday_minute_v0_env.state_dim
    assert observation[0] == 0
    assert observation[1] == 25000
    assert observation[2] == 1483453800.0
    assert observation[3] == 225.04
    assert observation[4] == 225.12
    assert observation[5] == 224.93
    assert observation[6] == 224.95
    assert observation[7] == 774063

def test_make_spy_daily_v0(spy_daily_v0_env):
    assert type(spy_daily_v0_env) == DailySpyEnv

def test_spy_daily_v0_init_values(spy_daily_v0_env):
    assert spy_daily_v0_env.cur_step == 0
    assert spy_daily_v0_env.n_step == len(spy_daily_v0_env.data)
    assert spy_daily_v0_env.initial_investment == 25000
    assert spy_daily_v0_env.cash_in_hand == spy_daily_v0_env.initial_investment
    assert spy_daily_v0_env.stock_owned == 0
    assert len(spy_daily_v0_env.headers) == 6
    assert spy_daily_v0_env.state_dim == 8
    assert spy_daily_v0_env._close_idx == 4
    assert spy_daily_v0_env._open_idx == 1

def test_spy_daily_v0_reset(spy_daily_v0_env):
    observation = spy_daily_v0_env.reset()
    assert spy_daily_v0_env.cur_step == 0
    assert spy_daily_v0_env.cash_in_hand == spy_daily_v0_env.initial_investment
    assert spy_daily_v0_env.stock_owned == 0
    assert len(observation) == spy_daily_v0_env.state_dim
    assert observation[0] == 0
    assert observation[1] == 25000
    assert observation[2] == 1230019200.0
    assert observation[3] == 87.53
    assert observation[4] == 87.93
    assert observation[5] == 85.80
    assert observation[6] == 86.16
    assert observation[7] == 221772560

def test_spy_daily_v0_reset_after_action(spy_daily_v0_env):
    spy_daily_v0_env.reset()
    spy_daily_v0_env.step(spy_daily_v0_env.action_space.sample())
    observation = spy_daily_v0_env.reset()
    assert observation[0] == 0
    assert observation[1] == 25000
    assert observation[2] == 1230019200.0
    assert observation[3] == 87.53
    assert observation[4] == 87.93
    assert observation[5] == 85.80
    assert observation[6] == 86.16
    assert observation[7] == 221772560

# These are more SpyEnv testing common functionality. 

def test_spy_daily_v0_step(spy_daily_v0_env):
    spy_daily_v0_env.reset()
    obs, reward, done, info = spy_daily_v0_env.step(spy_daily_v0_env.action_space.sample())
    assert spy_daily_v0_env.cur_step == 1
    assert done == False
    assert info['cur_val'] == 25000
    assert reward == 0
    assert obs[0] == 0
    assert obs[1] == 25000
    assert obs[2] == 1230105600.0
    assert obs[3] == 86.45
    assert obs[4] == 86.87
    assert obs[5] == 86
    assert obs[6] == 86.66
    assert obs[7] == 62142416.0

def test_spy_daily_v0_buying_shares(spy_daily_v0_env):
    spy_daily_v0_env.reset()
    spy_daily_v0_env.step([1,10])
    assert spy_daily_v0_env.stock_owned == 10

def test_spy_daily_v0_selling_shares(spy_daily_v0_env):
    spy_daily_v0_env.reset()
    spy_daily_v0_env.step([1,10])
    spy_daily_v0_env.step([2,5])
    assert spy_daily_v0_env.stock_owned == 5

def test_cash_in_hand_reduced_when_buying_shares(spy_daily_v0_env):
    spy_daily_v0_env.reset()
    spy_daily_v0_env.step([1,10])
    assert spy_daily_v0_env.cash_in_hand == 24135.499999999993

def test_cash_in_hand_increased_when_selling_shares(spy_daily_v0_env):
    spy_daily_v0_env.reset()
    spy_daily_v0_env.step([1,10])
    spy_daily_v0_env.step([2,10])
    assert spy_daily_v0_env.cash_in_hand == 25007.899999999994

def test_cash_in_hand_increase_when_selling_shares_more_than_i_have(spy_daily_v0_env):
    spy_daily_v0_env.reset()
    spy_daily_v0_env.step([1,10])
    spy_daily_v0_env.step([2,11])
    assert spy_daily_v0_env.cash_in_hand == 25007.899999999994

def test_reward_from_step(spy_daily_v0_env):
    spy_daily_v0_env.reset()
    _, reward, _, _ = spy_daily_v0_env.step([1,10])
    assert reward == 2.099999999991269

def test_done(spy_daily_v0_env):
    spy_daily_v0_env.reset()
    done = False
    while not done:
        _, _, done, _ = spy_daily_v0_env.step(spy_daily_v0_env.action_space.sample())
    
    assert spy_daily_v0_env.cur_step == len(spy_daily_v0_env.data)-1

def test_done_doesnt_crash(spy_daily_v0_env):
    spy_daily_v0_env.reset()
    done = False
    while not done:
        _, _, done, _ = spy_daily_v0_env.step(spy_daily_v0_env.action_space.sample())
    
    spy_daily_v0_env.step(spy_daily_v0_env.action_space.sample())

    assert spy_daily_v0_env.cur_step == len(spy_daily_v0_env.data)-1