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