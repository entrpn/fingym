from gym.gym import make, DailySpyEnv

import pytest

@pytest.fixture
def spy_daily_v0_env():
    return make('SPY-Daily-v0')

def test_make_spy_daily_v0(spy_daily_v0_env):
    assert type(spy_daily_v0_env) == DailySpyEnv