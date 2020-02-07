import pytest
from gym.spaces.space import BuyHoldSellSpace

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