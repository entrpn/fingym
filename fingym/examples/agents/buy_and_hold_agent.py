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

from __future__ import division
import argparse
from fingym import fingym

class BuyAndHoldAgent(object):
    def __init__(self, action_space):
        self.bought_yet = False
        self.action_space = action_space
    
    def act(self, observation, reward, done):
        if not self.bought_yet:
            
            cash_in_hand = observation[1]
            close_price = observation[6]
            num_shares_to_buy = cash_in_hand / close_price
            print('will buy {} shares'.format(num_shares_to_buy))
            self.bought_yet = True
            
            return [1,num_shares_to_buy]
        else:
            return [0,0]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='SPY-Daily-v0', help='Select the environment to run')
    args = parser.parse_args()

    env = fingym.make(args.env_id)
    agent = BuyAndHoldAgent(env.action_space)

    reward = 0
    done = False

    cur_val = 0

    ob = env.reset()
    initial_value = ob[1]
    
    while True:
        action = agent.act(ob, reward, done)
        ob, reward, done, info = env.step(action)
        if done:
            cur_val = info['cur_val']
            break
    
    print('initial value: {}'.format(initial_value))
    print('final value: {}'.format(cur_val))