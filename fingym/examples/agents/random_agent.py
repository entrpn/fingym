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

class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space
    
    def act(self, observation, reward, done):
        return self.action_space.sample()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='SPY-Daily-v0', help='Select the environment to run')
    args = parser.parse_args()

    env = fingym.make(args.env_id)
    agent = RandomAgent(env.action_space)

    episode_count = 100
    reward = 0
    done = False

    final_vals = []

    initial_value = 0

    for i in range(episode_count):
        ob = env.reset()
        initial_value = ob[1]
        while True:
            action = agent.act(ob, reward, done)
            ob, reward, done, info = env.step(action)
            if done:
                final_vals.append(info['cur_val'])
                break
    
    max_value = max(final_vals)
    min_value = min(final_vals)
    avg_value = sum(final_vals)/len(final_vals)
    print('initial value: {}'.format(initial_value))
    print('min_value: {}, avg_value: {}, max_value: {}'.format(min_value,avg_value,max_value))