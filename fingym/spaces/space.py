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

import numpy as np

class Space(object):

    def __init__(self, shape):
        self.shape = shape
    
    def sample(self):
        raise NotImplementedError


class BuyHoldSellSpace(Space):
    def __init__(self):
        super().__init__(2)
    
    def sample(self,seed = None):
        if seed:
            np.random.seed(seed)
        action = np.zeros(self.shape)
        action[0] = np.random.randint(low=0,high=3)
        action[1] = np.random.randint(low=0,high=100)
        return action