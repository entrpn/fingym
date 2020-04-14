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
from fingym import fingym
from collections import deque

import matplotlib.pyplot as plt

import ray

import os

ray.init()

env = fingym.make('SPY-Daily-v0')

CONFIG = {
    'env_name': 'SPY-Daily-v0',
    # removing time frame, stocks owned and cash in hand
    'state_size': env.state_dim - 3,
    'max_shares_to_trade_at_once': 100,
    'time_frame': 30,
    'sigma': 0.1,
    'learning_rate': 0.03,
    'population_size': 400,
    'iterations': 50,
    'train': False,
    'eval': True,
    'log_actions': True
}

def get_state_as_change_percentage(state, next_state):
    open = (next_state[2] - state[2]) / next_state[2]
    high = (next_state[3] - state[3]) / next_state[3]
    low = (next_state[4] - state[4]) / next_state[4]
    close = (next_state[5] - state[5]) / next_state[5]
    volume = (next_state[6] - state[6]) / next_state[6]
    return [open, high, low, close, volume]

@ray.remote
def reward_function(weights):
    time_frame = CONFIG['time_frame']
    state_size = CONFIG['state_size']
    model = Model(time_frame * state_size, 500, 3)
    model.set_weights(weights)
    agent = Agent(model,state_size, time_frame)
    _,_,_,reward = run_agent(agent)
    print('reward: ',reward)
    return reward
    

def run_agent(agent):
    env = fingym.make(CONFIG['env_name'])
    log_actions = CONFIG['log_actions']
    state = env.reset()
    # Removed time element from state
    state = np.delete(state, 2)

    next_state, reward, done, info = env.step([0,0])
    if len(next_state) > agent.state_size:
        next_state = np.delete(next_state, 2)
    state_as_percentages = get_state_as_change_percentage(state,next_state)
    state = next_state

    done = False
    states_buy = []
    states_sell = []
    closes = []

    i = 0
    while not done:
        closes.append(state[5])
        action = agent.act(state_as_percentages)
        #if log_actions:
            #print('action: ',action)
            #print('state: ',state)
        next_state, reward, done, info = env.step(action)
        if len(next_state) > agent.state_size:
            next_state = np.delete(next_state, 2)
        if action[0] == 1 and action[1] > 0 and state[1] > state[2]:
            if log_actions:
                print('stocks owned: ',state[0])
                print('stocks to buy: ',action[1])
                print('stock price: ',state[2])
                print('cash in hand: ',state[1])
                print('total value: ',info['cur_val'])
            states_buy.append(i)
        if action[0] == 2 and action[1] > 0 and state[0] > 0:
            if log_actions:
                print('stocks owned: ',state[0])
                print('stocks to sell: ',action[1])
                print('stock price: ',state[2])
                print('cash in hand: ',state[1])
                print('total value: ',info['cur_val'])
            states_sell.append(i)
        state_as_percentages = get_state_as_change_percentage(state, next_state)
        state = next_state
        i+=1
    return closes, states_buy, states_sell, info['cur_val']

class Deep_Evolution_Strategy:
    def __init__(self, weights):
        self.weights = weights
        self.population_size = CONFIG['population_size']
        self.sigma = CONFIG['sigma']
        self.learning_rate = CONFIG['learning_rate']
    
    def _get_weight_from_population(self,weights, population):
        weights_population = []
        for index, i in enumerate(population):
            jittered = self.sigma * i
            weights_population.append(weights[index] + jittered)
        return weights_population
    
    def get_weights(self):
        return self.weights

    def train(self,epoch = 500, print_every=1):
        for i in range(epoch):
            population = []
            rewards = np.zeros(self.population_size)
            for k in range(self.population_size):
                x = []
                for w in self.weights:
                    x.append(np.random.randn(*w.shape))
                population.append(x)
            
            futures = [reward_function.remote(self._get_weight_from_population(self.weights,population[k])) for k in range(self.population_size)]

            rewards = ray.get(futures)
            
            rewards = (rewards - np.mean(rewards)) / np.std(rewards)
            for index, w in enumerate(self.weights):
                A = np.array([p[index] for p in population])
                self.weights[index] = (
                    w + self.learning_rate / (self.population_size * self.sigma) * np.dot(A.T, rewards).T
                )
            
            if (i + 1) % print_every == 0:
                print('iter: {}. standard reward: {}'.format(i+1,ray.get(reward_function.remote((self.weights)))))

class Agent:
    def __init__(self, model, state_size, time_frame):
        self.model = model
        self.time_frame = time_frame
        self.state_size = state_size
        self.state_fifo = deque(maxlen=self.time_frame)
        self.max_shares_to_trade_at_once = CONFIG['max_shares_to_trade_at_once']
        self.des = Deep_Evolution_Strategy(self.model.get_weights())
    
    def act(self,state):
        self.state_fifo.append(state)
        # do nothing for the first time frames until we can start the prediction
        if len(self.state_fifo) < self.time_frame:
            return np.zeros(2)
        
        state = np.array(list(self.state_fifo))
        state = np.reshape(state,(self.state_size*self.time_frame,1))
        #print(state)
        decision, buy = self.model.predict(state.T)
        # print('decision: ', decision)
        # print('buy: ', buy)

        return [np.argmax(decision[0]), min(self.max_shares_to_trade_at_once,max(int(buy[0]),0))]
    
    def fit(self, iterations, checkpoint):
        self.des.train(iterations, print_every = checkpoint)

class Model:
    def __init__(self, input_size, layer_size, output_size):
        self.weights = [
            np.random.randn(input_size, layer_size),
            np.random.randn(layer_size, output_size),
            np.random.randn(layer_size, 1),
            np.random.randn(1, layer_size)
        ]
    
    def predict(self, inputs):
        feed = np.dot(inputs, self.weights[0]) + self.weights[-1]
        decision = np.dot(feed, self.weights[1])
        buy = np.dot(feed, self.weights[2])
        return decision, buy
    
    def get_weights(self):
        return self.weights
    
    def set_weights(self, weights):
        self.weights = weights

if __name__ == '__main__':

    train = False
    eval = False

    time_frame = CONFIG['time_frame']
    state_size = CONFIG['state_size']
    
    model = Model(time_frame * state_size, 500, 3)

    dirname = os.path.dirname(__file__)
    weights_file = os.path.join(dirname,'deep_evo_weights.npy')

    if os.path.exists(weights_file):
        print('loading weights')
        weights = np.load(weights_file,allow_pickle=True)
        model.set_weights(weights)

    # np.save(weights_file,model.get_weights())
    agent = Agent(model,state_size, time_frame)
    if CONFIG['train']:
        agent.fit(iterations=CONFIG['iterations'], checkpoint=10)
        agent.model.set_weights(agent.des.get_weights())
        np.save(weights_file, agent.des.get_weights())
    
    if CONFIG['eval']:
        closes, states_buy, states_sell, result = run_agent(agent)
        print('result: {}'.format(str(result)))
        plt.figure(figsize = (20, 10))
        plt.plot(closes, label = 'true close', c = 'g')
        plt.plot(closes, 'X', label = 'predict buy', markevery = states_buy, c = 'b')
        plt.plot(closes, 'o', label = 'predict sell', markevery = states_sell, c = 'r')
        plt.legend()
        plt.show()