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

ray.init()#(num_cpus=5)

CONFIG = {
    'env_name': 'SPY-Daily-Random-Walk',
    'max_shares_to_trade_at_once': 100,
    'time_frame': 10,
    'sigma': 0.1,
    'learning_rate': 0.01,
    'population_size': 250,
    'iterations': 5,
    'train': True,
    'eval': True,
    'log_actions': False,
    'layer_size' : 512,
    'layers' : 3,
    'days_to_random_walk' : 512
}
# removing time frame, stocks owned and cash in hand
env = fingym.make(CONFIG["env_name"],only_random_walk=True,no_days_to_random_walk=CONFIG['days_to_random_walk'])
CONFIG["state_size"] = env.state_dim - 3

def get_state_as_change_percentage(state, next_state):
    close = (next_state[2] - state[2]) / next_state[2]
    return close

@ray.remote
def reward_function(weights):
    time_frame = CONFIG['time_frame']
    state_size = CONFIG['state_size']
    layer_size = CONFIG['layer_size']
    model = Model(time_frame * state_size, layer_size, 3)
    model.set_weights(weights)
    agent = Agent(model,state_size, time_frame)
    _,_,_,reward = run_agent(agent)
    #print('reward: ',reward)
    return reward
    

def run_agent(agent,eval=False):
    if eval:
        env = fingym.make(CONFIG['env_name'], only_random_walk=True,no_days_to_random_walk=CONFIG['days_to_random_walk'])
    else:
        env = fingym.make(CONFIG['env_name'],only_random_walk=True,no_days_to_random_walk=CONFIG['days_to_random_walk'])
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
        closes.append(state[2])
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

        final_reward = env.buy_hold_reward - info['cur_val']

    return closes, states_buy, states_sell, final_reward#info['cur_val']

class Deep_Evolution_Strategy:
    def __init__(self, weights):
        self.weights = weights
        self.population_size = CONFIG['population_size']
        self.sigma = CONFIG['sigma']
        self.learning_rate = CONFIG['learning_rate']
    
    def _get_weight_from_population(self,weights, population):
        weights_population = []
        for index, i in enumerate(population):
            
            N = np.random.randn(i.shape[0],i.shape[1])
            jittered = self.sigma * N
            weights_population.append(weights[index] + jittered)
        return weights_population
    
    def get_weights(self):
        return self.weights

    def train(self,epoch = 500, print_every=1):
        for i in range(epoch):
            population = []
            for _ in range(self.population_size):
                x = []
                for w in self.weights:
                    x.append(np.random.randn(*w.shape))
                population.append(x)
            futures = [reward_function.remote(self._get_weight_from_population(self.weights,population[k])) for k in range(self.population_size)]

            rewards = ray.get(futures)
            
            # if std is 0 which can happen when model is starting out, add a small number
            rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 0.0001)
            #print("rewards after normalization: ",rewards)
            
            for index, w in enumerate(self.weights):
                A = np.array([p[index] for p in population])
                self.weights[index] = (
                    w + self.learning_rate / (self.population_size * self.sigma) * np.dot(A.T, rewards).T
                )
                #print("weights:",self.weights)
            
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
        #print("state:",state)
        decision, buy = self.model.predict(state.T)
        #print('decision: ', decision)
        #print('buy: ', buy)

        return [np.argmax(decision[0]), min(self.max_shares_to_trade_at_once,max(int(buy[0]),0))]
    
    def fit(self, iterations, checkpoint):
        self.des.train(iterations, print_every = checkpoint)

class Model:
    def __init__(self, input_size, layer_size, output_size):
        self.weights = []
        self.layers = CONFIG["layers"]
        for i in range(self.layers):
            if i == 0:
                self.weights.append(np.random.rand(input_size, layer_size))
                self.weights.append(np.random.rand(1,layer_size))
            else:
                self.weights.append(np.random.rand(layer_size, layer_size))
                self.weights.append(np.random.rand(1,layer_size))
        self.weights.append(np.random.randn(layer_size, output_size))
        self.weights.append(np.random.randn(layer_size, 1))
        self.weights.append(np.random.randn(1, layer_size))

    def predict(self, inputs):
        for i in range(0,self.layers,2):
            #print("i:",i)
            #feed = np.dot(inputs, self.weights[0]) + self.weights[-1]
            feed = np.dot(inputs, self.weights[i]) + self.weights[i+1]
            inputs = feed
        decision = np.dot(feed,self.weights[self.layers*2])
        buy = np.dot(feed,self.weights[self.layers*2+1])
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
    model = Model(time_frame * state_size, CONFIG['layer_size'], 3)

    dirname = os.path.dirname(__file__)
    weights_file = os.path.join(dirname,'deep_evo_weights.npy')

    if os.path.exists(weights_file):
        print('loading weights')
        weights = np.load(weights_file,allow_pickle=True)
        model.set_weights(weights)

    # np.save(weights_file,model.get_weights())
    agent = Agent(model,state_size, time_frame)
    if CONFIG['train']:
        agent.fit(iterations=CONFIG['iterations'], checkpoint=1)
        agent.model.set_weights(agent.des.get_weights())
        np.save(weights_file, agent.des.get_weights())
    
    if CONFIG['eval']:
        closes, states_buy, states_sell, result = run_agent(agent,eval=True)
        print('result: {}'.format(str(result)))
        plt.figure(figsize = (20, 10))
        plt.plot(closes, label = 'true close', c = 'g')
        plt.plot(closes, 'X', label = 'predict buy', markevery = states_buy, c = 'b')
        plt.plot(closes, 'o', label = 'predict sell', markevery = states_sell, c = 'r')
        plt.legend()
        #plt.show()
        plt.savefig('deep_evolution_results.jpg')