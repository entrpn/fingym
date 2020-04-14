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

# https://becominghuman.ai/genetic-algorithm-for-reinforcement-learning-a38a5612c4dc


from collections import deque

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import model_from_json, clone_model
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Embedding, Reshape
from tensorflow.keras.optimizers import Adam

from fingym import fingym

import argparse
import numpy as np

import copy
import os

class EvoAgent():
    def __init__(self, env_state_dim, time_frame):
        self.state_size = env_state_dim
        self.time_frame = time_frame
        # when this is full, start making predictions
        self.state_fifo = deque(maxlen=self.time_frame)

        # 0 - do nothing
        # 1 - buy w/ multiplier .33
        # 2 - buy w/ multiplier .5
        # 3 - buy w/ multiplier .66
        # 4 - sell w/ multiplier .33
        # 5 - sell w/ multiplier .5
        # 6 - sell w/ multiplier .66
        self.action_size = 7

        self.max_shares_to_trade_at_once = 100

        self.model = self._build_compile_model()
    
    def deep_copy(self):
        new_agent = EvoAgent(self.state_size, self.time_frame)
        new_agent.model.set_weights(self.model.get_weights())

        return new_agent
    
    def act(self, state, model):
        self.state_fifo.append(state)

        # do nothing for the first time frames until we can start the prediction
        if len(self.state_fifo) < self.time_frame:
            return np.zeros(2)
        
        state = np.array(list(self.state_fifo))
        state = np.reshape(state,(self.state_size*self.time_frame,1))

        output_probabilities = model.predict_on_batch(state.T)[0]
        output_probabilities = np.array(output_probabilities)
        output_probabilities /= output_probabilities.sum()
        try:
            action = np.random.choice(range(self.action_size),1,p=output_probabilities).item()
        except:
            print('output probabilities: ', output_probabilities)
            action = np.zeros(2)
        env_action = self._nn_action_to_env_action(action)
        return env_action
    
    def save_model_weights(self, filepath):
        self.model.save_weights(filepath)
    
    def _build_compile_model(self):
        model = Sequential()

        input_size = self.state_size * self.time_frame
        model.add(Dense(400, input_shape=(input_size,), activation='relu',kernel_initializer='he_uniform', use_bias=True, bias_initializer=keras.initializers.Constant(0.1)))
        model.add(Dense(300, activation='relu', kernel_initializer='he_uniform', use_bias=True, bias_initializer=keras.initializers.Constant(0.1)))
        #model.add(Dense(1, activation='softmax',use_bias=True))
        model.add(Dense(self.action_size, activation='softmax',use_bias=True, bias_initializer=keras.initializers.Constant(0.1)))
        # we won't really use loss or optimizer for evolutionary agents
        #model.compile(loss='mse', optimizer=Adam(learning_rate=0.01))
        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.01))
        #print(model.summary())
        return model
    
    def _nn_action_to_env_action(self,nn_action):
        env_action = [0,0]
        
        if nn_action == 0:
            env_action = [0,0]
        if nn_action == 1:
            env_action = [1, 0.33 * self.max_shares_to_trade_at_once]
        if nn_action == 2:
            env_action = [1, 0.5 * self.max_shares_to_trade_at_once]
        if nn_action == 3:
            env_action = [1, 0.66 * self.max_shares_to_trade_at_once]
        if nn_action == 4:
            env_action = [2, 0.33 * self.max_shares_to_trade_at_once]
        if nn_action == 5:
            env_action = [2, 0.5 * self.max_shares_to_trade_at_once]
        if nn_action == 6:
            env_action = [2, 0.66 * self.max_shares_to_trade_at_once]

        return env_action

def create_random_agents(num_agents, state_size, time_frame):
    agents = []
    for _ in range(num_agents):
        agent = EvoAgent(state_size, time_frame)
        agents.append(agent)

    return agents

def run_agent(env, agent):
    state = env.reset()
    # Removed time element from state
    state = np.delete(state, 2)
    done = False
    while not done:
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        state = next_state
        if len(state) > agent.state_size:
            state = np.delete(state, 2)
    return info['cur_val']

def return_average_score(env, agent, runs):
    score = 0
    print('***** agent score *****')
    for i in range(runs):
        score += run_agent(env, agent)
        print('score: ', score)
    return score/runs

def run_agents_n_times(env, agents, runs):
    avg_score = []
    for agent in agents:
        avg_score.append(return_average_score(env, agent, runs))
    return avg_score

def uniform_crossover(parentA, parentB):
    print('crossover')
    child_agent = parentA.deep_copy()

    parentB_weights = parentB.model.get_weights()

    weights = child_agent.model.get_weights()

    for idx, weight in enumerate(weights):
        
        if len(weight.shape) == 2:
            for i0 in range(weight.shape[0]):
                for i1 in range(weight.shape[1]):
                    if np.random.uniform() > 0.5:
                        weight[i0,i1] = parentB_weights[idx][i0,i1]

        if len(weight.shape) == 1:
            for i0 in range(weight.shape[0]):
                if np.random.uniform() > 0.5:
                    weight[i0] = parentB_weights[idx][i0]

    child_agent.model.set_weights(weights)
    
    return child_agent


def mutate(agent):
    print('mutate')
    child_agent = agent.deep_copy()

    mutation_power = 0.02

    weights = child_agent.model.get_weights()

    for weight in weights:
        #print('weight len: ', len(weight.shape))
        
        if len(weight.shape) == 2:
            for i0 in range(weight.shape[0]):
                for i1 in range(weight.shape[1]):
                    weight[i0,i1]+= mutation_power*np.random.randn()

        if len(weight.shape) == 1:
            for i0 in range(weight.shape[0]):
                weight[i0]+= mutation_power*np.random.randn()
    
    child_agent.model.set_weights(weights)
    #print('parent_weights: ', agent.model.get_weights())
    #print('child_weights: ', child_agent.model.get_weights())
    return child_agent

def add_elite(env, agents, sorted_parent_indexes, elite_index = None, only_consider_top_n=10):
    candidate_elite_index = sorted_parent_indexes[:only_consider_top_n]

    if(elite_index is not None):
        candidate_elite_index = np.append(candidate_elite_index,[elite_index])

    top_score = None
    top_elite_index = None
    
    for i in candidate_elite_index:
        score = return_average_score(env, agents[i],runs=3)
        print("Score for elite i ", i, " is ", score)
        
        if(top_score is None):
            top_score = score
            top_elite_index = i
        elif(score > top_score):
            top_score = score
            top_elite_index = i
            
    print("Elite selected with index ",top_elite_index, " and score", top_score)
    dirname = os.path.dirname(__file__)
    agents[top_elite_index].save_model_weights(os.path.join(dirname,'evo_weights.h5'))
    
    child_agent = agents[top_elite_index].deep_copy()
    return child_agent

def return_children(env, agents, sorted_parent_indexes, elite_index):
    children_agents = []

    for i in range(len(agents) -1):
        parentA = sorted_parent_indexes[np.random.randint(len(sorted_parent_indexes))]
        parentB = sorted_parent_indexes[np.random.randint(len(sorted_parent_indexes))]
        children_agents.append(mutate(uniform_crossover(agents[parentA], agents[parentB])))
        
    
    # now add one elite
    elite_child = add_elite(env, agents, sorted_parent_indexes, elite_index)
    children_agents.append(elite_child)
    elite_index=len(children_agents)-1

    return children_agents, elite_index

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='SPY-Daily-v0', help='Select the environment to run')
    args = parser.parse_args()

    env = fingym.make(args.env_id)

    # removing time element from state_dim
    state_size = env.state_dim - 1
    print('state_size: ', state_size)

    time_frame = 30
    num_agents = 400

    agents = create_random_agents(num_agents, state_size, time_frame)

    # first agent gets saved weights
    dirname = os.path.dirname(__file__)
    os.path.join(dirname,'evo_weights.h5')
    weights_file=os.path.join(dirname,'evo_weights.h5')
    if os.path.exists(weights_file):
        print('loading existing weights')
        agents[0].model.load_weights(weights_file)

    # how many top agents to consider as parents
    top_limit = 20

    # run evolution until x generations
    generations = 1000

    elite_index = None

    for generation in range(generations):
        rewards = run_agents_n_times(env,agents,3) # average of x times

        # sort by rewards
        sorted_parent_indexes = np.argsort(rewards)[::-1][:top_limit]

        top_rewards = []
        for best_parent in sorted_parent_indexes:
            top_rewards.append(rewards[best_parent])
        
        print("Generation ", generation, " | Mean rewards: ", np.mean(rewards), " | Mean of top 5: ",np.mean(top_rewards[:5]))
        print("Top ",top_limit," scores", sorted_parent_indexes)
        print("Rewards for top: ",top_rewards)

        children_agents, elite_index = return_children(env, agents, sorted_parent_indexes, elite_index)

        agents = children_agents