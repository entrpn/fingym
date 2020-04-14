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
# https://gist.github.com/karpathy/77fbb6a8dac5395f1b73e7a89300318d


from collections import deque

import uuid
import ray

from fingym import fingym

import argparse
import numpy as np

import copy
import os

ray.init(num_cpus=12)

NOISE = np.random.randn(250000000).astype(np.float32)

envs = []
spyEnv = fingym.make('SPY-Daily-v0')
envs.append(spyEnv)
# tslaEnv = fingym.make('TSLA-Daily-v0')
# envs.append(tslaEnv)
# googlEnv = fingym.make('GOOGL-Daily-v0')
# envs.append(googlEnv)
# cgcEnv = fingym.make('CGC-Daily-v0')
# envs.append(cgcEnv)
# cronEnv = fingym.make('CRON-Daily-v0')
# envs.append(cronEnv)
# baEnv = fingym.make('BA-Daily-v0')
# envs.append(baEnv)
# amznEnv = fingym.make('AMZN-Daily-v0')
# envs.append(amznEnv)
# amdEnv = fingym.make('AMD-Daily-v0')
# envs.append(amdEnv)
# abbvEnv = fingym.make('ABBV-Daily-v0')
# envs.append(abbvEnv)
# aaplEnv = fingym.make('AAPL-Daily-v0')
#envs.append(aaplEnv)

def build_compile_model(input_size, output_size):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import Model, Sequential
    from tensorflow.keras.layers import Dense, Embedding, Reshape
    from tensorflow.keras.optimizers import Adam

    model = Sequential()

    model.add(Dense(400, input_shape=(input_size,), activation='relu',kernel_initializer='he_uniform', use_bias=True, bias_initializer=keras.initializers.Constant(0.1)))
    model.add(Dense(300, activation='relu', kernel_initializer='he_uniform', use_bias=True, bias_initializer=keras.initializers.Constant(0.1)))
    model.add(Dense(output_size, activation='softmax',use_bias=True, bias_initializer=keras.initializers.Constant(0.1)))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.01))

    return model

def save_model_weights(model, filepath):
    model.save_weights(filepath)

class EvoAgent():
    def __init__(self, env_state_dim, time_frame, weights, id = None):
        if id is None:
            self.id = uuid.uuid4()
        else:
            self.id = id
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

        self.weights = weights
    
    def deep_copy(self, id=None):
        weights = np.copy(self.weights)
        new_agent = EvoAgent(self.state_size, self.time_frame, weights, id)
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
        model = build_compile_model(state_size * time_frame, 7)
        agent = EvoAgent(state_size, time_frame, model.get_weights())
        agents.append(agent)

    return agents

def run_agent(env, agent, model):

    state = env.reset()
    # Removed time element from state
    state = np.delete(state, 2)
    state_as_percentages = state
    done = False
    while not done:
        action = agent.act(state_as_percentages, model)
        next_state, reward, done, info = env.step(action)
        if len(next_state) > agent.state_size:
            next_state = np.delete(next_state, 2)
        opn = (next_state[2] - state[2]) / next_state[2]
        high = (next_state[3] - state[3]) / next_state[3]
        low = (next_state[4] - state[4]) / next_state[4]
        close = (next_state[5] - state[5]) / next_state[5]
        volume = (next_state[6] - state[6]) / next_state[6]
        state_as_percentages = [next_state[0], next_state[1], opn, high, low, close, volume]
        state = next_state
    return info['cur_val']

@ray.remote
def return_average_score(envs, agent, runs):
    model = build_compile_model(agent.state_size * agent.time_frame, agent.action_size)
    model.set_weights(agent.weights)
    score = 0
    for env in envs:
        envScore = 0
        for i in range(runs):
            envScore += run_agent(env, agent, model)
        envScore = envScore/runs
        print('AGENT: {}, env: {}, score: {}'.format(agent.id, type(env).__name__,envScore))
        score += envScore
    score = score/len(envs)
    print('AGENT: {}, total score: {}'.format(agent.id,score))
    return agent, score

def run_agents_n_times(envs, agents, runs):
    avg_score_map = {}
    avg_score = []
    futures = [return_average_score.remote(envs,agent,runs) for agent in agents]
    results = ray.get(futures)
    for result in results:
        avg_score_map[result[0].id] = result[1]
    for agent in agents:
        avg_score.append(avg_score_map[agent.id])
    return avg_score

def mutate(agent):
    print('mutate')
    obj_id = ray.put(agent.weights)
    child_agent = EvoAgent(agent.state_size, agent.time_frame, np.copy(ray.get(obj_id)))#ray.get(obj_id).deep_copy()
    
    #child_agent = agent.deep_copy()

    mutation_power = 0.02

    weights = child_agent.weights
    # array copied to object store (objects are immutable), not heap, so a copy must be made
    # https://github.com/ray-project/ray/issues/369
    #weights = np.copy(c_weights)
    #weights.flags.writeable = True
    #print('flags: ', weights.flags)

    for weight in weights:
        
        if len(weight.shape) == 2:
            for i0 in range(weight.shape[0]):
                for i1 in range(weight.shape[1]):
                    #print('flags: ', weight.flags)
                    weight.flags.writeable = True
                    weight[i0,i1]+= mutation_power*NOISE[np.random.randint(0,len(NOISE)-1)]

        if len(weight.shape) == 1:
            for i0 in range(weight.shape[0]):
                weight.flags.writeable = True
                weight[i0]+= mutation_power*NOISE[np.random.randint(0,len(NOISE)-1)]
    
    child_agent.weights = weights
    return child_agent

def add_elite(envs, agents, sorted_parent_indexes, elite_index = None, only_consider_top_n=10):
    candidate_elite_index = sorted_parent_indexes[:only_consider_top_n]

    if(elite_index is not None):
        candidate_elite_index = np.append(candidate_elite_index,[elite_index])

    top_score = None
    top_elite_index = None
    elite_agent = None

    futures = [return_average_score.remote(envs, agents[i],runs=5) for i in candidate_elite_index]
    results = ray.get(futures)
    
    dirname = os.path.dirname(__file__)

    i = 0
    
    for result in results:
        agent = result[0]
        score = result[1]

        print("Score for elite agent {} is {} ".format(agent.id,score))
        model = build_compile_model(agent.state_size * agent.time_frame, agent.action_size)
        model.set_weights(agent.weights)
        save_model_weights(model,os.path.join(dirname,'evo_weights_' + str(i) + '.h5'))

        if(top_score is None):
            top_score = score
            elite_agent = agent
        elif(score > top_score):
            top_score = score
            elite_agent = agent
        
        i+=1
            
    print("Elite selected has id ",elite_agent.id, " and score", top_score)
    
    child_agent = elite_agent.deep_copy(elite_agent.id)
    return child_agent

def return_children(envs, agents, sorted_parent_indexes, elite_index):
    children_agents = []

    for i in range(len(agents) -1):
        selected_agent_index = sorted_parent_indexes[np.random.randint(len(sorted_parent_indexes))]
        children_agents.append(mutate(agents[selected_agent_index]))
        
    
    # now add one elite
    elite_child = add_elite(envs, agents, sorted_parent_indexes, elite_index)
    children_agents.append(elite_child)
    elite_index=len(children_agents)-1

    return children_agents, elite_index

if __name__ == '__main__':

    # removing time element from state_dim
    state_size = envs[0].state_dim - 1
    print('state_size: ', state_size)

    time_frame = 30
    num_agents = 4

    agents = create_random_agents(num_agents, state_size, time_frame)

    # first agent gets saved weights
    dirname = os.path.dirname(__file__)

    # how many top agents to consider as parents
    top_limit = 2

    for i in range (top_limit):
        weights_file = os.path.join(dirname,'evo_weights_' + str(i) + '.h5')
        if os.path.exists(weights_file):
            print('loading existing weights')
            model = build_compile_model(agents[0].state_size * agents[0].time_frame, agents[0].action_size)
            model.load_weights(weights_file)
            agents[i].weights = model.get_weights()

    # run evolution until x generations
    generations = 1000

    elite_index = None

    for generation in range(generations):
        rewards = run_agents_n_times(envs,agents,5) # average of x times

        # sort by rewards
        sorted_parent_indexes = np.argsort(rewards)[::-1][:top_limit]

        top_rewards = []
        for best_parent in sorted_parent_indexes:
            top_rewards.append(rewards[best_parent])
        
        print("Generation ", generation, " | Mean rewards: ", np.mean(rewards), " | Mean of top 5: ",np.mean(top_rewards[:5]))
        print("Top ",top_limit," scores", sorted_parent_indexes)
        print("Rewards for top: ",top_rewards)

        children_agents, elite_index = return_children(envs, agents, sorted_parent_indexes, elite_index)

        agents = children_agents