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

from collections import deque

import argparse
import os
import os.path

from fingym import fingym

import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Embedding, Reshape
from tensorflow.keras.optimizers import Adam

import random
import numpy as np

class DQNAgent():

    def __init__(self, env_state_dim, time_frame, epsilon = 1, learning_rate=0.01, train = False):
        dirname = os.path.dirname(__file__)
        self.model_filepath = os.path.join(dirname,'weights.h5')
        self._state_size = env_state_dim
        self.trainMode = train

        # 0 - do nothing
        # 1 - buy w/ multiplier .33
        # 2 - buy w/ multiplier .5
        # 3 - buy w/ multiplier .66
        # 4 - sell w/ multiplier .33
        # 5 - sell w/ multiplier .5
        # 6 - sell w/ multiplier .66
        self._action_size = 7

        self.experience_replay = deque(maxlen=2000)

        self.gamma = 0.98
        if not self.trainMode:
            self.epsilon = 0
        else:
            self.epsilon = epsilon
        self.eps_decay = 0.995
        self.eps_min = 0.01

        self.max_shares_to_trade_at_once = 100

        # holds our last time_frame sequential state frames for prediction.
        self._time_frame = time_frame
        self.state_fifo = deque(maxlen=self._time_frame)

        # networks
        self.q_network = self._build_compile_model(learning_rate)
        self.target_network = self._build_compile_model(learning_rate)
        self._load_model_weights(self.q_network, self.model_filepath)
        self.align_target_model()

    def align_target_model(self):
        # reduce exploration rate as more
        # training happens
        if self.epsilon > self.eps_min:
            self.epsilon *= self.eps_decay
    
        print('epsilon: ', self.epsilon)

        self.target_network.set_weights(self.q_network.get_weights())
        self._save_model_weights(self.q_network, self.model_filepath)

    def _build_compile_model(self, learning_rate):
        '''
        Model taken from https://arxiv.org/pdf/1802.09477.pdf
        '''
        model = Sequential()

        # use a dense nn with inputs
        input_size = self._state_size * self._time_frame
        model.add(Dense(400, input_shape=(input_size,), activation='relu'))
        model.add(Dense(300, activation='relu'))
        #model.add(Dense(self._action_size, activation='tanh'))
        model.add(Dense(self._action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
        print(model.summary())
        return model
    
    def store(self, state, action, reward, next_state, terminated):
        state = np.reshape(state, (self._state_size,self._time_frame))
        next_state = np.reshape(next_state, (self._state_size, self._time_frame))
        self.experience_replay.append((state, action, reward, next_state, terminated))
    
    def act(self, state):
        self.state_fifo.append(state)

        # do nothing for the first time frames until we can start the prediction
        if len(self.state_fifo) < self._time_frame:
            # Our environment takes a tuple for action https://entrpn.github.io/fingym/#spaces
            return np.zeros(2)
        
        # epsilon decays over time
        if np.random.rand() <= self.epsilon:
            return self._random_action()

        state = np.array(list(self.state_fifo))
        state = np.reshape(state,(self._state_size*self._time_frame,1))

        q_values = self.q_network.predict_on_batch(state.T)
        env_action = self._nn_action_to_env_action(np.argmax(q_values[0]))
        return env_action
    
    def retrain(self, batch_size):
        if not self.trainMode:
            return
        
        minibatch = random.sample(self.experience_replay, batch_size)

        for state, action, reward, next_state, terminated in minibatch:
            state = np.reshape(state,(self._state_size * self._time_frame,1))
            next_state = np.reshape(next_state, (self._state_size * self._time_frame, 1))
            target = np.array(self.q_network.predict_on_batch(state.T))

            if terminated[-1]:
                target[0][np.argmax(self._nn_action_to_env_action(action))] = reward[-1]
            else:
                t = np.array(self.target_network.predict_on_batch(next_state.T))
                target[0][np.argmax(self._env_action_to_nn_action(action))] = reward[-1] + self.gamma * np.amax(t)
            
            self.q_network.fit(state.T, target, epochs=1, verbose = 0)

    def _random_action(self):
        return self._nn_action_to_env_action(np.random.choice(self._action_size))
    
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

    def _env_action_to_nn_action(self, action):
        nn_action = np.zeros(self._action_size)

        if action[0] == 0:
            nn_action[0] = 1

        if action[0] == 1:
            multiplier = action[1] / self.max_shares_to_trade_at_once
            if multiplier == 0.33:
                nn_action[1] = 1
            if multiplier == 0.5:
                nn_action[2] = 1
            if multiplier == 0.66:
                nn_action[3] = 1

        if action[0] == 2:
            multiplier = action[1] / self.max_shares_to_trade_at_once
            if multiplier == 0.33:
                nn_action[4] = 1
            if multiplier == 0.5:
                nn_action[5] = 1
            if multiplier == 0.66:
                nn_action[6] = 1

        return nn_action

    def _save_model_weights(self, model, filepath):
        model.save_weights(filepath)
    
    def _load_model_weights(self, model, filepath):
        if os.path.exists(filepath):
            print('loading existing weights')
            model.load_weights(filepath)

def reset_timeframe(time_frame, state_dim):
    s = np.zeros((time_frame,state_dim))
    r = np.zeros((time_frame))
    d = np.zeros((time_frame))
    ns = np.zeros((time_frame, state_dim))
    return s, r, ns, d 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='SPY-Daily-v0', help='Select the environment to run')
    args = parser.parse_args()

    train = True
    if train:
        rang = 100
    else:
        rang = 1

    # collect the last 10 time frames (10 days for daily env) and use that to make a prediction for current action
    time_frame = 10
    time_frame_counter = 0

    # train on this batch size
    batch_size = 32

    env = fingym.make(args.env_id)
    # removing time element from state_dim since I'm creating a sequence via time_frame
    state_size = env.state_dim - 1
    print('state_size: ', state_size)
    agent = DQNAgent(state_size, time_frame, train = train)

    for i in range(rang):

        # init our env
        state = env.reset()
        # remove time element
        state = np.delete(state, 2)
        done = False

        # init our timeframe
        s_timeframe, r_timeframe, ns_timeframe, d_timeframe = reset_timeframe(time_frame, state_size)

        # alighn every training iterations
        align_every_itt = 15
        align_counter = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            print('action: ', action)
            print('reward: ', reward)

            # remove time element
            if len(state) > state_size:
                state = np.delete(state, 2)
            next_state = np.delete(next_state, 2)

            if time_frame_counter >= time_frame:
                agent.store(s_timeframe, action, r_timeframe, ns_timeframe, d_timeframe)
                s_timeframe[:-1] = s_timeframe[1:]
                r_timeframe[:-1] = r_timeframe[1:]
                ns_timeframe[:-1] = ns_timeframe[1:]
                d_timeframe[:-1] = d_timeframe[1:]
                time_frame_counter-=1
            
            s_timeframe[time_frame_counter] = state
            r_timeframe[time_frame_counter] = reward
            ns_timeframe[time_frame_counter] = next_state
            d_timeframe[time_frame_counter] = done
            time_frame_counter+=1

            if len(agent.experience_replay) > batch_size:
                print('retrain')
                agent.retrain(batch_size)

            if align_counter >= align_every_itt:
                print('align target model')
                agent.align_target_model()
                align_counter = 0
                print(info)
            
            state = next_state
            align_counter+=1
