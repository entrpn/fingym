from fingym.examples.agents.multi_env_evo_agent import build_compile_model, EvoAgent

from fingym import fingym

import os

import numpy as np

import matplotlib.pyplot as plt

def test_agent(env, agent, model):

    state = env.reset()
    # Removed time element from state
    state = np.delete(state, 2)
    state_as_percentages = state
    done = False
    states_buy = []
    states_sell = []
    closes = []

    i = 0
    while not done:
        closes.append(state[5])
        action = agent.act(state_as_percentages, model)
        print(action)
        next_state, reward, done, info = env.step(action)
        if len(next_state) > agent.state_size:
            next_state = np.delete(next_state, 2)
        if action[0] == 1 and state[1] > next_state[2]:
            states_buy.append(i)
        if action[0] == 2 and state[0] > 0:
            states_sell.append(i)
        opn = (next_state[2] - state[2]) / next_state[2]
        high = (next_state[3] - state[3]) / next_state[3]
        low = (next_state[4] - state[4]) / next_state[4]
        close = (next_state[5] - state[5]) / next_state[5]
        volume = (next_state[6] - state[6]) / next_state[6]
        state_as_percentages = [next_state[0], next_state[1], opn, high, low, close, volume]
        state = next_state
        i+=1
    return closes, states_buy, states_sell, info['cur_val']

env = fingym.make('SPY-Daily-v0')
# removing time frame
state_size = env.state_dim - 1
time_frame = 30

dirname = os.path.dirname(__file__)
weights_file = os.path.join(dirname,'evo_weights_5.h5')
model = build_compile_model(state_size*time_frame,7)
model.load_weights(weights_file)

agent = EvoAgent(state_size, time_frame, model.get_weights())

closes, states_buy, states_sell, result = test_agent(env,agent,model)
print('result: {}'.format(str(result)))
plt.figure(figsize = (20, 10))
plt.plot(closes, label = 'true close', c = 'g')
plt.plot(closes, 'X', label = 'predict buy', markevery = states_buy, c = 'b')
plt.plot(closes, 'o', label = 'predict sell', markevery = states_sell, c = 'r')
plt.legend()
plt.show()