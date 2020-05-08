import numpy as np
import matplotlib.pyplot as plt

from fingym import make

if __name__ == '__main__':
  env = make('Alphavantage-Daily-v0',stock_symbol='SPY', alphavantage_key = '')
  
  close = np.zeros(env.n_step)
  obs = env.reset()
  close[0] = obs[3]

  while True:
    obs, reward, done, info = env.step([1,1000])
    close[env.cur_step] = obs[4]
    if done:
      print('total reward: {}'.format(info['cur_val']))
      break
  
  time = np.linspace(1, len(close), len(close))
  plt.plot(time, close, label = 'SPY',linewidth=1.0)
  plt.title('Alphavantage')
  plt.legend(loc = 'upper left')
  plt.show()