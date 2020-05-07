import numpy as np
import matplotlib.pyplot as plt

from fingym import make
from datetime import datetime, timedelta

from iexfinance.stocks import get_historical_data

if __name__ == '__main__':
  start = datetime(2019, 3, 1)
  end = datetime(2020, 9, 4)
  
  env = make('IEXCloud-Daily-v0',stock_symbol='SPY', iex_token = '', iex_start=start, iex_end=end)

  real_close = np.zeros(env.n_step)
  obs = env.reset()
  real_close[0] = obs[3]

  while True:
    obs, reward, done, info = env.step([1,1000])
    real_close[env.cur_step] = obs[4]
    if done:
      print('total reward: {}'.format(info['cur_val']))
      break
  
  time = np.linspace(1, len(real_close), len(real_close))
  plt.plot(time, real_close, label = 'Actual',linewidth=1.0)
  plt.title('IEX - SPY')
  plt.legend(loc = 'upper left')
  plt.show()

