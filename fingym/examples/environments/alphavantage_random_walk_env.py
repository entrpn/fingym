from fingym import make
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
  env = make('Alphavantage-Daily-Random-Walk',stock_symbol='SPY', alphavantage_key = '', no_days_to_random_walk=1000)

  random_walks = []

  for _ in range(5):
    real_close = np.zeros(env.n_step)
    random_walk = np.zeros(env.n_step)

    obs = env.reset()
    real_close[0] = obs[3]
    random_walk[0] = obs[3]

    while True:
      obs, reward, done, info = env.step([1,1000])
      real_close[env.cur_step] = info['original_close']
      random_walk[env.cur_step] = obs[3]

      if done:
        print('total reward: {}'.format(info['cur_val']))
        break

    random_walks.append(random_walk)


  time = np.linspace(1, len(random_walk), len(random_walk))

  for random_walk in random_walks:
    plt.plot(time, random_walk, ls = '--')

  plt.plot(time, real_close, label = 'Actual',linewidth=2.0)
  plt.title('Geometric Brownian Motion - SPY')
  plt.legend(loc = 'upper left')
  plt.show()
