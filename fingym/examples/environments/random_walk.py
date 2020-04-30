from fingym import fingym
import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(20,10))

if __name__ == '__main__':
  env = fingym.make('SPY-Daily-Random-Walk')

  random_walks = []

  for _ in range(100):
    real_close = np.zeros(env.n_step)
    random_walk = np.zeros(env.n_step)

    obs = env.reset()
    real_close[0] = obs[3]
    random_walk[0] = obs[3]

    while True:
      obs, reward, done, info = env.step([0,0])
      real_close[env.cur_step] = info['original_close']
      random_walk[env.cur_step] = obs[3]

      if done:
        break

    random_walks.append(random_walk)


  time = np.linspace(1, len(random_walk), len(random_walk))

  for random_walk in random_walks:
    plt.plot(time, random_walk, ls = '--')

  plt.plot(time, real_close, label = 'Actual',linewidth=4.0)
  plt.title('Geometric Brownian Motion - SPY')
  plt.legend(loc = 'upper left')
  plt.show()
