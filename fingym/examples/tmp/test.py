from fingym import fingym

env = fingym.make('SPY-Daily-v0')
env.reset()

for _ in range(1):
    observation, reward, done, info = env.step(env.action_space.sample())

env.close()

print("observation:",observation)
print("reward:",reward)
print("done:",done)
print("info",info)