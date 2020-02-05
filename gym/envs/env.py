class Env(object):
    def step(self, action):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def close(self):
        pass

    def seed(self, seed):
        return