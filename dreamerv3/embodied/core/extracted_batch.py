import numpy as np

from . import base

class ExtractedBatchEnv(base.Env):

  def __init__(self, env):
    self._env = env
    self._keys = list(self.obs_space.keys())

  @property
  def obs_space(self):
    return self._env.obs_space

  @property
  def act_space(self):
    return self._env.act_space

  def __len__(self):
    return self._env.n_envs

  def step(self, action):
    obs = self._env.step(action)
    return obs

  def reset(self):
    return self._env.reset()

  def render(self):
    return self._env.render()

  def close(self):
    try:
      self._env.close()
    except Exception:
      pass
