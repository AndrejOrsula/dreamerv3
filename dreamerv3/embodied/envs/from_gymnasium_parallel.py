import functools
from typing import Generic, TypeVar, Union, cast

import embodied
import gymnasium
import numpy as np

U = TypeVar("U")
V = TypeVar("V")


class FromGymnasiumParallel(embodied.Env, Generic[U, V]):
    def __init__(
        self,
        env: Union[str, gymnasium.Env[U, V]],
        obs_key="vector",
        act_key="action",
        **kwargs
    ):
        if isinstance(env, str):
            self._env: gymnasium.Env[U, V] = gymnasium.make(
                env, render_mode="rgb_array", **kwargs
            )
        else:
            self._env = env
        self._obs_dict = hasattr(self._env.observation_space, "spaces")
        self._act_dict = hasattr(self._env.action_space, "spaces")
        self._obs_key = obs_key
        self._act_key = act_key
        self._done = np.array([True for _ in range(self.n_envs)])
        # self._info = None

    # @property
    # def info(self):
        # return self._info
    
    @property
    def n_envs(self):
        return self._env.n_envs

    @functools.cached_property  # type: ignore
    def obs_space(self):
        if self._obs_dict:
            # cast is here to stop type checkers from complaining (we already check
            # that .spaces attr exists in __init__ as a proxy for the type check)
            obs_space = cast(gymnasium.spaces.Dict, self._env.observation_space)
            spaces = obs_space.spaces
        else:
            spaces = {self._obs_key: self._env.observation_space}
        spaces = {k: self._convert(v) for k, v in spaces.items()}
        return {
            **spaces,
            "reward": embodied.Space(np.float32),
            "is_first": embodied.Space(bool),
            "is_last": embodied.Space(bool),
            "is_terminal": embodied.Space(bool),
        }

    @functools.cached_property  # type: ignore
    def act_space(self):
        if self._act_dict:
            act_space = cast(gymnasium.spaces.Dict, self._env.action_space)
            spaces = act_space.spaces
        else:
            spaces = {self._act_key: self._env.action_space}
        spaces = {k: self._convert(v) for k, v in spaces.items()}
        spaces["reset"] = embodied.Space(bool)
        return spaces

    def step(self, action):
        if self._act_dict:
            gymnasium_action = cast(V, self._unflatten(action))
        elif isinstance(action, dict):
            gymnasium_action = cast(V, action[self._act_key])
        else:
            gymnasium_action = action

        envs_to_reset = np.where(self._done == True)[0]
        envs_to_step = np.where(self._done == False)[0]

        if len(envs_to_reset) > 0:
            reset_ret = self._env.reset_envs(envs_to_reset)    
            
            reset_observations = np.stack([obs for obs, _ in reset_ret])

            self._done[envs_to_reset] = False
        
        if len(envs_to_step) > 0:
            step_ret = self._env.step_envs(gymnasium_action, envs_to_step)

            step_observations = np.stack([obs for obs, _, _, _, _ in step_ret])
            step_rewards = np.array([reward for _, reward, _, _, _ in step_ret])
            step_terminated = np.array([terminated for _, _, terminated, _, _ in step_ret])
            step_truncated = np.array([truncated for _, _, _, truncated, _ in step_ret])
            # self._info = [info for _, _, _, _, info in step_ret]

            self._done[envs_to_step] = np.logical_or(step_terminated, step_truncated)


        obs = []
        reward = []
        is_first = []
        is_last = []
        is_terminal = []
        index_reset = 0
        index_step = 0
        for i in range(self.n_envs):
            if i in envs_to_reset:
                obs.append(reset_observations[index_reset])
                reward.append(0.0)
                is_first.append(True)
                is_last.append(False)
                is_terminal.append(False)
                index_reset += 1
            else:
                obs.append(step_observations[index_step])
                reward.append(step_rewards[index_step])
                is_first.append(False)
                is_last.append(self._done[i])
                is_terminal.append(step_terminated[index_step])
                index_step += 1

        return {
            self._obs_key: np.stack(obs),
            "reward": np.array(reward),
            "is_first": np.array(is_first),
            "is_last": np.array(is_last),
            "is_terminal": np.array(is_terminal),
        }


    def reset(self):
        return self._env.reset()

    def _obs(self, obs, reward, is_first=None, is_last=None, is_terminal=None):
        false_array = np.array([False for _ in range(len(obs))])
        if is_first is None:
            is_first = false_array.copy()
        if is_last is None:
            is_last = false_array.copy()
        if is_terminal is None:
            is_terminal = false_array.copy()

        if not self._obs_dict:
            obs = {self._obs_key: obs}
        obs.update(
            reward=reward,
            is_first=is_first,
            is_last=is_last,
            is_terminal=is_terminal,
        )
        return obs

    def render(self):
        image = self._env.render()
        assert image is not None
        return image

    def close(self):
        try:
            self._env.close()
        except Exception:
            pass

    def _flatten(self, nest, prefix=None):
        result = {}
        for key, value in nest.items():
            key = prefix + "/" + key if prefix else key
            if isinstance(value, gymnasium.spaces.Dict):
                value = value.spaces
            if isinstance(value, dict):
                result.update(self._flatten(value, key))
            else:
                result[key] = value
        return result

    def _unflatten(self, flat):
        result = {}
        for key, value in flat.items():
            parts = key.split("/")
            node = result
            for part in parts[:-1]:
                if part not in node:
                    node[part] = {}
                node = node[part]
            node[parts[-1]] = value
        return result

    def _convert(self, space):
        if hasattr(space, "n"):
            return embodied.Space(np.int32, (), 0, space.n)
        return embodied.Space(space.dtype, space.shape, space.low, space.high)
