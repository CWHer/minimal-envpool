# Copyright 2021 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Test for envpool.make."""

# print python version
import sys
print(sys.version)
# print python path
import sys
print(sys.path)
print(">")

import pprint
from typing import List

import dm_env
import gym
import gymnasium
from absl.testing import absltest

import envpool


class _MakeTest(absltest.TestCase):

  def test_version(self) -> None:
    print(envpool.__version__)

  def test_list_all_envs(self) -> None:
    pprint.pprint(envpool.list_all_envs())

  def check_step(self, env_list: List[str]) -> None:
    for task_id in env_list:
      envpool.make_spec(task_id)
      env_gym = envpool.make_gym(task_id)
      env_dm = envpool.make_dm(task_id)
      env_gymnasium = envpool.make_gymnasium(task_id)
      print(env_dm)
      print(env_gym)
      print(env_gymnasium)
      self.assertIsInstance(env_gym, gym.Env)
      self.assertIsInstance(env_dm, dm_env.Environment)
      self.assertIsInstance(env_gymnasium, gymnasium.Env)
      env_dm.reset()
      env_gym.reset()
      env_gymnasium.reset()

  def test_make_classic(self) -> None:
    self.check_step(
      [
        "CartPole-v0",
        "CartPole-v1",
        "Pendulum-v0",
        "Pendulum-v1",
        "MountainCar-v0",
        "MountainCarContinuous-v0",
        "Acrobot-v1",
      ]
    )


if __name__ == "__main__":
  absltest.main()
