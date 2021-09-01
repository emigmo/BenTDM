# Copyright 2019 The ROBEL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Gym environment registration for DClaw environments."""

from robel.utils.registration import register

#===============================================================================
# Pose tasks
#===============================================================================

# Default number of steps per episode.
_POSE_EPISODE_LEN = 80  # 80*20*2.5ms = 4s

register(
    env_id='DClawPoseFixed-v0',
    class_path='robel.dclaw.pose:DClawPoseFixed',
    max_episode_steps=_POSE_EPISODE_LEN)

register(
    env_id='DClawPoseRandom-v0',
    class_path='robel.dclaw.pose:DClawPoseRandom',
    max_episode_steps=_POSE_EPISODE_LEN)

register(
    env_id='DClawPoseRandomDynamics-v0',
    class_path='robel.dclaw.pose:DClawPoseRandomDynamics',
    max_episode_steps=_POSE_EPISODE_LEN)

#===============================================================================
# Turn tasks with touch sensor
#===============================================================================

# Default number of steps per episode.
_TURN_EPISODE_LEN = 400  # 40*40*2.5ms = 4s

register(
    env_id='DClawTurnFixedTS-v0',
    class_path='robel.dclaw.turn:DClawTurnFixedTS',
    max_episode_steps=_TURN_EPISODE_LEN)

register(
    env_id='DClawTurnRandomTS-v0',
    class_path='robel.dclaw.turn:DClawTurnRandomTS',
    max_episode_steps=_TURN_EPISODE_LEN)

register(
    env_id='DClawTurnRandomDynamicsTS-v0',
    class_path='robel.dclaw.turn:DClawTurnRandomDynamicsTS',
    max_episode_steps=_TURN_EPISODE_LEN)

register(
    env_id='DClawTurnRandomDampTS-v0',
    class_path='robel.dclaw.turn:DClawTurnRandomDampTS',
    max_episode_steps=_TURN_EPISODE_LEN)

#===============================================================================
# Turn tasks
#===============================================================================

# Default number of steps per episode.
_TURN_EPISODE_LEN = 40  # 40*40*2.5ms = 4s

register(
    env_id='DClawTurnFixed-v0',
    class_path='robel.dclaw.turn:DClawTurnFixed',
    max_episode_steps=_TURN_EPISODE_LEN)

register(
    env_id='DClawTurnRandom-v0',
    class_path='robel.dclaw.turn:DClawTurnRandom',
    max_episode_steps=_TURN_EPISODE_LEN)

register(
    env_id='DClawTurnRandomDynamics-v0',
    class_path='robel.dclaw.turn:DClawTurnRandomDynamics',
    max_episode_steps=_TURN_EPISODE_LEN)

register(
    env_id='DClawTurnRandomDamp-v0',
    class_path='robel.dclaw.turn:DClawTurnRandomDamp',
    max_episode_steps=_TURN_EPISODE_LEN)

#===============================================================================
# Screw tasks with touch sensor
#===============================================================================

# Default number of steps per episode.
_SCREW_EPISODE_LEN = 80  # 80*40*2.5ms = 8s

register(
    env_id='DClawScrewFixedTS-v0',
    class_path='robel.dclaw.screw:DClawScrewFixedTS',
    max_episode_steps=_SCREW_EPISODE_LEN)

register(
    env_id='DClawScrewRandomTS-v0',
    class_path='robel.dclaw.screw:DClawScrewRandomTS',
    max_episode_steps=_SCREW_EPISODE_LEN)

register(
    env_id='DClawScrewRandomDynamicsTS-v0',
    class_path='robel.dclaw.screw:DClawScrewRandomDynamicsTS',
    max_episode_steps=_SCREW_EPISODE_LEN)

register(
    env_id='DClawScrewRandomDampTS-v0',
    class_path='robel.dclaw.screw:DClawScrewRandomDampTS',
    max_episode_steps=_SCREW_EPISODE_LEN)

#===============================================================================
# Screw tasks
#===============================================================================

# Default number of steps per episode.
_SCREW_EPISODE_LEN = 80  # 80*40*2.5ms = 8s

register(
    env_id='DClawScrewFixed-v0',
    class_path='robel.dclaw.screw:DClawScrewFixed',
    max_episode_steps=_SCREW_EPISODE_LEN)

register(
    env_id='DClawScrewRandom-v0',
    class_path='robel.dclaw.screw:DClawScrewRandom',
    max_episode_steps=_SCREW_EPISODE_LEN)

register(
    env_id='DClawScrewRandomDynamics-v0',
    class_path='robel.dclaw.screw:DClawScrewRandomDynamics',
    max_episode_steps=_SCREW_EPISODE_LEN)

register(
    env_id='DClawScrewRandomDamp-v0',
    class_path='robel.dclaw.screw:DClawScrewRandomDamp',
    max_episode_steps=_SCREW_EPISODE_LEN)

register(
    env_id='DClawScrewGivenDamp-v0',
    class_path='robel.dclaw.screw:DClawScrewGivenDamp',
    max_episode_steps=_SCREW_EPISODE_LEN)

register(
    env_id='DClawScrewGivenDampAndFric-v0',
    class_path='robel.dclaw.screw:DClawScrewGivenDampAndFric',
    max_episode_steps=_SCREW_EPISODE_LEN)

#===============================================================================
# Screw different valves
#===============================================================================

valves = ['D0', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14']
asset_paths = ['robel/dclaw/assets/dclaw3xh_valve{}_v1.xml'.format(i) for i in range(15)]
for (valve_index, asset_path_index) in zip(valves, asset_paths):
    register(
        env_id='DClawScrewFixed{}-v0'.format(valve_index),
        class_path='robel.dclaw.screw:DClawScrewFixed',
        max_episode_steps=_SCREW_EPISODE_LEN,
        kwargs={'asset_path': asset_path_index,}
        )

    register(
        env_id='DClawScrewRandom{}-v0'.format(valve_index),
        class_path='robel.dclaw.screw:DClawScrewRandom',
        max_episode_steps=_SCREW_EPISODE_LEN,
        kwargs={'asset_path': asset_path_index,}
        )

    register(
        env_id='DClawScrewRandomDynamics{}-v0'.format(valve_index),
        class_path='robel.dclaw.screw:DClawScrewRandomDynamics',
        max_episode_steps=_SCREW_EPISODE_LEN,
        kwargs={'asset_path': asset_path_index,}
        )

    register(
        env_id='DClawScrewRandomDamp{}-v0'.format(valve_index),
        class_path='robel.dclaw.screw:DClawScrewRandomDamp',
        max_episode_steps=_SCREW_EPISODE_LEN,
        kwargs={'asset_path': asset_path_index,}
        )

    register(
        env_id='DClawScrewGivenDamp{}-v0'.format(valve_index),
        class_path='robel.dclaw.screw:DClawScrewGivenDamp',
        max_episode_steps=_SCREW_EPISODE_LEN,
        kwargs={'asset_path': asset_path_index,}
        )

    register(
        env_id='DClawScrewFixed{}TS-v0'.format(valve_index),
        class_path='robel.dclaw.screw:DClawScrewFixedTS',
        max_episode_steps=_SCREW_EPISODE_LEN,
        kwargs={'asset_path': asset_path_index,}
        )

    register(
        env_id='DClawScrewRandom{}TS-v0'.format(valve_index),
        class_path='robel.dclaw.screw:DClawScrewRandomTS',
        max_episode_steps=_SCREW_EPISODE_LEN,
        kwargs={'asset_path': asset_path_index,}
        )

    register(
        env_id='DClawScrewRandomDynamics{}TS-v0'.format(valve_index),
        class_path='robel.dclaw.screw:DClawScrewRandomDynamicsTS',
        max_episode_steps=_SCREW_EPISODE_LEN,
        kwargs={'asset_path': asset_path_index,}
        )

    register(
        env_id='DClawScrewRandomDamp{}TS-v0'.format(valve_index),
        class_path='robel.dclaw.screw:DClawScrewRandomDampTS',
        max_episode_steps=_SCREW_EPISODE_LEN,
        kwargs={'asset_path': asset_path_index,}
        )
    register(
        env_id='DClawScrewGivenDamp{}TS-v0'.format(valve_index),
        class_path='robel.dclaw.screw:DClawScrewGivenDampTS',
        max_episode_steps=_SCREW_EPISODE_LEN,
        kwargs={'asset_path': asset_path_index,})

    register(
        env_id='DClawTurnFixed{}-v0'.format(valve_index),
        class_path='robel.dclaw.turn:DClawTurnFixed',
        max_episode_steps=_TURN_EPISODE_LEN,
        kwargs={'asset_path': asset_path_index,}
        )

    register(
        env_id='DClawTurnRandom{}-v0'.format(valve_index),
        class_path='robel.dclaw.turn:DClawTurnRandom',
        max_episode_steps=_TURN_EPISODE_LEN,
        kwargs={'asset_path': asset_path_index,}
        )

    register(
        env_id='DClawTurnRandomDynamics{}-v0'.format(valve_index),
        class_path='robel.dclaw.turn:DClawTurnRandomDynamics',
        max_episode_steps=_TURN_EPISODE_LEN,
        kwargs={'asset_path': asset_path_index,}
        )

    register(
        env_id='DClawTurnRandomDamp{}-v0'.format(valve_index),
        class_path='robel.dclaw.turn:DClawTurnRandomDamp',
        max_episode_steps=_TURN_EPISODE_LEN,
        kwargs={'asset_path': asset_path_index,}
        )

    register(
        env_id='DClawTurnFixed{}TS-v0'.format(valve_index),
        class_path='robel.dclaw.turn:DClawTurnFixedTS',
        max_episode_steps=_TURN_EPISODE_LEN,
        kwargs={'asset_path': asset_path_index,}
        )

    register(
        env_id='DClawTurnRandom{}TS-v0'.format(valve_index),
        class_path='robel.dclaw.turn:DClawTurnRandomTS',
        max_episode_steps=_TURN_EPISODE_LEN,
        kwargs={'asset_path': asset_path_index,}
        )

    register(
        env_id='DClawTurnRandomDynamics{}TS-v0'.format(valve_index),
        class_path='robel.dclaw.turn:DClawTurnRandomDynamicsTS',
        max_episode_steps=_TURN_EPISODE_LEN,
        kwargs={'asset_path': asset_path_index,}
        )

    register(
        env_id='DClawTurnRandomDamp{}TS-v0'.format(valve_index),
        class_path='robel.dclaw.turn:DClawTurnRandomDampTS',
        max_episode_steps=_TURN_EPISODE_LEN,
        kwargs={'asset_path': asset_path_index,}
        )
        