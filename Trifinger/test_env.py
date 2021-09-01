from causal_world.envs import CausalWorld
from causal_world.task_generators.task import task_generator
import numpy as np

def make_env(task_id):
    task = task_generator(task_generator_id=task_id,
                         tool_block_mass=0.2,
                         tool_block_size=0.13,
                         )
    env = CausalWorld(task=task, 
                      action_mode="joint_torques", 
                      enable_visualization=True)
    return env

# how to do more intervention
# success_signal, obs = env.do_intervention(
#         {'stage_color': np.random.uniform(0, 1, [3])})
# print("Intervention success signal", success_signal)

tasks = ['picking', 'pushing']

for task_id in tasks:
    
    env = make_env(task_id)
    print("Env: TriFinger --- Task: {} --- State_dim: {} --- Action_dim: {}".format(
        task_id, env.observation_space.shape[0], env.action_space.shape[0]))
    for _ in range(10):
        env.reset()
        done = False
        step = 0
        while not done:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            step += 1
            # env.render()
        print(step)
    env.close()