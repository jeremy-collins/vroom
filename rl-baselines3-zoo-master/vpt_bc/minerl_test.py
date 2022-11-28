import gym
import minerl

'''
cd /home/codysoccerman/Documents/classes/Fall_22/Deep_Learning/Project/rl-baselines3-zoo-master
conda activate vpt
cd basalt-2022-behavioural-cloning-baseline
python3 minerl_test.py

'''

#env = gym.make('MineRLBasaltFindCave-v0')
env = gym.make("PandaPickAndPlace-v1")

obs = env.reset()

print(obs)

done = False

while not done:
    # Take a random action
    action = env.action_space.sample()
    # In BASALT environments, sending ESC action will end the episode
    # Lets not do that
    action["ESC"] = 0
    obs, reward, done, _ = env.step(action)
    env.render()