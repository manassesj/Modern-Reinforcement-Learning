import gym
import numpy as np
import matplotlib.pyplot as plt

#LEFT = 0 DOWN = 1 RIGHT = 2 UP = 3
#0S 1F 2F 3F
#4F 5H 6F 7H
#8F 9F 10F 11H
#12H 13F 14F 15G

policy = {0: 1, 1: 2, 2: 1, 3: 0, 4: 1, 6: 1, 8: 2, 9: 1,
          10: 1, 13: 2, 14: 2}
  
env = gym.make('FrozenLake-v0')
n_games = 1000
win_pct = []
scores = []
obss = []

for i in range(n_games):
    done = False
    obs = env.reset()
    score = 0
    while not done:
        action = policy[obs]
        obs, reward, done, info = env.step(action)
        obss.append(obs)
        score += reward
    scores.append(score)
    if i % 10:
        average = np.mean(scores[-10:])
        win_pct.append(average)

plt.plot(win_pct)
plt.show()
print(obss)