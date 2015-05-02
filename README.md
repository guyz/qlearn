# Reinforcement Learning using Q-Learning

An implementation of Q-Learning for training a machine to play Swingy Monkey (a Flappy Bird clone).

Both the game (SwingyMonkey.py) and the learning models are provided (qlearn_mX.py). The 3 different models are as follows:

1. qlearn_m1.py - World state is a function of ( horizontal_dist(monkey, tree), vertical_dist(monkey, dist), velocity(monkey) ).
2. qlearn_m2.py - Same model, but 2x the resolution of horizontal_dist and vertical_dist.
3. qlearn_m3.py - World state is ( horizontal_dist(monkey, tree), vertical_dist(monkey, dist), position(monkey), velocity(monkey) ). The resolution is as in #2.

## How to run
1. Make sure you have pygame installed.
2. use 'python qlearn_mX.py' to start training.

NOTE: qlearn_m1.py learns a pretty good representation after ~20 minutes (on a simple MBP system). qlearn_m2.py takes several hours to converge to its optimal policy, and qlearn_m3.py was still improving after ~18h of runtime, but by that time it is already better than the first two models (and it keeps improving, although very slowly). See analysis folder for some charts (generated after all three models were run for ~18 hours).

