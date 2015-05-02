import numpy as np
import numpy.random as npr
import sys

from time import gmtime, strftime

from SwingyMonkey import SwingyMonkey

SCREEN_WIDTH  = 600
SCREEN_HEIGHT = 400
BINSIZE = 25 # Pixels per bin
GAMMA = 0.9 # Discount factor
VSTATES = 5 # Velocity states
EPS_FACTOR = 0.001 # Start e-greedy factor

def v_discrete(v):
    pass

class Learner:

    def __init__(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.epoch = 1
        # self.Q = np.zeros((2,SCREEN_WIDTH/BINSIZE+1,SCREEN_HEIGHT/BINSIZE+1,VSTATES))
        self.Q = np.zeros((2,SCREEN_WIDTH/BINSIZE+1,SCREEN_HEIGHT/BINSIZE+1,VSTATES))
        self.k = np.zeros((2,SCREEN_WIDTH/BINSIZE+1,SCREEN_HEIGHT/BINSIZE+1,VSTATES)) # number of times action a has been taken from state s
        self.iters = 0
        self.mem = [0, 0]
        self.scores = []
        self.best_score = 50
        self.bestQ = None

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.epoch += 1

    def action_callback(self, state):
        self.iters += 1
        '''Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.'''

        # You might do some learning here based on the current state and the last state.

        # You'll need to take an action, too, and return it.
        # Return 0 to swing and 1 to jump.

        '''
        Q matrix: 
        ndarray of dimensions A X D x T x M
        A: <action space: 0 or 1>
        D: <pixels to next tree trunk>
        T: <screen height of bottom of tree trunk>
        M: <screen height of bottom of monkey>
        '''

        # current state
        D = state['tree']['dist'] / BINSIZE
        if D < 0:
            D = 0
        T = (state['tree']['top']-state['monkey']['top']+0) / BINSIZE
        V = state['monkey']['vel'] / 20

        if np.abs(V) > 2:
            V = np.sign(V)*2

        def default_action(p=0.5):
            return 1 if npr.rand() < p else 0

        new_action = default_action()
        if not self.last_action == None:
            # previous state
            d = self.last_state['tree']['dist'] / BINSIZE
            t = (self.last_state['tree']['top']-self.last_state['monkey']['top']+0) / BINSIZE
            v = self.last_state['monkey']['vel'] / 20

            if np.abs(v) > 2:
                v = np.sign(v)*2

            max_Q = np.max(self.Q[:,D,T,V])
            new_action = 1 if self.Q[1][D,T,V] > self.Q[0][D,T,V] else 0
            
            # epsilon-greedy
            if self.k[new_action][D,T,V] > 0:
                eps = EPS_FACTOR/self.k[new_action][D,T,V]
            else:
                eps = EPS_FACTOR

            if (npr.rand() < eps):
                new_action = default_action()

            ALPHA = 1/self.k[self.last_action][d,t,v]
            self.Q[self.last_action][d,t,v] += ALPHA*(self.last_reward+GAMMA*max_Q-self.Q[self.last_action][d,t,v])

        self.mem[0] = state['monkey']['top']

        self.last_action = new_action
        self.last_state  = state
        self.k[new_action][D,T,V] += 1
        return new_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''

        self.last_reward = reward

iters = 10000
learner = Learner()

for ii in xrange(iters):

    # Make a new monkey object.
    swing = SwingyMonkey(sound=False,            # Don't play sounds.
                         tick_length=1,          # Make game ticks super fast.
                         # Display the epoch on screen and % of Q matrix filled
                         text="Epoch %d " % (ii) + str(round(float(np.count_nonzero(learner.Q))*100/learner.Q.size,3)) + "%", 
                         action_callback=learner.action_callback,
                         reward_callback=learner.reward_callback)

    # Loop until you hit something.
    while swing.game_loop():
        pass

    # Keep track of the score for that epoch.
    learner.scores.append(learner.last_state['score'])
    if learner.last_state['score'] > learner.best_score:
        print 'New best Q'
        learner.best_score = learner.last_state['score']
        learner.bestQ = learner.Q.copy()

    print 'score %d' % learner.last_state['score'], str(round(float(np.count_nonzero(learner.Q))*100/learner.Q.size,3)) + "%"

    # Reset the state of the learner.
    learner.reset()

print np.mean(scores)
print learner.imputed
# np.savetxt(strftime("out/%m-%d %H:%M:%S", gmtime())+'-'+str(BINSIZE)+'-'+str(GAMMA)+".txt", scores)
