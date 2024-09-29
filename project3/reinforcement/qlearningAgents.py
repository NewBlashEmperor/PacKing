# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *
from backend import ReplayMemory

import nn
import model
import backend
import gridworld


import random,util,math
import numpy as np
import copy

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent
      Functions you should fill in:
        - computeValueFromQValues # 我觉得这个就是返回最大的QValues
        - computeActionFromQValues # 每个QValues都有对应的(state,action),那么可以直接返回其中的action
        - getQValue # 返回(state,action)对应的QValue
        - getAction # 不知道
        - update  # 不知道
      Instance variables you have access to
        - self.epsilon (exploration prob) # 用在后面的epsilon greedy policies
        - self.alpha (learning rate) # 用在q-learning 的 Q = alpha*sample + (1-alpha) * Q
        - self.discount (discount rate) # 用在 Qvalue iteration
      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        # Q-values 可以使用 util.counter() 作为一个容器，凭借(state,action)作为key，得到对应的q-value
        self.q_values = util.Counter()

    def getQValue(self, state, action): # 根据(state,action)拿到对应的q_value
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        if (state,action) not in self.q_values.keys(): # 当Q(state,action)没有对应的qvalues的时候
          return 0.0
        else:
          return self.q_values[(state,action)]
        util.raiseNotDefined()

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        legalActions = self.getLegalActions(state) # 拿到所有的合理的动作
        tmp = util.Counter() # 一个用于存储 (state,action)对应的Q-value的容器
        for action in legalActions:
            tmp[action] = self.getQValue(state,action)
        return tmp[tmp.argMax()] # tmp.argMax()得到的是最大的value所对应的action 
        util.raiseNotDefined()

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        # 我觉得和前面的computeValueFromQValues()差不多，只不过返回argMax就可以了
        legalActions = self.getLegalActions(state)
        tmp = util.Counter()
        for action in legalActions:
            tmp[action] = self.getQValue(state,action)
        return tmp.argMax()
        util.raiseNotDefined()

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.
          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        # 这个函数的目的就是通过epsilon来决定是采取一个随机的动作，还是选取Q_value最大的那个动作
        if util.flipCoin(self.epsilon):
          action = random.choice(legalActions) # 从里面随机采取一个动作
        else:
          action = self.computeActionFromQValues(state)
        return action
        util.raiseNotDefined()

    def update(self, state, action, nextState, reward: float): # 更新self.q_values
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here
          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        sample = reward + self.discount*self.computeValueFromQValues(nextState) # 拿到sample
        self.q_values[(state,action)] = (1-self.alpha) * self.q_values[(state,action)] + self.alpha*sample

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1
        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action

class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent
       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        features = self.featExtractor.getFeatures(state,action)
        return sum(features[item]*self.weights[item] for item in features)
        util.raiseNotDefined()

    def update(self, state, action, nextState, reward: float):
        """
           Should update your weights based on transition
        """
        diff = reward + self.discount*self.computeValueFromQValues(nextState) - self.getQValue(state,action)
        features = self.featExtractor.getFeatures(state,action)
        for item in features:
            self.weights[item] += self.alpha*diff*features[item]
        "*** YOUR CODE HERE ***"

    def final(self, state):
        """Called at the end of each game."""
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
