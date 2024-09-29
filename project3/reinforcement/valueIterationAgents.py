# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp: mdp.MarkovDecisionProcess, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        """
          Run the value iteration algorithm. Note that in standard
          value iteration, V_k+1(...) depends on V_k(...)'s.
        """
        "*** YOUR CODE HERE ***"
        for i in range(self.iterations): # 总共迭代iterations次
            # 在每一次迭代中，要对所有的状态进行修改
            # 先拿到所有的状态
            states = self.mdp.getStates()
            temp_counter = util.Counter() # 用来记录每个state 所对应的value
            for state in states:
                # 每个state take action后有一个Q-state,那么我的value就是最大的那个q-value
                # 当当前的节点是端点的时候，那么最大的value = 0
                if len(self.mdp.getPossibleActions(state)) == 0:
                    maxVal = 0
                # 否则则进行计算
                else:
                    maxVal = -float('inf')
                    for action in self.mdp.getPossibleActions(state):
                        Q = self.computeQValueFromValues(state,action) # 计算所有的Q值
                        if Q > maxVal:
                            maxVal = Q 
                temp_counter[state] = maxVal
            self.values = temp_counter



    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        # 完成上面使用的计算Qvale的函数 其实就是计算后续所有的可能的状态并求和
        total = 0 
        for nextstate,pro in self.mdp.getTransitionStatesAndProbs(state,action):
            total += pro*(self.mdp.getReward(state,action,nextstate) + self.discount*self.getValue(nextstate))
        return total
        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # 这个函数的目的就是给你一个状态，叫你得出这个状态下的最优的动作
        # 实现方法就是找到这个状态下的所有的动作，比较每个动作对应的Q值，用Q值最大的那个动作
        best_action = None
        max_value = -float('inf')
        for action in self.mdp.getPossibleActions(state):
            Q = self.computeQValueFromValues(state,action)
            if Q > max_value:
                max_value = Q
                best_action = action # 根据最大的value进行更新
        return best_action
        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
