# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]
    # consider both the food and the ghosts' position
    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # using the md between pacman and dots
        # get the food position
        foodPos = newFood.asList()
        MD1 = []
        # get the manhattan distance between the pacman and the food
        for food in foodPos:
            md1 = util.manhattanDistance(food,newPos)
            MD1.append(md1)
        # using the md between pacman and ghost
        # get the ghost position
        MD2 = []
        for ghost in newGhostStates:
            md2 = util.manhattanDistance(ghost.getPosition(),newPos)
            if newScaredTimes != 0: # 当不在恐吓时间的时候，这个时候与鬼的曼哈顿距离才有意义
                MD2.append(md2)
        if len(MD1) == 0 and min(MD2) != 0: #最后一步是个豆子，且上面没有鬼
            return (1000000)
        else: # 最后上是一个豆子，但上面是个鬼
            if len(MD1) == 0:
                return (-10000)
        # 这个score受到 与豆子的曼哈顿距离 与鬼的曼哈顿距离的影响，最终返回的值要尽可能大
        return min(MD2)/min(MD1)**1.5 + 2*successorGameState.getScore()

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """

    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def max_value(state:GameState,depth):
            depth = depth + 1
            v = -float('inf')
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            for action in state.getLegalActions(0):
                if action != "stop":
                    # get the successorState of the pacman
                    successorState = state.generateSuccessor(self.index,action)
                    v = max(v,min_value(successorState,depth,1))
            return v
        
        def min_value(state:GameState,depth,ghostNum):
            v = float('inf')
            if state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            for action in state.getLegalActions(ghostNum):
                # get the successorState of the ghost
                if action != "stop":
                    successorState = state.generateSuccessor(ghostNum,action)
                    if ghostNum == state.getNumAgents() -1 :
                        v = min(v,max_value(successorState,depth))
                    else:
                        v = min(v,min_value(successorState,depth,ghostNum+1))
            return v
        v = -float('inf')
        for action in gameState.getLegalActions(0): # get the legal action for the pacman
            if action != "stop":
                successorState = gameState.generateSuccessor(0,action)
                value = max(v,min_value(successorState,0,1))
                if value > v:
                    v = value
                    result = action
        return result
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    # when call this function, it aims at generating a action for the pacman
    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def max_value(state:GameState,alpha,beta,depth,actionList):
            depth = depth+1
            # initialize v = negatively infinity
            v = -float('inf')
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            # get the successorGameState after each legal action
            action = ''
            for a in state.getLegalActions(0):
                successorGameState = state.generateSuccessor(0,a)
                value = min_value(successorGameState,alpha,beta,depth,actionList,1)
                if value>v:
                    v = value
                    action = a
                if v > beta:
                    actionList[0] = action
                    return v
                alpha = max(v,alpha)
            actionList[0] = action
            return v
        def min_value(state:GameState,alpha,beta,depth,actionList,ghostNum):
            v = float('inf')
            if state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            # get the successorGameState 
            for a in state.getLegalActions(ghostNum):
                successorGameState = state.generateSuccessor(ghostNum,a)
                # when this is the final ghost
                if ghostNum == state.getNumAgents() -1 :
                    value = max_value(successorGameState,alpha,beta,depth,actionList)
                else:
                    value = min_value(successorGameState,alpha,beta,depth,actionList,ghostNum+1)
                if value < v: # v must be the minimum value
                    v = value
                if v < alpha:
                    return v
                beta = min(v,beta)
            return v
        actionList = [""]
        max_value(gameState,-float('inf'),float('inf'),-1,actionList)
        return actionList[0]
        util.raiseNotDefined()
       
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # each successorState has the same chance
        def max_value(state:GameState,depth):
            depth = depth + 1
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            v = -float('inf')
            # get the legal action for the pacman
            for action in state.getLegalActions(0):
                # get the successorGameState according to the action
                successorGameState = state.generateSuccessor(0,action)
                v = max(v,exp_value(successorGameState,depth,1))
            return v
        
        def exp_value(state:GameState,depth,ghostNum):
            if state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            # initialize v = 0
            v = 0
            count = len(state.getLegalActions(ghostNum))
            # get the leaglAction for the ghost
            for action in state.getLegalActions(ghostNum):
                successorGameState = state.generateSuccessor(ghostNum,action)
                # the final ghost
                if ghostNum == state.getNumAgents() -1:
                    v += max_value(successorGameState,depth)
                else:
                    v += exp_value(successorGameState,depth,ghostNum+1)
            return v/count
        v = -float('inf')
        result = ''
        for action in gameState.getLegalActions(0):
            successorGameState = gameState.generateSuccessor(0,action)
            value = exp_value(successorGameState,0,1)
            if value > v:
                v = value
                result = action
        return result      
        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # Useful information you can extract from a GameState (pacman.py)
    # the pos of the pacman
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostState = currentGameState.getGhostStates()
    foodList = newFood.asList() # get the list of available food

    # get the current score
    score = currentGameState.getScore()

    # set different value of the action
    Food_Weight = 10.0
    Ghost_Weight = -20.0
    Scared_Ghost_Weight = 80

    # get the manhattan distance between the pacman and the food
    distanceToFoodList = [util.manhattanDistance(newPos,foodPos)for foodPos in newFood.asList()]
    if len(distanceToFoodList) == 0:
        score += Food_Weight
    else:
        score += Food_Weight/min(distanceToFoodList)

    # get the manhattan distance between the pacman and the ghost
    for ghost in newGhostState:
        ghostDistance = util.manhattanDistance(ghost.getPosition(),newPos)
        if ghostDistance > 0:
            # check whether the ghosts are in panic
            if ghost.scaredTimer > 0:
                score += Scared_Ghost_Weight / ghostDistance
            # run away 
            else:
                score += Ghost_Weight / ghostDistance
        else: # dead
            return -float('inf')
    return score


    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
