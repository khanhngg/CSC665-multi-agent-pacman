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
import copy

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
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

    def evaluationFunction(self, currentGameState, action):
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
        score = 0

        # # Maximize distance from pacman to ghost
        # newGhostPositions = successorGameState.getGhostPositions()
        # ghostDistances = [manhattanDistance(newPos, ghostPosition) for ghostPosition in newGhostPositions]
        # closestGhost = min(ghostDistances)

        closestGhostPosition = newGhostStates[0].configuration.pos
        closestGhost = manhattanDistance(newPos, closestGhostPosition)

        # Minimize distance from pacman to food
        newFoodPositions = newFood.asList()
        foodDistances = [manhattanDistance(newPos, foodPosition) for foodPosition in newFoodPositions]

        if len(foodDistances) == 0:
            return 0

        closestFood = min(foodDistances)

        # Stop action would reduce score because of the pacman's timer constraint
        if action == 'Stop':
            score -= 50

        return successorGameState.getScore() + closestGhost/(closestFood * 10) + score

def scoreEvaluationFunction(currentGameState):
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
        self.level = 0

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
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
        """
        "*** YOUR CODE HERE ***"

        result = self.get_value(gameState)
        return result[1]

    def get_value(self, gameState):
        """
        Returns value as pair of [score, action] based on the different cases:
        1. Terminal state
        2. Max-agent
        3. Min-agent
        """
        # WORKING STUFF, KEEP THIS FOR NOW
        # if gameState.state in gameState.problem.evaluation:
        #     if self.index == 0:
        #         self.depth -= 1
        #
        #     if self.depth == 0:
        #         # print "   KINDA WIN STATE---self.level=", self.level, " state=", gameState.state, " SCORE=", gameState.getScore()
        #         return [gameState.getScore(), ""]

        # NOT WORK FOR SOME 7-* and 8 cases ...
        try:
            if self.evaluationFunction(gameState):
                if self.index == 0:
                    self.depth -= 1

                if self.depth == 0:
                    print "   KINDA WIN STATE---self.level=", self.level, " state=", gameState.state, " SCORE=", gameState.getScore()
                    return [gameState.getScore(), ""]

        except Exception:
            pass

        if gameState.isWin():
            print "   WIN STATE---self.level=", self.level, " state=", gameState.state, " SCORE=", gameState.getScore()
            return [gameState.getScore(), ""]

        if gameState.isLose():
            return None

        # Max-agent: Pacman has index = 0
        if self.index == 0:
            return self.max_value(gameState)

        # Min-agent:Ghost has index > 0
        else:
            return self.min_value(gameState)

    def max_value(self, gameState):
        """
        Returns the max utility value for max-agent
        """
        legalMoves = gameState.getLegalActions(self.index)
        value = float("-inf")
        max_action = ""

        successor_agent_level = self.level + 1
        print "   MAX---self.level=", self.level, " state=", gameState.state

        for action in legalMoves:
            successor = gameState.generateSuccessor(self.index, action)
            successor_agent = copy.deepcopy(self)

            # ghost agent has > 0 index
            successor_agent.level = successor_agent_level
            if successor_agent_level % gameState.getNumAgents() != 0:
                successor_agent.index = 1
            else:
                successor_agent.index = 0

            print "   -----successor_agent.level=",successor_agent.level, " successor_agent.index=",successor_agent.index

            current_value = successor_agent.get_value(successor)
            if current_value is not None and current_value[0] > value:
                value = current_value[0]
                max_action = action

        print " ---> max val=", value, " action=", max_action
        return [value, max_action]

    def min_value(self, gameState):
        """
        Returns the min utility value for min-agent
        """
        legalMoves = gameState.getLegalActions(self.index)
        value = float("inf")
        min_action = ""

        successor_agent_level = self.level + 1
        print "   MIN---self.level=", self.level, " state=", gameState.state

        for action in legalMoves:
            successor = gameState.generateSuccessor(self.index, action)
            successor_agent = copy.deepcopy(self)

            # pacman agent has 0 index
            successor_agent.level = successor_agent_level
            if successor_agent_level % gameState.getNumAgents() != 0:
                successor_agent.index = 1
            else:
                successor_agent.index = 0

            print "   -----successor_agent.level=",successor_agent.level, " successor_agent.index=",successor_agent.index

            current_value = successor_agent.get_value(successor)
            if current_value is not None and current_value[0] < value:
                value = current_value[0]
                min_action = action

        print " ---> min val=", value, " action=", min_action
        return [value, min_action]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

