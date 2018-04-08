## CSC 665 - Multi-agent Pacman Search
#### Khanh Nguyen
This project has 2 parts: 

- Implements the evaluation function for Pacman as a Reflex Agent to escape the Ghost(s) while eating as many dots as possible, and the basic adversarial multi-agents using Minimax.
- Implements the adversarial multi-agents using Minimax with Alpha-Beta Pruning, Expectimax, Expectimax with improved evaluation function.

## Overview

* [Link to the project's original specs](http://ai.berkeley.edu/multiagent.html)
* With the new game setup, Pacman now needs to find its way out from being captured by ghost agents. For part 1 of this project, the program will be implementing Pacman to act as a **Reflex Agent** and a smarter **Adversarial Agent** using **Minimax** strategy. For part 2, Pacman will build upon the Minimax Agent in order to improve picking out the maximum achievable state taking into account the effects from minimizer ghost agent(s).
* The search agents can be found in `multiagent/multiAgents.py`
* The pacman can be found in `multiagent/pacman.py`
* The utilities class can be found in `multiagent/util.py`

## Problem Statements
### Part 1 - Reflex Agent + Minimax Agent
* Improve the ReflexAgent in `multiAgents.py` to play respectably. The provided reflex agent code provides some helpful examples of methods that query the GameState for information. A capable reflex agent will have to consider both food locations and ghost locations to perform well.

* Write an adversarial search agent in the provided MinimaxAgent class stub in `multiAgents.py`. Your minimax agent should work with any number of ghosts, so you'll have to write an algorithm that is slightly more general than what you've previously seen in lecture. In particular, your minimax tree will have multiple min layers (one for each ghost) for every max layer. Your code should also expand the game tree to an arbitrary depth. Score the leaves of your minimax tree with the supplied `self.evaluationFunction`, which defaults to `scoreEvaluationFunction`. `MinimaxAgent` extends `MultiAgentSearchAgent`, which gives access to `self.depth` and `self.evaluationFunction`. Make sure your minimax code makes reference to these two variables where appropriate as these variables are populated in response to command line options.

### Part 2 - Alpha-Beta Pruning + Expectimax + Improved Evaluation Function
* Make a new agent that uses alpha-beta pruning to more efficiently explore the minimax tree, in `AlphaBetaAgent`. Your algorithm will be slightly more general than the pseudocode from lecture, so part of the challenge is to extend the alpha-beta pruning logic appropriately to multiple minimizer agents.

* Minimax and alpha-beta are great, but they both assume that you are playing against an adversary who makes optimal decisions. As anyone who has ever won tic-tac-toe can tell you, this is not always the case. In this question you will implement the `ExpectimaxAgent`, which is useful for modeling probabilistic behavior of agents who may make suboptimal choices.

* Write a better evaluation function for pacman in the provided function `betterEvaluationFunction`. The evaluation function should evaluate states, rather than actions like your reflex agent evaluation function did. You may use any tools at your disposal for evaluation, including your search code from the last project. With depth 2 search, your evaluation function should clear the `smallClassic` layout with one random ghost more than half the time and still run at a reasonable rate (to get full credit, Pacman should be averaging around 1000 points when he's winning).

## How to run
* Download or git clone this repo if not already
* In terminal, go to directory:
	*  `cd path/to/CSC665-multi-agent-pacman/multiagent/`
	
* To test **Reflex Agent**:
	* `python autograder.py -q q1`
	* `python autograder.py -q q1 --no-graphics` *(to run without graphics)*
	* `python pacman.py -p ReflexAgent -l testClassic`

* To test **Minimax**:
	* `python autograder.py -q q2`
	
* To test **Alpha-Beta Pruning**:
	* `python autograder.py -q q3`
	* `python pacman.py -p AlphaBetaAgent -a depth=3 -l smallClassic`
	
* To test **Expectimax**:
	* `python autograder.py -q q4`
	* `ppython pacman.py -p ExpectimaxAgent -l minimaxClassic -a depth=3`
	
* To test **Evaluation Function**:
	* `python autograder.py -q q5`	
	
## Solution Design

### Reflex Agent - Evaluation Function

```python
# in multiagents.py
class ReflexAgent(Agent):

    def evaluationFunction(self, currentGameState, action):
		
```    

- Given the list of foods and ghosts, Pacman can easily find the distances to such item or agents on the map. To enhance the evaluation function, Pacman needs to find what would be the immediate best action to take, based on the score from this function.

### Minimax

```python
# in multiagents.py
class MinimaxAgent(MultiAgentSearchAgent):

    def getAction(self, gameState):
        # Format of result = [score, action]
        result = self.get_value(gameState, 0, 0)

        # Return the action from result
        return result[1]
            
    def get_value(self, gameState, index, depth):
    	# implementation

    def max_value(self, gameState, index, depth):
    	# implementation

    def min_value(self, gameState, index, depth):
    	# implementation
		
```    

- In order for Minimax to determine which action to take, it will need to search in this adversarial agents tree - minimax - to maximize the utiliy for Pacman if the agent is a maximizer (Pacman), and minimize the utility for Pacman if the agent is a minimizer (Ghost).
- The logic for get_value is straightforward and very much close to the provided pseudocode:
	- If terminal states: return utility value (gameState.score() or evaluationFunction)
	- If maximizer agent: calls max_value
	- If minimizer agent: calls min_value   
- The logic for max_value and min_value are very similar since they both have to iterate through a list of legal moves for the current gameState and call get_value() recursively to obtain the utility value. The only difference is that maximizer agent will want to find the max value from each action, and the reverse for a minimizer agent.


### Alpha-Beta Pruning

```python
# in multiagents.py
class AlphaBetaAgent(MultiAgentSearchAgent):

    def getAction(self, game_state):
        # Format of result = [action, score]
        # Initial state: index = 0, depth = 0, alpha = -infinity, beta = +infinity
        result = self.getBestActionAndScore(game_state, 0, 0, float("-inf"), float("inf"))

        # Return the action from result
        return result[0]
            
    def getBestActionAndScore(self, game_state, index, depth, alpha, beta):
    	# implementation

    def max_value(self, game_state, index, depth):
    	# implementation

    def min_value(self, game_state, index, depth):
    	# implementation
		
```    

- In order for Minimax to improve its performance by not having to expand unnecessary nodes, alpha-beta pruning will help determine which branches to be omitted and not being explored.
- The logic for getBestActionAndScore is straightforward and very much close to the provided pseudocode:
	- If terminal states: return utility value (gameState.score() or evaluationFunction)
	- If maximizer agent: calls max_value
	- If minimizer agent: calls min_value   
- The logic for max_value and min_value are very similar since they both have to iterate through a list of legal moves for the current gameState and call get\_value() recursively to obtain the utility value. The only difference is that the maximizer agent will update the alpha value accordingly to the most recently found value from the list of actions. Also, maximizer will break early if the newly found max value is greater than beta; this is the pruning logic -- if max value > beta, this agent can possibly find even greater max value later on, which essentially won't be considered when they got returned back up to the minimizer agent. The reverse is same for min_value with updating beta and pruning using alpha values. 


### Expectimax

```python
# in multiagents.py
class ExpectimaxAgent(MultiAgentSearchAgent):

    def getAction(self, game_state):
        # Format of result = [action, score]
        action, score = self.get_value(game_state, 0, 0)
        return action
            
    def get_value(self, game_state, index, depth):
    	# implementation

    def max_value(self, game_state, index, depth):
    	# implementation

    def expected_value(self, game_state, index, depth):
    	# implementation
		
```    

- With Expectimax, we don't always assume that the minimizer agents will try to optimize their actions; therefore, we include the aspect of probability to capture this nature.
- The logic for get_value is straightforward and very much close to the provided pseudocode:
	- If terminal states: return utility value from evaluationFunction()
	- If maximizer agent: calls max_value
	- If expectation agent: calls expected_value   
- The logic for max_value and expected_value are very similar since they both have to iterate through a list of legal moves for the current gameState and call get\_value() recursively to obtain the utility value. The only difference is that maximizer agent will want to find the max value from each action. And the expectation agent will try to use uniformly distributed probability to obtain the utility value instead of using the minimum value like Minimax Agent.

### Improved Evaluation Function

```python
# in multiagents.py
def betterEvaluationFunction(currentGameState):
		
```    

- To further improve the performance for Expectimax, this new eval function will calculate the utility differently based on the features and their corresponding weights based off my observation and intution.
- For instance, distance to closest food would have high priority, so I use the reciprocal `1.0/closestFood` as one of the features to tremendously affect the utility score.


## Solution Implementation

### Part 1

**REFLEX AGENT**

* The relevant files for this part are `multiagents.py ` and `pacman.py`.

* Essentially, the key to the implementation for this function is to use the **reciprocal** of important values (such as distance to food) rather than just the values themselves. 
* This would tremendously affect the score for each action based on the distance to foods/ghost.
* In other words, instead of using a linear combination of distances, the function can utilize this relationship **closestGhostDistance / closestFoodDistance**

**MINIMAX AGENT**

* The relevant files for this part are `multiagents.py ` and `pacman.py`.

* Essentially, the key to the implementation for this agent is the knowing how to update the correct agent's index and depth depending on whether it's a pacman or ghost.

### Part 2

**MINIMAX AGENT WITH ALPHA-BETA PRUNING**

* The relevant files for this part are `multiagents.py ` and `pacman.py`.

* The key to this implementation is basically the pruning logic in max_value and min_value, where it has to keep comparing to the beta or alpha value accordingly to ensure the current max or min value won't further worsen the already min or max value along the path from root.

**EXPECTIMAX AGENT**

* The relevant files for this part are `multiagents.py ` and `pacman.py`.

* Essentially, the key to the implementation for this agent is using the uniform distribution to calculate the probablity for an action, and use them to calcuate the expected value:
	* `successor_probability = 1.0 / len(legalMoves)`
	* `expected_value += successor_probability * current_value`

**EXPECTIMAX AGENT WITH IMPROVED EVAL FUNCTION**

* The relevant files for this part are `multiagents.py ` and `pacman.py`.

* Essentially, the key to the implementation for this new eval function is to pick the right features and weights, very much same idea to the eval function in **Reflex Agent**.

## Solution Result/Evaluation
* My program was able to successfully enhance the scores for Pacman to eat more dots while avoid the ghosts. The average score from 10 games running by the `autograder.py` was in the 500 threshold.
* My program also passed all the other tests for part 2 which includes the implementation for the adversarial agents.
* As mentioned in the project prompt, Pacman indeed will die in some test cases, but because the logic it follows is correctly implemented, the tests will still pass.
* One notable thing in the improved eval function I did was that because I used the reciprocal on closest distance to food, the average from all 10 games was able to reach the threshold of 1000.
	* Output from running question 5:

```
Question q5
===========

Pacman emerges victorious! Score: 1344
Pacman emerges victorious! Score: 1244
Pacman emerges victorious! Score: 1049
Pacman emerges victorious! Score: 883
Pacman emerges victorious! Score: 826
Pacman emerges victorious! Score: 1284
Pacman emerges victorious! Score: 1088
Pacman emerges victorious! Score: 964
Pacman emerges victorious! Score: 794
Pacman emerges victorious! Score: 1269
Average Score: 1074.5
Scores:        1344.0, 1244.0, 1049.0, 883.0, 826.0, 1284.0, 1088.0, 964.0, 794.0, 1269.0
Win Rate:      10/10 (1.00)
Record:        Win, Win, Win, Win, Win, Win, Win, Win, Win, Win

``` 

## Conclusion
* This can be thought of a simpler way of thinking of heuristics function, which was the majority of project 1.
* In the future, evaluation function can be enhanced by adding characteristics of heuristics function.
* Reflex Agent, although easy to compute, is not smart or optimal enough. Therefore, we include in the second part of this project to utilize the minimax algorithm.
* Adversarial Agents definitely defeat the Reflex Agent in its way of finding smarter way to "predict" its components' future moves.