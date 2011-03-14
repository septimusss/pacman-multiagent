# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance
from game import Directions
import random, util
import math

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
    
    #if the pacman ate the power pellet, he does not consider the ghost
    if newScaredTimes[0] == 0 :
        #if the pacman's new position is near ghost(Manhattan Distance equals 1 or 0), the new position has the lowest score
        for ghost in newGhostStates:
            if manhattanDistance(newPos,ghost.getPosition()) <= 1:
                return -999999
    
    #if the pacman's new position is a food position, the new position will get the highest score
    if len(successorGameState.getFood().asList()) < len(currentGameState.getFood().asList()):
        return 999999
    
    #calculate the distances between each food and pacman
    foodToPacmanDistList = [manhattanDistance(newPos, food) for food in newFood.asList()]
    
    #find the shortest distance
    shortestFoodToPacmanDist = min(foodToPacmanDistList)

    return 1/shortestFoodToPacmanDist
    

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
    
    

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (question 2)
  """


  def MaxValue(self,gameState,depth):
      #terminal test of game state
      if depth == 0 or gameState.isWin() or gameState.isLose():
          return [self.evaluationFunction(gameState),"none"]
      
      value = [-999999,"none"]
      
      #get pacman's action
      pacmanActionList = [action for action in gameState.getLegalActions(0) if action != Directions.STOP]
      #get the game state after pacman moves
      pacmanSuccessorList = [gameState.generateSuccessor(0, action) for action in pacmanActionList]
      
      count = 0
      
      depth -= 1
      for gameState in pacmanSuccessorList:
          tmpValue = self.MinValue(gameState,depth)
          if value[0] < tmpValue:
              value = [tmpValue,pacmanActionList[count]]
          count +=1
          
          
      return value
  
  

  
  def MinValue(self,gameState,depth):
      #terminal test of game state
      if gameState.isWin() or gameState.isLose():
          return self.evaluationFunction(gameState)
      
      value = 999999
      
      ghostActionList = []
      ghostSuccessorList = []
      ghostSuccessorList.append(gameState)
      
      #get the number of the ghosts
      ghostNum = gameState.getNumAgents()-1
      
      #store the final game state after these ghosts move
      ghostFinalState = []
      
      
      #get the game state after ghosts move
      for ghostIndex in range(1,ghostNum+1):
          tempSuccessorList = []
          for gameState in ghostSuccessorList:
              #if the ghost's move makes the Pacman Game come to an end,store the game state. No need for other ghosts to move.
              if gameState.isLose() or gameState.isWin():
                  ghostFinalState.append(gameState)
                  continue;
              ghostActionList = [action for action in gameState.getLegalActions(ghostIndex)]
              tempSuccessorList += [gameState.generateSuccessor(ghostIndex, action) for action in ghostActionList]
          ghostSuccessorList = tempSuccessorList[:]


      ghostFinalState += ghostSuccessorList[:]
      
      
      for gameState in ghostFinalState:
          tmpValue = self.MaxValue(gameState,depth)
          value = min(value,tmpValue[0])

      
      return value
      
    
  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
    "*** YOUR CODE HERE ***"

    value = self.MaxValue(gameState,self.depth)
    
    return value[1] 

alpha = -999999
beta = 999999

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"
    value = self.MaxValue(gameState,self.depth,alpha,beta)

    return value[1] 
    
    

  def MaxValue(self,gameState,depth,alpha,beta):
      #terminal test of game state
      if depth == 0 or gameState.isWin() or gameState.isLose():
          return [self.evaluationFunction(gameState),"none"]
      
      value = [-999999,"none"]
      
      #get pacman's action
      pacmanActionList = [action for action in gameState.getLegalActions(0) if action != Directions.STOP]
      #get the game state after pacman moves
      pacmanSuccessorList = [gameState.generateSuccessor(0, action) for action in pacmanActionList]
      
      count = 0
      
      depth -= 1
      for gameState in pacmanSuccessorList:
          tmpValue = self.MinValue(gameState,depth,alpha,beta)
          if value[0] < tmpValue:
              value = [tmpValue,pacmanActionList[count]]
          count +=1
          
          if value[0] >= beta:
              return value
          
          alpha = max(alpha,value[0])
          
          
      return value
  
  
  
  def MinValue(self,gameState,depth,alpha,beta):
      #terminal test of game state
      if gameState.isWin() or gameState.isLose():
          return self.evaluationFunction(gameState)
      
      value = 999999
      
      ghostActionList = []
      ghostSuccessorList = []
      ghostSuccessorList.append(gameState)
      
      #get the number of the ghosts
      ghostNum = gameState.getNumAgents()-1
      
      #store the final game state after these ghosts move
      ghostFinalState = []
      
      
      #get the game state after ghosts move
      for ghostIndex in range(1,ghostNum+1):
          tempSuccessorList = []
          for gameState in ghostSuccessorList:
              #if the ghost's move makes the Pacman Game come to an end,store the game state. No need for other ghosts to move.
              if gameState.isLose() or gameState.isWin():
                  ghostFinalState.append(gameState)
                  continue;
              ghostActionList = [action for action in gameState.getLegalActions(ghostIndex)]
              tempSuccessorList += [gameState.generateSuccessor(ghostIndex, action) for action in ghostActionList]
          ghostSuccessorList = tempSuccessorList[:]


      ghostFinalState += ghostSuccessorList[:]
      
      
      for gameState in ghostFinalState:
          tmpValue = self.MaxValue(gameState,depth,alpha,beta)
          value = min(value,tmpValue[0])
          
          if value <= alpha:
              return value
          
          beta = min(beta,value)
          

      return value


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
    
    value = self.ExpectMaxValue(gameState,self.depth)

    return value[1]



    
  def ExpectMaxValue(self,gameState,depth):
      #terminal test of game state
      if depth == 0 or gameState.isWin() or gameState.isLose():
          return [self.evaluationFunction(gameState),"none"]
      
      value = [-999999,"none"]
      
      #get pacman's action
      pacmanActionList = [action for action in gameState.getLegalActions(0) if action != Directions.STOP]
      #get the game state after pacman moves
      pacmanSuccessorList = [gameState.generateSuccessor(0, action) for action in pacmanActionList]
      
      count = 0
      
      depth -= 1
      for gameState in pacmanSuccessorList:
          tmpValue = self.ExpectMinValue(gameState,depth)
          if value[0] < tmpValue:
              value = [tmpValue,pacmanActionList[count]]
          count +=1
          
          
      return value
  
  
  
  def ExpectMinValue(self,gameState,depth):
      #terminal test of game state
      if gameState.isWin() or gameState.isLose():
          return self.evaluationFunction(gameState)
      
      value = 0
      
      ghostActionList = []
      ghostSuccessorList = []
      ghostSuccessorList.append(gameState)
      
      #get the number of the ghosts
      ghostNum = gameState.getNumAgents()-1
      
      #store the final game state after these ghosts move
      ghostFinalState = []
      
      
      #get the game state after ghosts move
      for ghostIndex in range(1,ghostNum+1):
          tempSuccessorList = []
          for gameState in ghostSuccessorList:
              #if the ghost's move makes the Pacman Game come to an end,store the game state. No need for other ghosts to move.
              if gameState.isLose() or gameState.isWin():
                  ghostFinalState.append(gameState)
                  continue;
              ghostActionList = [action for action in gameState.getLegalActions(ghostIndex)]
              tempSuccessorList += [gameState.generateSuccessor(ghostIndex, action) for action in ghostActionList]
          ghostSuccessorList = tempSuccessorList[:]


      ghostFinalState += ghostSuccessorList[:]
      
      
      for gameState in ghostFinalState:
          tmpValue = self.ExpectMaxValue(gameState,depth)
          value += tmpValue[0]
          #value = min(value,tmpValue[0])

      
      value /= len(ghostFinalState)
      
      return value
    

def minDistanceToGoal(pacmanPosition, foodList):
 
  
  xy1 = pacmanPosition
  distance = 0.0

  if(len(foodList) == 0):
      return distance
  
  #calculate which food is the nearest one to the pacman
  minDistanceFood = foodList[0]
  minDistanceToFood =  abs(xy1[0] - minDistanceFood[0]) + abs(xy1[1] - minDistanceFood[1]) 
  
  for food in foodList:
      xy2 = food
      tmpDistance =  abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])
      if(tmpDistance < minDistanceToFood):
          minDistanceToFood = tmpDistance
          minDistanceFood = xy2
  
  foodList.remove(minDistanceFood)

  
  #calculate which food is the nearest to the minDistanceFood,for example, FoodA
  #then, calculate which food is the nearest food to FoodA, and so on...
  while len(foodList)>0:
      dist = 999999
      fd = []
      for food in foodList:
          tmpDist = abs(food[0]-minDistanceFood[0]) + abs(food[1]-minDistanceFood[1])
          if(dist > tmpDist):
              dist = tmpDist
              fd = food
      minDistanceToFood += dist
      minDistanceFood = fd
      foodList.remove(fd)

      
  return minDistanceToFood

def MSTDistance(foodList,pacmanPosition):
 
  
  #use MST:Minimum Spanning Tree to solve the problem
  #use the all the food and pacman to form a MST (Prim Algorithm)
  

  distance = 0.0

  if(len(foodList) == 0):
      return distance
  
  #calculate the distance between each node,using manhattan distance
  allNodesDistanceList = []
  allNodesList = foodList[:]
  
  allNodesList.insert(0,pacmanPosition)
  
  for foodX in allNodesList:
      nodeDistanceDict = {}
      for foodY in allNodesList:
          nodeDistanceDict[foodY] = abs(foodX[0]-foodY[0]) + abs(foodX[1] - foodY[1])
      allNodesDistanceList.append(nodeDistanceDict)
  
  
  #store the shortest distance from the node to the MST
  lowcost = []    
  
  #store which node is the nearest one to that node
  closest = []
  for node in allNodesList:
      lowcost.append(allNodesDistanceList[0][node])
      closest.append(pacmanPosition)
  
  count = 1
  while count < len(allNodesList):
      tmpLowCost = lowcost[:]
      tmpLowCost.sort(cmp=None, key=None, reverse=False)
      tmpSubLowCost = tmpLowCost[count:]
      
      lowCostNodeIndex = lowcost.index(tmpSubLowCost[0])
      lowcost[lowCostNodeIndex] = 0
      
      theLowCostNode = allNodesList[lowCostNodeIndex]
      
      #update the values of lowcost and closest
      for node in allNodesList:
          if allNodesDistanceList[lowCostNodeIndex][node] < lowcost[allNodesList.index(node)]:
              lowcost[allNodesList.index(node)] = allNodesDistanceList[lowCostNodeIndex][node]
              closest[allNodesList.index(node)] = theLowCostNode
      
      count += 1
  
  index = 0
  #calculate sum of each edge in MST
  for node in allNodesList:
      distance += allNodesDistanceList[index][closest[index]]
      index += 1

  return distance
     

def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
  """
  "*** YOUR CODE HERE ***"
  
  #get the ghosts' states
  ghostStates = [gameState for gameState in currentGameState.getGhostStates()]
  #get the ghosts' positions
  ghostPositionList = [ gameState.getPosition() for gameState in ghostStates]
  #get the coordinates of current food
  foodList = currentGameState.getFood().asList()
  #get pacman's position
  pacmanPosition = currentGameState.getPacmanPosition()
  #get the times of pellet
  newScaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

  
  distToFood = 0.0001
  foodWeight = 20
  ghostWeight = -0.5
  currentScore = currentGameState.getScore()
  powerPelletFactor = 0
  
  #get the sum of distance from pacman to nearer ghost and the distance between two ghosts
  distToGhost = minDistanceToGoal(pacmanPosition,ghostPositionList)
  
  #calculate the distance between each food to pacman
  foodToPacmanDistList = []
  for food in foodList:
     foodToPacmanDistList.append(manhattanDistance(food,pacmanPosition))
  
  #sort the distance      
  foodToPacmanDistList.sort(cmp=None, key=None, reverse=False)
  

  #count the k nearest food's distances to pacman and sum them up
  knn = 4
  
  if len(foodToPacmanDistList) < knn:
      knn = len(foodToPacmanDistList)
                
  for i in range(knn):
      distToFood += foodToPacmanDistList[i]
     
     
     
  #if the power pellet is actived, the pacman is the most powerful agent
  if newScaredTimes[0] != 0 :
      powerPelletFactor = 999999      

  
  return currentScore + 1.0/distToFood * foodWeight + powerPelletFactor + distToGhost * ghostWeight
  




# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """

  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    "*** YOUR CODE HERE ***"
    value = self.ExpectMaxValue(gameState,2)

    return value[1]



    
  def ExpectMaxValue(self,gameState,depth):
      #terminal test of game state
      if depth == 0 or gameState.isWin() or gameState.isLose():
          return [evaluationFunctionForContest(gameState),"none"]
      
      value = [-999999,"none"]
      
      #get pacman's action
      pacmanActionList = [action for action in gameState.getLegalActions(0) if action != Directions.STOP]
      #get the game state after pacman moves
      pacmanSuccessorList = [gameState.generateSuccessor(0, action) for action in pacmanActionList]
      
      count = 0
      
      depth -= 1
      for gameState in pacmanSuccessorList:
          tmpValue = self.ExpectMinValue(gameState,depth)
          if value[0] < tmpValue:
              value = [tmpValue,pacmanActionList[count]]
          count +=1
          
          
      return value
  
  
  
  def ExpectMinValue(self,gameState,depth):
      #terminal test of game state
      if gameState.isWin() or gameState.isLose():
          return evaluationFunctionForContest(gameState)
      
      value = 0
      
      ghostActionList = []
      ghostSuccessorList = []
      ghostSuccessorList.append(gameState)
      
      #get the number of the ghosts
      ghostNum = gameState.getNumAgents()-1
      
      #store the final game state after these ghosts move
      ghostFinalState = []
      
      
      #get the game state after ghosts move
      for ghostIndex in range(1,ghostNum+1):
          tempSuccessorList = []
          for gameState in ghostSuccessorList:
              #if the ghost's move makes the Pacman Game come to an end,store the game state. No need for other ghosts to move.
              if gameState.isLose() or gameState.isWin():
                  ghostFinalState.append(gameState)
                  continue;
              ghostActionList = [action for action in gameState.getLegalActions(ghostIndex)]
              tempSuccessorList += [gameState.generateSuccessor(ghostIndex, action) for action in ghostActionList]
          ghostSuccessorList = tempSuccessorList[:]


      ghostFinalState += ghostSuccessorList[:]
      
      
      for gameState in ghostFinalState:
          tmpValue = self.ExpectMaxValue(gameState,depth)
          value += tmpValue[0]
          #value = min(value,tmpValue[0])

      
      value /= len(ghostFinalState)
      
      return value

def evaluationFunctionForContest(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
  """
  "*** YOUR CODE HERE ***"
  
  #get the ghosts' states
  ghostStates = [gameState for gameState in currentGameState.getGhostStates()]
  #get the ghosts' positions
  ghostPositionList = [ gameState.getPosition() for gameState in ghostStates]
  #get the coordinates of current food
  foodList = currentGameState.getFood().asList()
  #get pacman's position
  pacmanPosition = currentGameState.getPacmanPosition()
  #get the times of pellet
  newScaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

  
  distToFood = 0.0001
  foodWeight = 100
  ghostWeight = -0.01
  currentScore = currentGameState.getScore()
  powerPelletFactor = 0
  
  #get the sum of distance from pacman to nearer ghost and the distance between two ghosts
  distToGhost = minDistanceToGoal(pacmanPosition,ghostPositionList)
  
  #calculate the distance between each food to pacman
  foodToPacmanDistList = []
  for food in foodList:
     foodToPacmanDistList.append(manhattanDistance(food,pacmanPosition))
  
  #sort the distance      
  foodToPacmanDistList.sort(cmp=None, key=None, reverse=False)
  

  #count the k nearest food's distances to pacman and sum them up
  knn = 4
  
  if len(foodToPacmanDistList) < knn:
      knn = len(foodToPacmanDistList)
                
  for i in range(knn):
      distToFood += foodToPacmanDistList[i]
     
     
     
  #if the power pellet is actived, the pacman is the most powerful agent
  if newScaredTimes[0] != 0 :
      powerPelletFactor = 999999      

  
  return currentScore + 1.0/distToFood * foodWeight + powerPelletFactor + distToGhost * ghostWeight