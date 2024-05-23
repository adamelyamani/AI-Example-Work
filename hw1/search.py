# search.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""

import util
from util import heappush, heappop


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
      Returns the start state for the search problem
      """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
      state: Search state

      Returns True if and only if the state is a valid goal state
      """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
      state: Search state

      For a given state, this should return a list of triples,
      (successor, action, stepCost), where 'successor' is a
      successor to the current state, 'action' is the action
      required to get there, and 'stepCost' is the incremental
      cost of expanding to that successor
      """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
      actions: A list of actions to take

      This method returns the total cost of a particular sequence of actions.  The sequence must
      be composed of legal moves
      """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.
    Your search algorithm needs to return a list of actions that reaches
    the goal. Make sure that you implement the graph search version of DFS,
    which avoids expanding any already visited states. 
    Otherwise your implementation may run infinitely!
    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """

    # 'visited' contains the nodes that have already been analyzed
    visited = {}
    # 'output' contains the sequence of nodes to reach the goal
    output = []
    # 'stack' contains the triplets provided by getSuccessors
    stack = util.Stack()
    # 'parents' contains the nodes and their parent nodes
    parents = {}

    # obtain starting node and push it onto the stack
    start = problem.getStartState()
    stack.push((start, 'Undefined', 0))
    visited[start] = 'Undefined'

    # check is starting node is the goal
    if problem.isGoalState(start):
        return output

    # go through each of the nodes until the goal is reached or there are no more nodes
    goal = False
    while stack.isEmpty() != True and goal != True:
        node = stack.pop()
        # store the visited node and the direction
        visited[node[0]] = node[1]
        # check if goal has been reached
        if problem.isGoalState(node[0]):
            path_node = node[0]
            goal = True
            break
        # expand the node and search the child nodes
        for _node in problem.getSuccessors(node[0]):
            if _node[0] not in visited.keys():
                parents[_node[0]] = node[0]
                stack.push(_node)

    # find the path for Pac-Man
    while path_node in parents.keys():
        path_node_prev = parents[path_node]
        output.insert(0, visited[path_node])
        path_node = path_node_prev

    return output


def breadthFirstSearch(problem):
    # 'visited' contains the nodes that have already been analyzed
    visited = {}
    # 'output' contains the sequence of nodes to reach the goal
    output = []
    # 'queue' contains the triplets provided by getSuccessors
    queue = util.Queue()
    # 'parents' contains the nodes and their parent nodes
    parents = {}

    # obtain starting node
    start = problem.getStartState()
    queue.push((start, 'Undefined', 0))
    visited[start] = 'Undefined'

    # check if starting node is goal
    if problem.isGoalState(start):
        return output

    # go through each of the nodes until the goal is reached or there are no more nodes
    goal = False
    while queue.isEmpty() != True and goal != True:
        node = queue.pop()
        # store the visited node and the direction
        visited[node[0]] = node[1]
        # check if node is goal
        if problem.isGoalState(node[0]):
            path_node = node[0]
            goal = True
            break
        # expand each node
        for _node in problem.getSuccessors(node[0]):
            # if the child has not been visited or expanded then add it to the queue
            if _node[0] not in visited.keys() and _node[0] not in parents.keys():
                parents[_node[0]] = node[0]
                queue.push(_node)

    # find the path for Pac-Man
    while path_node in parents.keys():
        path_node_prev = parents[path_node]
        output.insert(0, visited[path_node])
        path_node = path_node_prev

    return output


def uniformCostSearch(problem):
    # 'visited' contains the nodes that have already been analyzed
    visited = {}
    # 'output' contains the sequence of nodes to reach the goal
    output = []
    # 'queue' contains the triplets provided by getSuccessors
    queue = util.PriorityQueue()
    # 'parents' contains the nodes and their parent nodes
    parents = {}
    # 'cost' contains the costs of each node
    cost = {}

    # obtain starting node
    start = problem.getStartState()
    queue.push((start, 'Undefined', 0), 0)
    visited[start] = 'Undefined'
    cost[start] = 0

    # check if starting node is goal
    if problem.isGoalState(start):
        return output

    # go through each of the nodes until the goal is reached or there are no more nodes
    goal = False
    while queue.isEmpty() != True and goal != True:
        node = queue.pop()
        # store the visited node and its direction
        visited[node[0]] = node[1]
        # check if node is goal
        if problem.isGoalState(node[0]):
            path_node = node[0]
            goal = True
            break
        # expand each node and update the costs if children haven't been visited
        for _node in problem.getSuccessors(node[0]):
            if _node[0] not in visited.keys():
                new_cost = node[2] + _node[2]
                # if the new cost is more or if the child cost was calculated earlier then exit loop
                if _node[0] in cost.keys():
                    if cost[_node[0]] <= new_cost:
                        continue
                # else, add node to queue and update cost
                queue.push((_node[0], _node[1], new_cost), new_cost)
                cost[_node[0]] = new_cost
                parents[_node[0]] = node[0]

    # find the path for Pac-Man
    while path_node in parents.keys():
        path_node_prev = parents[path_node]
        output.insert(0, visited[path_node])
        path_node = path_node_prev

    return output


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    # 'visited' contains the nodes that have already been analyzed
    visited = {}
    # 'output' contains the sequence of nodes to reach the goal
    output = []
    # 'queue' contains the triplets provided by getSuccessors
    queue = util.PriorityQueue()
    # 'parents' contains the nodes and their parent nodes
    parents = {}
    # 'cost' contains the costs of each node
    cost = {}

    # obtain starting node
    start = problem.getStartState()
    queue.push((start, 'Undefined', 0), 0)
    visited[start] = 'Undefined'
    cost[start] = 0

    # check if starting node is goal
    if problem.isGoalState(start):
        return output

    # go through each of the nodes until the goal is reached or there are no more nodes
    goal = False
    while queue.isEmpty() != True and goal != True:
        node = queue.pop()
        # store each node and its direction
        visited[node[0]] = node[1]
        # check if node is goal
        if problem.isGoalState(node[0]):
            path_node = node[0]
            goal = True
            break
        # expand each node and update the costs if children haven't been visited
        for _node in problem.getSuccessors(node[0]):
            if _node[0] not in visited.keys():
                new_cost = node[2] + _node[2] + heuristic(_node[0], problem)
                # if the new cost is more or if the child cost was calculated earlier then exit loop
                if _node[0] in cost.keys():
                    if cost[_node[0]] <= new_cost:
                        continue
                # else, add node to queue and update cost
                queue.push((_node[0], _node[1], _node[2] + node[2]), new_cost)
                cost[_node[0]] = new_cost
                parents[_node[0]] = node[0]

    # find the path for Pac-Man
    while path_node in parents.keys():
        path_node_prev = parents[path_node]
        output.insert(0, visited[path_node])
        path_node = path_node_prev

    return output


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
