import copy
import numpy as np
import random
import time

# Directions
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

def get_action_name(action):
    if action == UP:
        return "U"
    if action == DOWN:
        return "D"
    if action == LEFT:
        return "L"
    if action == RIGHT:
        return "R"

OBSTACLES = [(1, 1)]
EXIT_STATE = (-1, -1)

def dict_argmax(d):   
# =============================================================================
#     max_value = max(d.values())
#     for k, v in d.items():
#         if v == max_value:
#             return k.any()
# =============================================================================
    return max(d, key=d.get)
        
class Grid:

    def __init__(self):
        self.x_size = 4
        self.y_size = 3
        self.p = 0.8
        self.actions = [UP, DOWN, LEFT, RIGHT]
        self.rewards = {(3, 1): -100, (3, 2): 1}
        self.discount = 0.9

        self.states = list((x, y) for x in range(self.x_size) for y in range(self.y_size))
        self.states.append(EXIT_STATE)
        for obstacle in OBSTACLES:
            self.states.remove(obstacle)

    def attempt_move(self, s, a):
        # s: (x, y), x = s[0], y = s[1]
        # a: {UP, DOWN, LEFT, RIGHT}

        x, y = s[0], s[1]

        # Check absorbing state
        if s in self.rewards:
            return EXIT_STATE

        if s == EXIT_STATE:
            return s

        # Default: no movement
        result = s 

        # Check borders
        """
        Write code here to check if applying an action 
        keeps the agent with the boundary
        """
        if a == LEFT and x > 0:
            result = (x - 1, y)
        if a == RIGHT and x < self.x_size - 1:
            result = (x + 1, y)
        if a == UP and y < self.y_size - 1:
            result = (x, y + 1)
        if a == DOWN and y > 0:
            result = (x, y - 1)

        # Check obstacle cells
        """
        Write code here to check if applying an action 
        moves the agent into an obstacle cell
        """
        if result in OBSTACLES:
            return s

        return result

    def stoch_action(self, a):
        # Stochasitc actions probability distributions
        if a == RIGHT: 
            stoch_a = {RIGHT: self.p , UP: (1-self.p)/2, DOWN: (1-self.p)/2}
        if a == UP:
            stoch_a = {UP: self.p , LEFT: (1-self.p)/2, RIGHT: (1-self.p)/2}
        if a == LEFT:
            stoch_a = {LEFT: self.p , UP: (1-self.p)/2, DOWN: (1-self.p)/2}
        if a == DOWN:
            stoch_a = {DOWN: self.p , LEFT: (1-self.p)/2, RIGHT: (1-self.p)/2}
        return stoch_a

    def get_reward(self, s):
        if s == EXIT_STATE:
            return 0

        if s in self.rewards:
            return self.rewards[s]
        else:
            return 0

class ValueIteration:
    def __init__(self, grid):
        self.grid = Grid()
        self.values = {state: 0 for state in self.grid.states}

    def next_iteration(self):
        new_values = dict()
        """
        Write code here to imlpement the VI value update
        Iterate over self.grid.states and self.grid.actions
        Use stoch_action(a) and attempt_move(s,a)
        """
        for s in self.grid.states:
            # Maximum value
            action_values = list()
            for a in self.grid.actions:
                total = 0
                for stoch_action, p in self.grid.stoch_action(a).items():
                    # Apply action
                    s_next = self.grid.attempt_move(s, stoch_action)
                    total += p * (self.grid.get_reward(s) + (self.grid.discount * self.values[s_next]))
                action_values.append(total)
            # Update state value with maximum
            new_values[s] = max(action_values)

        self.values = new_values

    def print_values(self):
        for state, value in self.values.items():
            print(state, value)



class PolicyIteration:
    def __init__(self, grid):
        self.grid = Grid()
        self.values = {state: 0 for state in self.grid.states}
        self.policy = {pi: RIGHT for pi in self.grid.states}
        self.k = 10
        self.converged = False

   
    def convergence_check(self, new_policy):
        """
        Write code to check if PI has converged here
        """   
        if self.policy == new_policy: # fragile
            self.converged = True
                   
    def next_evaluation_iteration(self):
        policy_values = dict()
        """
        Alternative to using linear algebra:
        Imlpement the value propagation update for MPI
        Iterate over self.grid.states and self.grid.actions for self.policy
        Use stoch_action(a) and attempt_move(s,a)
        """
        for s in self.grid.states:
            a = self.policy[s]
            policy_values[s] = 0.0
            for stoch_action, p in self.grid.stoch_action(a).items():
                # Apply action
                s_next = self.grid.attempt_move(s, stoch_action)
                policy_values[s] += p * (self.grid.get_reward(s) + (self.grid.discount * self.values[s_next]))
        self.values = policy_values  
        
    def policy_evaluation(self):
        """
        Using MPI to evaluate self.policy
        """                 
        for kk in range(self.k):
            self.next_evaluation_iteration()
            
    def policy_improvement(self):
        """
        Write code to extract the best policy for a given value function here
        """ 
        new_policy = dict()
        for s in self.grid.states:
            action_values = dict()
            for a in self.grid.actions:
                total = 0
                for stoch_action, p in self.grid.stoch_action(a).items():
                    # Apply action
                    s_next = self.grid.attempt_move(s, stoch_action)
                    total += p * (self.grid.get_reward(s) + (self.grid.discount * self.values[s_next]))
                action_values[a] = total
            # Update policy with argmax
            new_policy[s] = dict_argmax(action_values)
        self.convergence_check(new_policy)            
        self.policy = new_policy
        
    def print_values(self):
        for state, value in self.values.items():
            print(state, value)
    
    def print_policy(self):
        for state, policy in self.policy.items():
            print(state, policy)

    def print_values_and_policy(self):
        for state in self.grid.states:
            print(state, self.values[state], get_action_name(self.policy[state]))

def run_value_iteration(max_iter = 100):
    grid = Grid
    vi = ValueIteration(grid)

    start = time.time()
    print("Initial values:")
    vi.print_values()
    print()

    for i in range(max_iter):
        vi.next_iteration()
        print("Values after iteration", i + 1)
        vi.print_values()
        print()

    end = time.time()
    print("Time to copmlete", max_iter, "VI iterations")
    print(end - start)


def run_policy_iteration(max_iter = 100):
    grid = Grid
    pi = PolicyIteration(grid)

    start = time.time()
    print("Initial values and policy:")
    pi.print_values_and_policy()
    print()

    for i in range(max_iter):
        # pi.next_iteration()
        pi.policy_evaluation()
        pi.policy_improvement()
        print("Values and policy after iteration", i + 1)
        pi.print_values_and_policy()
        print()
        if pi.converged == True:
            print("Policy iteration has converged")
            break

    end = time.time()
    print("Time to copmlete", i+1, "PI iterations")
    print(end - start)
    
if __name__ == "__main__":
    run_value_iteration()
    run_policy_iteration()
