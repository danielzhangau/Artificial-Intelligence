from laser_tank import LaserTankMap, DotDict
import numpy as np
import operator
import time
import random

"""
Template file for you to implement your solution to Assignment 4. You should implement your solution by filling in the
following method stubs:
    train_q_learning()
    train_sarsa()
    get_policy()
    
You may add to the __init__ method if required, and can add additional helper methods and classes if you wish.

To ensure your code is handled correctly by the autograder, you should avoid using any try-except blocks in your
implementation of the above methods (as this can interfere with our time-out handling).

COMP3702 2020 Assignment 4 Support Code
"""
MAX_EPISODE = 100

class Solver:

    def __init__(self):
        """
        Initialise solver without a Q-value table.
        """
        self.learning_rate = 1

        #
        # TODO
        # You may add code here if you wish (e.g. define constants used by both methods).
        #
        # The allowed time for this method is 1 second.
        #

        self.q_values = None

    def check_state_exist(self, state):
        if hash(state) not in self.q_values:
            self.q_values[hash(state)] = {}
            for action in LaserTankMap.MOVES:
                self.q_values[hash(state)][action] = 0

    def choose_action(self, state):
        self.check_state_exist(state)

        state_actions = self.q_values[hash(state)]
        # print(self.q_values[hash(state)])

        # we random select if multiple action have same max value
        actions = []
        # maximum value
        maximum = max(state_actions.values())
        for action, value in state_actions.items():
            if value == maximum:
                actions.append(action)
        action_name = random.choice(actions)
        # action_name = max(state_actions.items(), key=operator.itemgetter(1))[0]
        return action_name

    def train_q_learning(self, simulator):
        """
        Train the agent using Q-learning, building up a table of Q-values.
        :param simulator: A simulator for collecting episode data (LaserTankMap instance)
        PSEUDOCODE:
        Initialize Q(s,a) arbitrarily
        Repeat (for each episode):
            Initialize s
            Repeat (for each step of episode):
                Choose a from s using policy derived from Q (e.g. gamma_greedy)
                Take action a, observe r, s'
                Q(s,a) <- Q(s,a) + alpha[r + discount*maxQ(s',a') - Q(s,a)]
                s <- s'
            until s is terminal
        """
        start = time.time()
        # Q(s, a) table
        # suggested format: key = hash(state), value = dict(mapping actions to values)
        self.q_values = {hash(simulator): {}}

        # initial state and action (must be legal)
        for action in LaserTankMap.MOVES:
            self.q_values[hash(simulator)][action] = 0
        # print(self.q_values[hash(simulator)])

        while time.time() - start < simulator.time_limit:
            step_counter = 0
            is_terminated = False
            current_map = simulator.make_clone()
            S = current_map
            while not is_terminated:
                # RL choose action based on observation
                A = self.choose_action(S)
                # print("chosen action: ", A)

                # RL take action and get next observation and reward
                newMap = S.make_clone()
                R, is_over = newMap.apply_move(A)

                # RL learn from this transition
                self.check_state_exist(newMap)
                q_predict = self.q_values[hash(S)][A]
                # print("q_predict: ", q_predict)
                if not is_over:
                    q_target = R + S.gamma * max(
                        self.q_values[hash(newMap)].values())  # next state is not terminal
                else:
                    q_target = R  # next state is terminal
                    is_terminated = True  # terminate this episode

                self.q_values[hash(S)][A] += self.learning_rate * (q_target - q_predict)  # update
                S = newMap  # move to next state

                step_counter += 1

                if time.time() - start > simulator.time_limit:
                    break

        #
        # TODO
        # Write your Q-Learning implementation here.
        #
        # When this method is called, you are allowed up to [state.time_limit] seconds of compute time. You should
        # continue training until the time limit is reached.
        #

    def train_sarsa(self, simulator):
        """
        Train the agent using SARSA, building up a table of Q-values.
        :param simulator: A simulator for collecting episode data (LaserTankMap instance)
        """

        # Q(s, a) table
        # suggested format: key = hash(state), value = dict(mapping actions to values)
        q_values = {}

        #
        # TODO
        # Write your SARSA implementation here.
        #
        # When this method is called, you are allowed up to [state.time_limit] seconds of compute time. You should
        # continue training until the time limit is reached.
        #

        # store the computed Q-values
        self.q_values = q_values

    def get_policy(self, state):
        """
        Get the policy for this state (i.e. the action that should be performed at this state).
        :param state: a LaserTankMap instance
        :return: pi(s) [an element of LaserTankMap.MOVES]
        """
        # value = dict(mapping actions to values)
        # state_actions = self.q_values[hash(state)]
        # self.check_state_exist(state)
        # action_name = max(state_actions.items(), key=operator.itemgetter(1))[0]
        action = self.choose_action(state)
        return action

        # return max(value, key=value.get)
        #
        # TODO
        # You can assume that either train_q_learning( ) or train_sarsa( ) has been called before this
        # method is called.

        pass
