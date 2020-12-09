from laser_tank import LaserTankMap, DotDict
import numpy as np
import time

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
MAX_EPISODE = 40000

class Solver:

    def __init__(self, learning_rate=0.01):
        """
        Initialise solver without a Q-value table.
        """
        self.learning_rate = learning_rate
        self.exploit_prob = 0.7  # 'epsilon' in epsilon-greedy
        self.AvgR_list = []

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

        if np.random.uniform() > self.exploit_prob:
            action_name = np.random.choice(LaserTankMap.MOVES)
        else:
            # we random select if multiple action have same max value
            actions = []
            # maximum value
            maximum = max(state_actions.values())
            for action, value in state_actions.items():
                if value == maximum:
                    actions.append(action)
            action_name = np.random.choice(actions)
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

        epis_counter = 0
        iter_counter = 0
        Rinside_list = [0]
        R_list = []
        AvgR_list = []

        while epis_counter < MAX_EPISODE and (time.time() - start) < simulator.time_limit:
            epis_counter += 1
            is_terminated = False
            current_map = simulator.make_clone()
            S = current_map

            R_list.append(np.mean(Rinside_list))
            while len(R_list) == 50:
                AvgR_list.append(np.mean(R_list))
                R_list.pop(0)

            Rinside_list = []
            while not is_terminated:
                iter_counter += 1
                # RL choose action based on observation
                A = self.choose_action(S)
                # print("chosen action: ", A)

                # RL take action and get next observation and reward
                oldhash = hash(S) # newMap = S.make_clone()
                R, is_over = S.apply_move(A)

                Rinside_list.append(R)

                # RL learn from this transition
                self.check_state_exist(S)
                q_predict = self.q_values[oldhash][A]
                # print("q_predict: ", q_predict)
                if not is_over:
                    q_target = R + S.gamma * max(self.q_values[hash(S)].values())  # next state is not terminal
                else:
                    q_target = R  # next state is terminal
                    is_terminated = True  # terminate this episode

                self.q_values[oldhash][A] += self.learning_rate * (q_target - q_predict)  # update
                # move to next state s->s'

                if time.time() - start > simulator.time_limit:
                    break

        self.AvgR_list = AvgR_list
        print("epis_counter: ", epis_counter)
        print("iter_counter: ", iter_counter)
        print("learning_rate: ", self.learning_rate)
        print("AvgR_list len: ", len(AvgR_list))
        print(time.time() - start)

    def train_sarsa(self, simulator):
        """
        Train the agent using SARSA, building up a table of Q-values.
        :param simulator: A simulator for collecting episode data (LaserTankMap instance)
        PSEUDOCODE:
        Initialize Q(s,a) arbitrarily
        Repeat (for each episode):
            Initialize s
            Choose a from s using policy derived from Q
            Repeat (for each step of episode):
                Take action a, observe r, s'
                Choose a' from s' using policy derived from Q
                Q(s,a) <- Q(s,a) + alpha*[r + gamma * Q(s',a') - Q(s,a)]
                s <- s'; a <- a'
            Until s is terminal
        """

        # Q(s, a) table
        # suggested format: key = hash(state), value = dict(mapping actions to values)
        start = time.time()
        # Q(s, a) table
        # suggested format: key = hash(state), value = dict(mapping actions to values)
        self.q_values = {hash(simulator): {}}

        # initial state and action (must be legal)
        for action in LaserTankMap.MOVES:
            self.q_values[hash(simulator)][action] = 0
        # print(self.q_values[hash(simulator)])

        epis_counter = 0
        iter_counter = 0
        Rinside_list = [0]
        R_list = []
        AvgR_list = []

        while epis_counter < MAX_EPISODE and (time.time() - start) < simulator.time_limit:
            epis_counter += 1
            # initial observation
            is_terminated = False
            current_map = simulator.make_clone()
            S = current_map
            # RL choose action based on observation
            A = self.choose_action(S)

            R_list.append(np.mean(Rinside_list))
            while len(R_list) == 50:
                AvgR_list.append(np.mean(R_list))
                R_list.pop(0)

            Rinside_list = []
            while not is_terminated:
                iter_counter += 1
                # RL take action and get next observation and reward
                # newMap = S.make_clone()
                oldHash = hash(S)
                R, is_over = S.apply_move(A)

                Rinside_list.append(R)

                # RL choose action based on next observation
                A_ = self.choose_action(S)

                # RL learn from this transition (s, a, r, s, a) --> Sarsa
                self.check_state_exist(S)
                q_predict = self.q_values[oldHash][A]
                if not is_over:
                    q_target = R + S.gamma * self.q_values[hash(S)][A_]
                else:
                    q_target = R
                    is_terminated = True

                self.q_values[oldHash][A] += self.learning_rate * (q_target - q_predict)  # update
                # move to next state
                A = A_

                if time.time() - start > simulator.time_limit:
                    break

        self.AvgR_list = AvgR_list
        print("epis_counter: ", epis_counter)
        print("iter_counter: ", iter_counter)
        print("learning_rate: ", self.learning_rate)
        print("AvgR_list len: ", len(AvgR_list))
        print(time.time() - start)

    def get_policy(self, state):
        """
        Get the policy for this state (i.e. the action that should be performed at this state).
        :param state: a LaserTankMap instance
        :return: pi(s) [an element of LaserTankMap.MOVES]
        """
        state_actions = self.q_values[hash(state)]
        actions = []
        # maximum value
        maximum = max(state_actions.values())
        for action, value in state_actions.items():
            if value == maximum:
                actions.append(action)
        action_name = np.random.choice(actions)

        return action_name

    def get_list(self):
        """
        Get the policy for this state (i.e. the action that should be performed at this state).
        :param state: a LaserTankMap instance
        :return: pi(s) [an element of LaserTankMap.MOVES]
        """
        return self.AvgR_list
