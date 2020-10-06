from laser_tank import LaserTankMap, DotDict

"""
Template file for you to implement your solution to Assignment 3. You should implement your solution by filling in the
following method stubs:
    run_value_iteration()
    run_policy_iteration()
    get_offline_value()
    get_offline_policy()
    get_mcts_policy()
    
You may add to the __init__ method if required, and can add additional helper methods and classes if you wish.

To ensure your code is handled correctly by the autograder, you should avoid using any try-except blocks in your
implementation of the above methods (as this can interfere with our time-out handling).

COMP3702 2020 Assignment 3 Support Code
"""


class Solver:

    def __init__(self, game_map):
        self.game_map = game_map

        #
        # TODO
        # Write any environment preprocessing code you need here (e.g. storing teleport locations).
        #
        # You may also add any instance variables (e.g. root node of MCTS tree) here.
        #
        # The allowed time for this method is 1 second, so your Value Iteration or Policy Iteration implementation
        # should be in the methods below, not here.
        #

        self.values = None
        self.policy = None

    def run_value_iteration(self):
        """
        Build a value table and a policy table using value iteration, and store inside self.values and self.policy.
        """
        values = [[[0 for _ in LaserTankMap.DIRECTIONS]
                   for __ in range(1, self.game_map.y_size - 1)]
                  for ___ in range(1, self.game_map.x_size - 1)]
        policy = [[[-1 for _ in LaserTankMap.DIRECTIONS]
                   for __ in range(1, self.game_map.y_size - 1)]
                  for ___ in range(1, self.game_map.x_size - 1)]

        #
        # TODO
        # Write your Value Iteration implementation here.
        #
        # When this method is called, you are allowed up to [state.time_limit] seconds of compute time. You should stop
        # iterating either when max_delta < epsilon, or when the time limit is reached, whichever occurs first.
        #

        # store the computed values and policy
        self.values = values
        self.policy = policy

    def run_policy_iteration(self):
        """
        Build a value table and a policy table using policy iteration, and store inside self.values and self.policy.
        """
        values = [[[0 for _ in LaserTankMap.DIRECTIONS]
                   for __ in range(1, self.game_map.y_size - 1)]
                  for ___ in range(1, self.game_map.x_size - 1)]
        policy = [[[-1 for _ in LaserTankMap.DIRECTIONS]
                   for __ in range(1, self.game_map.y_size - 1)]
                  for ___ in range(1, self.game_map.x_size - 1)]

        #
        # TODO
        # Write your Policy Iteration implementation here.
        #
        # When this method is called, you are allowed up to [state.time_limit] seconds of compute time. You should stop
        # iterating either when max_delta < epsilon, or when the time limit is reached, whichever occurs first.
        #

        # store the computed values and policy
        self.values = values
        self.policy = policy

    def get_offline_value(self, state):
        """
        Get the value of this state.
        :param state: a LaserTankMap instance
        :return: V(s) [a floating point number]
        """

        #
        # TODO
        # Write code to return the value of this state based on the stored self.values
        #
        # You can assume that either run_value_iteration( ) or run_policy_iteration( ) has been called before this
        # method is called.
        #
        # When this method is called, you are allowed up to 1 second of compute time.
        #

        pass

    def get_offline_policy(self, state):
        """
        Get the policy for this state (i.e. the action that should be performed at this state).
        :param state: a LaserTankMap instance
        :return: pi(s) [an element of LaserTankMap.MOVES]
        """

        #
        # TODO
        # Write code to return the optimal action to be performed at this state based on the stored self.policy
        #
        # You can assume that either run_value_iteration( ) or run_policy_iteration( ) has been called before this
        # method is called.
        #
        # When this method is called, you are allowed up to 1 second of compute time.
        #

        pass

    def get_mcts_policy(self, state):
        """
        Choose an action to be performed using online MCTS.
        :param state: a LaserTankMap instance
        :return: pi(s) [an element of LaserTankMap.MOVES]
        """

        #
        # TODO
        # Write your Monte-Carlo Tree Search implementation here.
        #
        # Each time this method is called, you are allowed up to [state.time_limit] seconds of compute time - make sure
        # you stop searching before this time limit is reached.
        #

        pass







