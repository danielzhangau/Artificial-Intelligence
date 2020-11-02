import time
from laser_tank import LaserTankMap, DotDict
import numpy as np
import copy

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

MOVES = [LaserTankMap.MOVE_FORWARD, LaserTankMap.TURN_LEFT, LaserTankMap.TURN_RIGHT]


class Solver:

    def __init__(self, game_map):
        self.game_map = game_map
        self.success_prob = self.game_map.t_success_prob
        self.error_prob = self.game_map.t_error_prob / 5
        self.originPlayer_x = self.game_map.player_x
        self.originPlayer_y = self.game_map.player_y
        self.move_cost = self.game_map.move_cost
        self.collision_cost = self.game_map.collision_cost
        self.game_over_cost = self.game_map.game_over_cost
        self.goal_reward = self.game_map.goal_reward

        # pi
        self.k = 10
        self.converged = False
        self.count = 0

        # mct
        self.calculation_time = 1
        self.max_move = 100
        # You may also add any instance variables (e.g. root node of MCTS tree) here.

        self.values = None
        self.policy = None

    def cell_is_game_over(self, y, x):
        """
        rewrite dont consider movable object
        """
        # check for water
        if self.game_map.grid_data[y][x] == LaserTankMap.WATER_SYMBOL:
            return True

        # no water or anti-tank danger
        return False

    def cell_is_blocked(self, y, x):
        """
        rewrite dont consider movable object
        """
        symbol = self.game_map.grid_data[y][x]
        # collision: obstacle, bridge, mirror (all types), anti-tank (all types)
        if symbol == LaserTankMap.OBSTACLE_SYMBOL:
            return True
        return False

    def apply_fix_move(self, game_map, move, next_y, next_x):
        """
        Apply a player fixed move to the map.
        :param move: self.MOVE_FORWARD, self.TURN_LEFT, self.TURN_RIGHT
        :return: the reward received for performing this action (a real number)
        """
        if move == LaserTankMap.MOVE_FORWARD:
            # handle special tile types
            if game_map.grid_data[next_y][next_x] == LaserTankMap.ICE_SYMBOL:
                if game_map.player_heading == 0:  # LaserTankMap.UP:
                    for i in range(next_y, -1, -1):
                        if game_map.grid_data[i][next_x] != LaserTankMap.ICE_SYMBOL:
                            if game_map.grid_data[i][next_x] == LaserTankMap.WATER_SYMBOL:
                                return game_map.game_over_cost
                            elif self.cell_is_blocked(i, next_x):
                                next_y = i + 1
                                break
                            else:
                                next_y = i
                                break
                elif game_map.player_heading == 1:  # LaserTankMap.DOWN:
                    for i in range(next_y, game_map.y_size):
                        if game_map.grid_data[i][next_x] != LaserTankMap.ICE_SYMBOL:
                            if game_map.grid_data[i][next_x] == LaserTankMap.WATER_SYMBOL:
                                return game_map.game_over_cost
                            elif self.cell_is_blocked(i, next_x):
                                next_y = i - 1
                                break
                            else:
                                next_y = i
                                break
                elif game_map.player_heading == 2:  # LaserTankMap.LEFT:
                    for i in range(next_x, -1, -1):
                        if game_map.grid_data[next_y][i] != LaserTankMap.ICE_SYMBOL:
                            if game_map.grid_data[next_y][i] == LaserTankMap.WATER_SYMBOL:
                                return game_map.game_over_cost
                            elif self.cell_is_blocked(next_y, i):
                                next_x = i + 1
                                break
                            else:
                                next_x = i
                                break
                else:
                    for i in range(next_x, game_map.x_size):
                        if game_map.grid_data[next_y][i] != LaserTankMap.ICE_SYMBOL:
                            if game_map.grid_data[next_y][i] == LaserTankMap.WATER_SYMBOL:
                                return game_map.game_over_cost
                            elif self.cell_is_blocked(next_y, i):
                                next_x = i - 1
                                break
                            else:
                                next_x = i
                                break
            if game_map.grid_data[next_y][next_x] == LaserTankMap.TELEPORT_SYMBOL:
                # handle teleport - find the other teleport
                tpy, tpx = (None, None)
                for i in range(game_map.y_size):
                    for j in range(game_map.x_size):
                        if game_map.grid_data[i][j] == LaserTankMap.TELEPORT_SYMBOL and i != next_y and j != next_x:
                            tpy, tpx = (i, j)
                            break
                    if tpy is not None:
                        break
                if tpy is None:
                    raise Exception("LaserTank Map Error: Unmatched teleport symbol")
                next_y, next_x = (tpy, tpx)
            else:
                # if not ice or teleport, perform collision check
                if self.cell_is_blocked(next_y, next_x):
                    return game_map.collision_cost

            # check for game over conditions
            if self.cell_is_game_over(next_y, next_x):
                return game_map.game_over_cost

            # no collision and no game over - update player position
            game_map.player_y = next_y
            game_map.player_x = next_x

        elif move == LaserTankMap.TURN_LEFT:
            if game_map.player_heading == LaserTankMap.UP:
                game_map.player_heading = LaserTankMap.LEFT
            elif game_map.player_heading == LaserTankMap.DOWN:
                game_map.player_heading = LaserTankMap.RIGHT
            elif game_map.player_heading == LaserTankMap.LEFT:
                game_map.player_heading = LaserTankMap.DOWN
            else:
                game_map.player_heading = LaserTankMap.UP

        elif move == LaserTankMap.TURN_RIGHT:
            if game_map.player_heading == LaserTankMap.UP:
                game_map.player_heading = LaserTankMap.RIGHT
            elif game_map.player_heading == LaserTankMap.DOWN:
                game_map.player_heading = LaserTankMap.LEFT
            elif game_map.player_heading == LaserTankMap.LEFT:
                game_map.player_heading = LaserTankMap.UP
            else:
                game_map.player_heading = LaserTankMap.DOWN

        if game_map.grid_data[game_map.player_y][game_map.player_x] == LaserTankMap.FLAG_SYMBOL:
            return game_map.goal_reward
        else:
            return game_map.move_cost

    def succProbReward(self, rSucc, cSucc, dirSucc, action):
        # INPUT CORRECT
        # print(cSucc, rSucc, dirSucc)
        # return a list of (newState ,prob, reward) triples
        # state = s, action = a, newState = s'
        # prob = T(s, a, s'), reward = Reward(s, a ,s')
        result = []
        if action == LaserTankMap.MOVE_FORWARD:
            if dirSucc == 0:  # self.game_map.UP:
                if rSucc - 1 < 0:
                    # three topLeft, topRight, and top
                    result.append((cSucc, rSucc, dirSucc, self.success_prob, self.game_map.collision_cost))
                    result.append((cSucc, rSucc, dirSucc, self.error_prob, self.game_map.collision_cost))
                    result.append((cSucc, rSucc, dirSucc, self.error_prob, self.game_map.collision_cost))
                else:
                    self.game_map.player_x = cSucc
                    self.game_map.player_y = rSucc
                    self.game_map.player_heading = dirSucc
                    if self.game_map.grid_data[rSucc - 1][cSucc] == LaserTankMap.WATER_SYMBOL:
                        reward = self.game_over_cost
                    else:
                        reward = self.apply_fix_move(self.game_map, action, rSucc - 1, cSucc)
                    result.append((self.game_map.player_x, self.game_map.player_y, dirSucc, self.success_prob, reward))
                    self.game_map.player_x = cSucc
                    self.game_map.player_y = rSucc
                    self.game_map.player_heading = dirSucc
                    if self.game_map.grid_data[rSucc - 1][cSucc - 1] == LaserTankMap.WATER_SYMBOL:
                        reward = self.game_over_cost
                    else:
                        reward = self.apply_fix_move(self.game_map, action, rSucc - 1, cSucc - 1)
                    result.append((self.game_map.player_x, self.game_map.player_y, dirSucc, self.error_prob, reward))
                    self.game_map.player_x = cSucc
                    self.game_map.player_y = rSucc
                    self.game_map.player_heading = dirSucc
                    if self.game_map.grid_data[rSucc - 1][cSucc + 1] == LaserTankMap.WATER_SYMBOL:
                        reward = self.game_over_cost
                    else:
                        reward = self.apply_fix_move(self.game_map, action, rSucc - 1, cSucc + 1)
                    result.append((self.game_map.player_x, self.game_map.player_y, dirSucc, self.error_prob, reward))
                self.game_map.player_x = cSucc
                self.game_map.player_y = rSucc
                self.game_map.player_heading = dirSucc
                if self.game_map.grid_data[rSucc][cSucc - 1] == LaserTankMap.WATER_SYMBOL:
                    reward = self.game_over_cost
                else:
                    reward = self.apply_fix_move(self.game_map, action, rSucc, cSucc - 1)
                result.append((self.game_map.player_x, self.game_map.player_y, dirSucc, self.error_prob, reward))
                self.game_map.player_x = cSucc
                self.game_map.player_y = rSucc
                self.game_map.player_heading = dirSucc
                if self.game_map.grid_data[rSucc][cSucc + 1] == LaserTankMap.WATER_SYMBOL:
                    reward = self.game_over_cost
                else:
                    reward = self.apply_fix_move(self.game_map, action, rSucc, cSucc + 1)
                result.append((self.game_map.player_x, self.game_map.player_y, dirSucc, self.error_prob, reward))
                self.game_map.player_x = cSucc
                self.game_map.player_y = rSucc
                self.game_map.player_heading = dirSucc
                reward = self.apply_fix_move(self.game_map, action, rSucc, cSucc)
                result.append((self.game_map.player_x, self.game_map.player_y, dirSucc, self.error_prob, reward))
            elif dirSucc == 1:  # self.game_map.DOWN:
                if rSucc + 1 > self.game_map.y_size:
                    result.append((cSucc, rSucc, dirSucc, self.success_prob, self.game_map.collision_cost))
                    result.append((cSucc, rSucc, dirSucc, self.error_prob, self.game_map.collision_cost))
                    result.append((cSucc, rSucc, dirSucc, self.error_prob, self.game_map.collision_cost))
                else:
                    self.game_map.player_x = cSucc
                    self.game_map.player_y = rSucc
                    self.game_map.player_heading = dirSucc
                    if self.game_map.grid_data[rSucc + 1][cSucc] == LaserTankMap.WATER_SYMBOL:
                        reward = self.game_over_cost
                    else:
                        reward = self.apply_fix_move(self.game_map, action, rSucc + 1, cSucc)
                    result.append((self.game_map.player_x, self.game_map.player_y, dirSucc, self.success_prob, reward))
                    self.game_map.player_x = cSucc
                    self.game_map.player_y = rSucc
                    self.game_map.player_heading = dirSucc
                    if self.game_map.grid_data[rSucc + 1][cSucc - 1] == LaserTankMap.WATER_SYMBOL:
                        reward = self.game_over_cost
                    else:
                        reward = self.apply_fix_move(self.game_map, action, rSucc + 1, cSucc - 1)
                    result.append((self.game_map.player_x, self.game_map.player_y, dirSucc, self.error_prob, reward))
                    self.game_map.player_x = cSucc
                    self.game_map.player_y = rSucc
                    self.game_map.player_heading = dirSucc
                    if self.game_map.grid_data[rSucc + 1][cSucc + 1] == LaserTankMap.WATER_SYMBOL:
                        reward = self.game_over_cost
                    else:
                        reward = self.apply_fix_move(self.game_map, action, rSucc + 1, cSucc + 1)
                    result.append((self.game_map.player_x, self.game_map.player_y, dirSucc, self.error_prob, reward))
                self.game_map.player_x = cSucc
                self.game_map.player_y = rSucc
                self.game_map.player_heading = dirSucc
                if self.game_map.grid_data[rSucc][cSucc - 1] == LaserTankMap.WATER_SYMBOL:
                    reward = self.game_over_cost
                else:
                    reward = self.apply_fix_move(self.game_map, action, rSucc, cSucc - 1)
                result.append((self.game_map.player_x, self.game_map.player_y, dirSucc, self.error_prob, reward))
                self.game_map.player_x = cSucc
                self.game_map.player_y = rSucc
                self.game_map.player_heading = dirSucc
                if self.game_map.grid_data[rSucc][cSucc + 1] == LaserTankMap.WATER_SYMBOL:
                    reward = self.game_over_cost
                else:
                    reward = self.apply_fix_move(self.game_map, action, rSucc, cSucc + 1)
                result.append((self.game_map.player_x, self.game_map.player_y, dirSucc, self.error_prob, reward))
                self.game_map.player_x = cSucc
                self.game_map.player_y = rSucc
                self.game_map.player_heading = dirSucc
                reward = self.apply_fix_move(self.game_map, action, rSucc, cSucc)
                result.append((self.game_map.player_x, self.game_map.player_y, dirSucc, self.error_prob, reward))
            elif dirSucc == 2:  # self.game_map.LEFT:
                if cSucc - 1 < 0:
                    result.append((cSucc, rSucc, dirSucc, self.success_prob, self.game_map.collision_cost))
                    result.append((cSucc, rSucc, dirSucc, self.error_prob, self.game_map.collision_cost))
                    result.append((cSucc, rSucc, dirSucc, self.error_prob, self.game_map.collision_cost))
                else:
                    self.game_map.player_x = cSucc
                    self.game_map.player_y = rSucc
                    self.game_map.player_heading = dirSucc
                    if self.game_map.grid_data[rSucc][cSucc - 1] == LaserTankMap.WATER_SYMBOL:
                        reward = self.game_over_cost
                    else:
                        reward = self.apply_fix_move(self.game_map, action, rSucc, cSucc - 1)
                    result.append((self.game_map.player_x, self.game_map.player_y, dirSucc, self.success_prob, reward))
                    self.game_map.player_x = cSucc
                    self.game_map.player_y = rSucc
                    self.game_map.player_heading = dirSucc
                    if self.game_map.grid_data[rSucc - 1][cSucc - 1] == LaserTankMap.WATER_SYMBOL:
                        reward = self.game_over_cost
                    else:
                        reward = self.apply_fix_move(self.game_map, action, rSucc - 1, cSucc - 1)
                    result.append((self.game_map.player_x, self.game_map.player_y, dirSucc, self.error_prob, reward))
                    self.game_map.player_x = cSucc
                    self.game_map.player_y = rSucc
                    self.game_map.player_heading = dirSucc
                    if self.game_map.grid_data[rSucc + 1][cSucc - 1] == LaserTankMap.WATER_SYMBOL:
                        reward = self.game_over_cost
                    else:
                        reward = self.apply_fix_move(self.game_map, action, rSucc + 1, cSucc - 1)
                    result.append((self.game_map.player_x, self.game_map.player_y, dirSucc, self.error_prob, reward))
                self.game_map.player_x = cSucc
                self.game_map.player_y = rSucc
                self.game_map.player_heading = dirSucc
                if self.game_map.grid_data[rSucc - 1][cSucc] == LaserTankMap.WATER_SYMBOL:
                    reward = self.game_over_cost
                else:
                    reward = self.apply_fix_move(self.game_map, action, rSucc - 1, cSucc)
                result.append((self.game_map.player_x, self.game_map.player_y, dirSucc, self.error_prob, reward))
                self.game_map.player_x = cSucc
                self.game_map.player_y = rSucc
                self.game_map.player_heading = dirSucc
                if self.game_map.grid_data[rSucc + 1][cSucc] == LaserTankMap.WATER_SYMBOL:
                    reward = self.game_over_cost
                else:
                    reward = self.apply_fix_move(self.game_map, action, rSucc + 1, cSucc)
                result.append((self.game_map.player_x, self.game_map.player_y, dirSucc, self.error_prob, reward))
                self.game_map.player_x = cSucc
                self.game_map.player_y = rSucc
                self.game_map.player_heading = dirSucc
                reward = self.apply_fix_move(self.game_map, action, rSucc, cSucc)
                result.append((self.game_map.player_x, self.game_map.player_y, dirSucc, self.error_prob, reward))
            elif dirSucc == 3:  # self.game_map.RIGHT:
                if cSucc + 1 >= self.game_map.x_size:
                    result.append((cSucc, rSucc, dirSucc, self.success_prob, self.game_map.collision_cost))
                    result.append((cSucc, rSucc, dirSucc, self.error_prob, self.game_map.collision_cost))
                    result.append((cSucc, rSucc, dirSucc, self.error_prob, self.game_map.collision_cost))
                else:
                    self.game_map.player_x = cSucc
                    self.game_map.player_y = rSucc
                    self.game_map.player_heading = dirSucc
                    if self.game_map.grid_data[rSucc][cSucc + 1] == LaserTankMap.WATER_SYMBOL:
                        reward = self.game_over_cost
                    else:
                        reward = self.apply_fix_move(self.game_map, action, rSucc, cSucc + 1)
                    result.append((self.game_map.player_x, self.game_map.player_y, dirSucc, self.success_prob, reward))
                    self.game_map.player_x = cSucc
                    self.game_map.player_y = rSucc
                    self.game_map.player_heading = dirSucc
                    if self.game_map.grid_data[rSucc - 1][cSucc + 1] == LaserTankMap.WATER_SYMBOL:
                        reward = self.game_over_cost
                    else:
                        reward = self.apply_fix_move(self.game_map, action, rSucc - 1, cSucc + 1)
                    result.append((self.game_map.player_x, self.game_map.player_y, dirSucc, self.error_prob, reward))
                    self.game_map.player_x = cSucc
                    self.game_map.player_y = rSucc
                    self.game_map.player_heading = dirSucc
                    if self.game_map.grid_data[rSucc + 1][cSucc + 1] == LaserTankMap.WATER_SYMBOL:
                        reward = self.game_over_cost
                    else:
                        reward = self.apply_fix_move(self.game_map, action, rSucc + 1, cSucc + 1)
                    result.append((self.game_map.player_x, self.game_map.player_y, dirSucc, self.error_prob, reward))
                self.game_map.player_x = cSucc
                self.game_map.player_y = rSucc
                self.game_map.player_heading = dirSucc
                if self.game_map.grid_data[rSucc - 1][cSucc] == LaserTankMap.WATER_SYMBOL:
                    reward = self.game_over_cost
                else:
                    reward = self.apply_fix_move(self.game_map, action, rSucc - 1, cSucc)
                result.append((self.game_map.player_x, self.game_map.player_y, dirSucc, self.error_prob, reward))
                self.game_map.player_x = cSucc
                self.game_map.player_y = rSucc
                self.game_map.player_heading = dirSucc
                if self.game_map.grid_data[rSucc + 1][cSucc] == LaserTankMap.WATER_SYMBOL:
                    reward = self.game_over_cost
                else:
                    reward = self.apply_fix_move(self.game_map, action, rSucc + 1, cSucc)
                result.append((self.game_map.player_x, self.game_map.player_y, dirSucc, self.error_prob, reward))
                self.game_map.player_x = cSucc
                self.game_map.player_y = rSucc
                self.game_map.player_heading = dirSucc
                reward = self.apply_fix_move(self.game_map, action, rSucc, cSucc)
                result.append((self.game_map.player_x, self.game_map.player_y, dirSucc, self.error_prob, reward))
        elif action == LaserTankMap.TURN_LEFT or action == LaserTankMap.TURN_RIGHT:
            self.game_map.player_x = cSucc
            self.game_map.player_y = rSucc
            self.game_map.player_heading = dirSucc
            reward = self.apply_fix_move(self.game_map, action, rSucc, cSucc)
            result.append((self.game_map.player_x, self.game_map.player_y, self.game_map.player_heading, 1, reward))
        # print(result)
        self.game_map.player_x = self.originPlayer_x
        self.game_map.player_y = self.originPlayer_y
        return result

    def Q(self, row, col, direc, action):
        # result = succProbReward(row, col, dir, action)
        # total = 0
        # # print('{:20} {:20} {:20} {:20} {:20}'.format('col', 'row', 'direction', 'newV(s)', 'reward'))
        # for r, c, d, prob, reward in result:
        #     # print('{:15} {:15} {:15} {:15} {:15}'.format(c, r, d, prob, reward))
        #     total += prob * (reward + self.game_map.gamma * values[c - 1][r - 1][d])
        # return total
        return sum(
            prob * (reward + self.game_map.gamma * self.values[co - 1][ro - 1][di]) for co, ro, di, prob, reward in
            self.succProbReward(row, col, direc, action))

    def unique_pairs(self):
        """Produce pairs of indexes in range(n)"""
        for c in range(0, self.game_map.x_size - 2):
            for r in range(0, self.game_map.y_size - 2):
                for d in LaserTankMap.DIRECTIONS:
                    yield c, r, d

    def run_value_iteration(self):
        """
        Build a value table and a policy table using value iteration, and store inside self.values and self.policy.

        When this method is called, you are allowed up to [state.time_limit] seconds of compute time. You should stop
        # iterating either when max_delta < epsilon, or when the time limit is reached, whichever occurs first.
        """
        # print("actual map size (x, y): ", self.game_map.x_size - 2, self.game_map.y_size - 2)
        # print("player postion (x, y): ", self.game_map.player_x, self.game_map.player_y)
        # print("flag position (x, y): ", self.game_map.flag_x, self.game_map.flag_y)
        start = time.time()
        # initialize state -> Vopt[state]
        values = [[[0 for _ in LaserTankMap.DIRECTIONS]
                   for __ in range(1, self.game_map.y_size - 1)]
                  for ___ in range(1, self.game_map.x_size - 1)]
        policy = [[[-1 for _ in LaserTankMap.DIRECTIONS]
                   for __ in range(1, self.game_map.y_size - 1)]
                  for ___ in range(1, self.game_map.x_size - 1)]
        self.values = values
        self.policy = policy

        count = 0
        while True:
            newV = [[[0 for _ in LaserTankMap.DIRECTIONS]
                     for __ in range(1, self.game_map.y_size - 1)]
                    for ___ in range(1, self.game_map.x_size - 1)]
            # compute the new values (newV) given the old values (V)
            for c, r, d in self.unique_pairs():
                if self.game_map.grid_data[r + 1][c + 1] == LaserTankMap.FLAG_SYMBOL:
                    newV[c][r][d] = 0
                    policy[c][r][d] = -1
                else:
                    # return result in tuple (value, action)
                    result = max((self.Q(r + 1, c + 1, d, action), action) for action in MOVES)
                    newV[c][r][d] = result[0]
                    # read out policy
                    policy[c][r][d] = result[1]

            # if time.time() > start + 1:
            #     print(time.time() - start)
            #     print("finished early")
            #     break
            # if count == 10:
            #     print(time.time() - start)
            #     print("finished early")
            #     break

            # check for convergence
            if max(abs(self.values[c][r][d] - newV[c][r][d]) for c, r, d in
                   self.unique_pairs()) < self.game_map.epsilon:
                print(time.time() - start)
                print(count)
                break
            elif time.time() - start > self.game_map.time_limit:
                break
            # store the computed values and policy
            self.values = newV
            self.policy = policy
            count += 1

            # print('{:15} {:15} {:15} {:25} {:30}'.format('col', 'row', 'direction', 'newV(s)', 'policy(s)'))
            # for c, r, d in self.unique_pairs():
            #     print('{:15} {:15} {:15} {:25} {:>30}'.format(c, r, d, self.values[c][r][d], self.policy[c][r][d]))
            # input()

    def convergence_check(self, new_policy):
        """
        Write code to check if PI has converged here
        """
        if self.policy == new_policy:  # fragile
            self.converged = True

    def next_evaluation_iteration(self):
        newV = [[[0 for _ in LaserTankMap.DIRECTIONS]
                 for __ in range(1, self.game_map.y_size - 1)]
                for ___ in range(1, self.game_map.x_size - 1)]
        # compute the new values (newV) given the old values (V)
        for c, r, d in self.unique_pairs():
            action = self.policy[c][r][d]
            if self.game_map.grid_data[r + 1][c + 1] == LaserTankMap.FLAG_SYMBOL:
                newV[c][r][d] = 0
            else:
                newV[c][r][d] = self.Q(r + 1, c + 1, d, action)
        # store the computed values and policy
        self.values = newV

    def policy_improvement(self):
        """
        Write code to extract the best policy for a given value function here
        """
        newP = [[[-1 for _ in LaserTankMap.DIRECTIONS]
                 for __ in range(1, self.game_map.y_size - 1)]
                for ___ in range(1, self.game_map.x_size - 1)]
        for c, r, d in self.unique_pairs():
            if self.game_map.grid_data[r + 1][c + 1] == LaserTankMap.FLAG_SYMBOL:
                newP[c][r][d] = -1
            else:
                newP[c][r][d] = max((self.Q(r + 1, c + 1, d, action), action) for action in MOVES)[1]
        self.convergence_check(newP)
        self.policy = newP
        self.count += 1

    def policy_evaluation(self):
        """
        Using MPI to evaluate self.policy
        """
        for kk in range(self.k):
            self.next_evaluation_iteration()

    def run_policy_iteration(self):
        """
        Build a value table and a policy table using policy iteration, and store inside self.values and self.policy.
        """
        start = time.time()
        values = [[[0 for _ in LaserTankMap.DIRECTIONS]
                   for __ in range(1, self.game_map.y_size - 1)]
                  for ___ in range(1, self.game_map.x_size - 1)]
        policy = [[[-1 for _ in LaserTankMap.DIRECTIONS]
                   for __ in range(1, self.game_map.y_size - 1)]
                  for ___ in range(1, self.game_map.x_size - 1)]
        self.values = values
        self.policy = policy

        while True:
            self.policy_evaluation()
            self.policy_improvement()
            # if time.time() > start + 1:
            #     print(time.time() - start)
            #     print("finished early")
            #     break
            # if self.count == 10:
            #     print(time.time() - start)
            #     print("finished early")
            #     break
            if self.converged:
                print(time.time() - start)
                print(self.count)
                break
            # print('{:15} {:15} {:15} {:25} {:30}'.format('col', 'row', 'direction', 'newV(s)', 'policy(s)'))
            # for c, r, d in self.unique_pairs():
            #     print('{:15} {:15} {:15} {:25} {:>30}'.format(c, r, d, self.values[c][r][d], self.policy[c][r][d]))
            # input()

    def get_offline_value(self, state):
        """
        Get the value of this state.
        :param state: a LaserTankMap instance
        :return: V(s) [a floating point number]
        """
        r = state.player_y - 1
        c = state.player_x - 1
        d = state.player_heading
        return self.values[c][r][d]

    def get_offline_policy(self, state):
        """
        Get the policy for this state (i.e. the action that should be performed at this state).
        :param state: a LaserTankMap instance
        :return: pi(s) [an element of LaserTankMap.MOVES]
        """
        r = state.player_y - 1
        c = state.player_x - 1
        d = state.player_heading
        return self.policy[c][r][d]

    def run_simulation(self):
        newMap = self.game_map.make_clone(self)
        state = [newMap.player_x][newMap.player_y][newMap.player_heading]

        for t in range(self.max_move):
            for action in LaserTankMap.MOVES:
                newMap.apply_move(action)



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


class Node:
    def __init__(self, state, parent=None):
        self.visits = 1
        self.reward = 0.0
        self.state = state
        self.children = []
        self.parent = parent

    def add_child(self, child_state):
        child = Node(child_state, self)
        self.children.append(child)

    def update(self, reward):
        self.reward += reward
        self.visits += 1

    def fully_expanded(self):
        if len(self.children) == self.state.num_moves:
            return True
        return False

    def __repr__(self):
        s = "Node; children: %d; visits: %d; reward: %f" % (len(self.children), self.visits, self.reward)
        return s
