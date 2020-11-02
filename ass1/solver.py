import queue
import sys
from time import time

from laser_tank import LaserTankMap

"""
Template file for you to implement your solution to Assignment 1.

COMP3702 2020 Assignment 1 Support Code
"""


class State:

    def __init__(self, game_map, cost, parents):
        self.game_map = game_map
        self.cost = cost
        self.parents = parents
        self.moves = LaserTankMap.MOVES
        self.value_for_priority = 0
        self.id = self.__hash__()

    def get_successor(self):
        next_states = []
        for move in self.moves:
            new_data = [row[:] for row in self.game_map.grid_data]
            new_map = LaserTankMap(self.game_map.x_size, self.game_map.y_size, new_data,
                                   player_x=self.game_map.player_x, player_y=self.game_map.player_y,
                                   player_heading=self.game_map.player_heading)
            # new_state = deepcopy(self.get_map())
            new_parents = [row[:] for row in self.parents]
            # new_parents = deepcopy(self.parents)
            if new_map.apply_move(move) == LaserTankMap.SUCCESS:
                new_parents.append(move)
                nextState = State(new_map, 1, new_parents)
                next_states.append((nextState, move))

        return next_states

    def is_goal(self, goal):
        return (self.game_map.player_y, self.game_map.player_x) == goal

    def estimate_cost_to_go(self, goal):
        return 1*(abs(goal[0] - self.game_map.player_y) + abs(goal[1] - self.game_map.player_x))

    def man_distance(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def estimate_cost_to_go_teleport(self, goal):
        teleports = []
        cost = 0
        for y in range(self.game_map.y_size):
            for x in range(self.game_map.x_size):
                if self.game_map.grid_data[y][x] == LaserTankMap.TELEPORT_SYMBOL:
                    teleports.append((y, x))

        manhattan = abs(goal[0] - self.game_map.player_y) + abs(goal[1] - self.game_map.player_x)

        while teleports:
            min_distance = sys.maxsize
            min_teleport = teleports[0]
            for teleport in teleports:
                distance = self.man_distance(teleport, (self.game_map.player_y, self.game_map.player_x))
                if distance < min_distance:
                    min_distance = distance
                    min_teleport = teleport
            teleports.remove(min_teleport)
            cost += min_distance
            break

        teleportToFlag = self.man_distance(teleports[0], goal)
        cost += teleportToFlag

        if manhattan < cost:
            return manhattan

        return cost

    def estimate_cost_to_go_ice(self, goal):
        ices = []
        cost = 0
        for y in range(self.game_map.y_size):
            for x in range(self.game_map.x_size):
                if self.game_map.grid_data[y][x] == LaserTankMap.ICE_SYMBOL:
                    ices.append((y, x))

        manhattan = abs(goal[0] - self.game_map.player_y) + abs(goal[1] - self.game_map.player_x)

        while ices:
            min_distance = sys.maxsize
            min_ice = ices[0]
            for ice in ices:
                distance = self.man_distance(ice, (self.game_map.player_y, self.game_map.player_x))
                if distance < min_distance:
                    min_distance = distance
                    min_ice = ice
            ices.remove(min_ice)
            cost += min_distance
            break

        iceToFlag = self.man_distance(ices[0], goal)
        cost += iceToFlag

        if manhattan < cost:
            return manhattan

        return cost

    def __hash__(self):
        return hash((self.game_map.player_x, self.game_map.player_y, self.game_map.player_heading)
                    + tuple([item for sublist in self.game_map.grid_data for item in sublist]))

    def __eq__(self, other):
        return self.id == other.id

    def __lt__(self, other):
        return self.value_for_priority < other.value_for_priority


def transition(start, search_type, goal):
    start_time = time()
    priorQ = queue.PriorityQueue()
    priorQ.put(start)
    # visited_states = set()
    visited_states = {start.id: start.cost}
    nodes = 1
    fringe = 1

    while 1:
        node = priorQ.get()
        # print(node.parents)
        fringe += 1
        if node.is_goal(goal):
            moves = node.parents
            return moves, nodes, priorQ.qsize(), len(visited_states), time() - start_time
        for states, action in node.get_successor():
            cost_so_far = visited_states[node.id] + node.cost
            # print("start:", node.game_map.player_y, node.game_map.player_x, "end:", s.game_map.player_y, s.game_map.player_x)
            nodes += 1
            if states.is_goal(goal):
                moves = states.parents
                return moves, nodes, priorQ.qsize(), len(visited_states), time() - start_time
            if (states.id not in visited_states) or (cost_so_far < visited_states[states.id]):
                visited_states[states.id] = cost_so_far
                if search_type == "ucs":
                    vfp = cost_so_far
                elif search_type == "a*":
                    vfp = cost_so_far + node.estimate_cost_to_go(goal)
                elif search_type == "a*-teleport":
                    if (states.game_map.player_y, node.game_map.player_x) == LaserTankMap.TELEPORT_SYMBOL:
                        search_type = "a*"
                    vfp = cost_so_far + node.estimate_cost_to_go_teleport(goal)
                elif search_type == "a*-ice":
                    vfp = cost_so_far + node.estimate_cost_to_go_ice(goal)
                states.value_for_priority = vfp
                priorQ.put(states)


def write_output_file(filename, actions):
    """
    Write a list of actions to an output file. You should use this method to write your output file.
    :param filename: name of output file
    :param actions: list of actions where is action is in LaserTankMap.MOVES
    """
    f = open(filename, 'w')
    for i in range(len(actions)):
        f.write(str(actions[i]))
        if i < len(actions) - 1:
            f.write(',')
    f.write('\n')
    f.close()


def main(arglist):
    input_file = arglist[0]
    output_file = arglist[1]

    # Read the input testcase file
    game_map = LaserTankMap.process_input_file(input_file)

    actions = []

    coord = (game_map.player_y, game_map.player_x)
    for y in range(game_map.y_size):
        for x in range(game_map.x_size):
            if game_map.grid_data[y][x] == game_map.FLAG_SYMBOL:
                goal_coord = (y, x)
    # print(coord)
    # print(goal_coord)

    game_map.coord = coord

    state = State(game_map, 0, [])

    # UCS
    # result = transition(state, "ucs", goal_coord)
    # A*
    result = transition(state, "a*", goal_coord)
    # A* with heuristic of teleport
    # result = transition(state, "a*-teleport", goal_coord)
    # A* with heuristic of ice
    # result = transition(state, "a*-ice", goal_coord)

    print("Nodes Generated:", result[1])
    print("Nodes on Fringe:", result[2])
    print("Explored Nodes:", result[3])
    print("Time Taken:", result[4], "seconds")

    output_string = ','.join(result[0]) # moves
    # print(outputString)
    actions.append(output_string)

    # Write the solution to the output file
    write_output_file(output_file, actions)


# t3_the_river.txt
if __name__ == '__main__':
    main(sys.argv[1:])
