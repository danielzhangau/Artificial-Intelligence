# faster
import math
import random
import sys
import time

from angle import Angle
from problem_spec import ProblemSpec
from robot_config import make_robot_config_from_ee1, make_robot_config_from_ee2, \
    write_robot_config_list_to_file
from tester import test_obstacle_collision, __get_lenient_obstacles, test_config_equality, test_self_collision, \
    test_environment_bounds


# 0.001 radians = 0.05729565 degrees.


class GraphNode:
    def __init__(self, spec, config):
        self.spec = spec
        self.config = config
        self.neighbors = []

    def __eq__(self, other):
        return test_config_equality(self.config, other.config, self.spec)

    def __hash__(self):
        return hash(tuple(self.config.points))

    def get_successors(self):
        return self.neighbors

    @staticmethod
    def add_connection(n1, n2):
        """
        Creates a neighbor connection between the 2 given GraphNode objects.

        :param n1: a GraphNode object
        :param n2: a GraphNode object
        """
        n1.neighbors.append(n2)
        n2.neighbors.append(n1)


def collison_checking(new_config, spec, obstacles):
    """
    return true for pass, false for fail
    1. It will not collide with any of the obstacles
    2. It will not collide with itself
    3. The entire Canadarm robotic arm must lie inside the workspace
    """
    if (test_obstacle_collision(new_config, spec, obstacles) and test_self_collision(new_config, spec) and
            test_environment_bounds(new_config)):
        return True
    return False


def distance_checking(spec, phase, config_first, config_sec):
    """
    Calculate the angle and length difference between two identical sequences.
    0.8 here as the threshold for neighbor limit
    """
    for i in range(spec.num_segments):
        if config_sec.lengths[i] - config_first.lengths[i] > 0.85:
            return False
        if phase % 2 == 0:
            if abs(config_sec.ee1_angles[i].radians - config_first.ee1_angles[i].radians) > 0.85:
                return False
        else:
            if abs(config_sec.ee2_angles[i].radians - config_first.ee2_angles[i].radians) > 0.85:
                return False

    return True


def interpolation_checking(spec, phase, config_first, config_sec, obstacles):
    """
    test whether the path between two sample points is valid, the path is discretised into 20 segments,
    and verification checks will be applied to ensure that there are no invalid configurations in the path.

    :param config_first: sample points
    :param config_sec: exiting configs

    q <- config
    list x <- distance between angles
    list y <- distance between lengths of segment
    for j := range[0, 20] do
        for n := 0 to number of segments do
            new angles[n] = q.angles[n] + x[n]*0.05*j
            new lengths[n] = q.lengths[n] + y[n]*0.05*j
        using new angles and lengths to construct new configuration c
        if c is invalid -> return false
    return true

    note: 0.05 = 1/20
    """
    angles_first = config_first.ee1_angles if phase % 2 == 0 else config_first.ee2_angles
    angles_sec = config_sec.ee1_angles if phase % 2 == 0 else config_sec.ee2_angles

    between_angle = []
    lengths_first = config_first.lengths
    lengths_sec = config_sec.lengths
    between_lengths = []

    # calculate the lists of length and angle differences
    for i in range(len(angles_first)):
        between_angle.append(angles_sec[i].__sub__(angles_first[i]))
        between_lengths.append(lengths_sec[i] - lengths_first[i])

    # discretize the each list into 20 parts
    for j in range(21):
        valid_angle_list = []
        valid_length_list = []
        for n in range(len(angles_first)):
            if phase % 2 == 0:
                valid_angle_list.append(config_first.ee1_angles[n].__add__(between_angle[n].__mul__(0.05 * j)))
            else:
                valid_angle_list.append(config_first.ee2_angles[n].__add__(between_angle[n].__mul__(0.05 * j)))

            valid_length_list.append(config_first.lengths[n] + between_lengths[n] * 0.05 * j)

        if phase % 2 == 0:
            x, y = config_first.points[0]
            new_config = make_robot_config_from_ee1(x, y, valid_angle_list, valid_length_list,
                                                    ee1_grappled=True)
        else:
            x, y = config_first.points[-1]
            new_config = make_robot_config_from_ee2(x, y, valid_angle_list, valid_length_list,
                                                    ee2_grappled=True)

        if not (collison_checking(new_config, spec, obstacles)):
            return False

    return True


def uniformly_sampling(spec, obstacles, phase):
    """
    Sample a configuration q by choosing a random arm posture. Check whether the random arm position is within the
    minimum and maximum conditions, whether it is within the obstacle, whether it collides with the obstacle or itself
    (if q -> F, then add to G), repeat this operation until N samples are created .
    Each configuration becomes a node to be added to the state diagram.
    """
    while True:
        sampling_angles = []
        sampling_lengths = []
        for i in range(spec.num_segments):
            # The angle between adjacent arm segments cannot be tighter than 15 degrees
            # (i.e. angles 2, 3, 4... must be between -165 and +165).
            sample_angle = random.uniform(-165, 165)
            sampling_angles.append(Angle(degrees=float(sample_angle)))
            # The segment lengths must be within the bounds specified in the input file
            # (i.e. within min and max lengths)
            sample_length = random.uniform(spec.min_lengths[i], spec.max_lengths[i])
            sampling_lengths.append(sample_length)

        if phase % 2 == 0:  # if grapple points is 1 or odd
            new_config = make_robot_config_from_ee1(spec.grapple_points[phase][0], spec.grapple_points[phase][1],
                                                    sampling_angles, sampling_lengths, ee1_grappled=True)
        else:
            new_config = make_robot_config_from_ee2(spec.grapple_points[phase][0], spec.grapple_points[phase][1],
                                                    sampling_angles, sampling_lengths, ee2_grappled=True)

        # Verification inspection
        if collison_checking(new_config, spec, obstacles):
            return new_config


def find_graph_path(spec, obstacles, phase, partial_init=None, bridge_configs=None):
    """
    This method performs a breadth first search of the state graph and return a list of configs which form a path
    through the state graph between the initial and the goal. Note that this path will not satisfy the primitive step
    requirement - you will need to interpolate between the configs in the returned list.

    :param spec: ProblemSpec object
    :param obstacles: Obstacles object
    :return: List of configs forming a path through the graph from initial to goal

    for i := range[1, N] do
        configs = uniformly sampling();
        for config in all configs do
            if distance between c1 and c2 is below certain amount and path is valid do
                add a neighbor relationship between nodes;
    search graph
    if goal reached:
        break
    """

    # set init and goal nodes, put into nodes list
    init_node = GraphNode(spec, spec.initial) if partial_init is None else GraphNode(spec, partial_init)

    nodes = [init_node]

    if bridge_configs is None:
        goal_node = GraphNode(spec, spec.goal)
        nodes.append(goal_node)
    else:
        for partial_goal in bridge_configs:
            nodes.append(GraphNode(spec, partial_goal))

    # sample configs
    sample_number = 1000 if spec.num_segments <= 3 else 2000
    for i in range(sample_number):
        sample_point = uniformly_sampling(spec, obstacles, phase)
        new_node = GraphNode(spec, sample_point)
        nodes.append(new_node)

        for node in nodes:
            if distance_checking(spec, phase, sample_point, node.config) \
                    and interpolation_checking(spec, phase, sample_point, node.config, obstacles):
                GraphNode.add_connection(node, new_node)

    # search the graph
    init_container = [init_node]

    # here, each key is a graph node, each value is the list of configs visited on the path to the graph node
    init_visited = {init_node: [init_node.config]}

    while len(init_container) > 0:
        current = init_container.pop(0)

        complete_find = False
        if bridge_configs is None:
            if test_config_equality(current.config, spec.goal, spec):
                complete_find = True
        else:
            for partial_goal in bridge_configs:
                if test_config_equality(current.config, partial_goal, spec):
                    complete_find = True

        if complete_find is True:
            # found path to goal
            config_list = init_visited[current]
            path_list = []
            for n in range(len(config_list) - 1):
                angles_first = config_list[n].ee1_angles if phase % 2 == 0 else config_list[n].ee2_angles
                angles_sec = config_list[n + 1].ee1_angles if phase % 2 == 0 else config_list[n + 1].ee2_angles

                between_angle = []
                between_lengths = []
                lengths_first = config_list[n].lengths
                lengths_sec = config_list[n + 1].lengths

                for i in range(len(angles_first)):
                    between_angle.append(angles_sec[i].__sub__(angles_first[i]))
                    between_lengths.append(lengths_sec[i] - lengths_first[i])

                # interpolated the path into steps less than 0,001
                for j in range(851):
                    valid_angle_list = []
                    valid_length_list = []
                    for k in range(len(angles_first)):
                        if phase % 2 == 0:
                            valid_angle_list.append(config_list[n].ee1_angles[k].__add__
                                                  (between_angle[k].__mul__(1/850 * j)))
                        else:
                            valid_angle_list.append(config_list[n].ee2_angles[k].__add__
                                                  (between_angle[k].__mul__(1/850 * j)))

                        valid_length_list.append(config_list[n].lengths[k] + between_lengths[k] * 1/850 * j)

                    if phase % 2 == 0:
                        x, y = config_list[n].points[0]
                        new_config = make_robot_config_from_ee1(x, y, valid_angle_list, valid_length_list,
                                                                ee1_grappled=True)
                        path_list.append(new_config)
                    else:
                        x, y = config_list[n].points[-1]
                        new_config = make_robot_config_from_ee2(x, y, valid_angle_list, valid_length_list,
                                                                ee2_grappled=True)
                        path_list.append(new_config)

            return path_list

        successors = current.get_successors()
        for suc in successors:
            if suc not in init_visited:
                init_container.append(suc)
                init_visited[suc] = init_visited[current] + [suc.config]


def generate_bridge_configs(spec, obstacles, phase):
    """
    start point (x1, y1) is the current grapple point
    goal point (x2, y2) is the next grapple point
    create an empty sub goal list to keep the goal we find
    for i := range[1, 10] do
        for j := range[1, num of segments - 1] do
            sample some lengths and angles
        if grapple point is even order do
            partial config = generate configuration (start point, length, and angle)
            if the distance between ee2 and goal point is within the length limit do
                calculate the angle and length of the last segment
                use the parameters to generate a completed configuration
            if the configuration is collision free do
                add the configuration into bridge_configs list
        else do the former steps but with ee2 grappled at the grapple point
    """
    start_point = spec.grapple_points[phase]
    end_point = spec.grapple_points[phase + 1]

    bridge_configs = []
    for i in range(10):
        complete_find = False
        # repeat until 10 bridges generated
        while complete_find is False:
            sampling_angles = []
            sampling_lengths = []
            # finding a set of angles and lengths which form a bridge is to generate the first n-1 links
            for j in range(spec.num_segments - 1):
                sample_angle = random.uniform(-165, 165)
                sampling_angles.append(Angle(degrees=float(sample_angle)))
                sample_length = random.uniform(spec.min_lengths[j], spec.max_lengths[j])
                sampling_lengths.append(sample_length)

            if phase % 2 == 0:
                x1, y1 = start_point
                x2, y2 = end_point
                partial_bridge = make_robot_config_from_ee1(x1, y1, sampling_angles,
                                                            sampling_lengths, ee1_grappled=True)
                if (partial_bridge.points[-1][0] - x2) ** 2 + (partial_bridge.points[-1][1] - y2) ** 2 < \
                        spec.max_lengths[spec.num_segments - 1] ** 2:
                    # using tan(delta_y / delta_x)
                    calculated_net_angle = math.atan((partial_bridge.points[-1][1] - y2) /
                                                   (partial_bridge.points[-1][0] - x2))
                    # length sqrt(delta_x^2 + delta_y^2)
                    sampling_lengths.append(math.sqrt((partial_bridge.points[-1][0] - x2) ** 2 +
                                                      (partial_bridge.points[-1][1] - y2) ** 2))

                    if y2 > partial_bridge.points[-1][1] and x2 < partial_bridge.points[-1][0]:
                        calculated_net_angle += math.pi
                    elif y2 < partial_bridge.points[-1][1] and x2 < partial_bridge.points[-1][0]:
                        calculated_net_angle -= math.pi

                    partial_net_angle = Angle(radians=0)
                    for n in range(len(sampling_angles)):
                        partial_net_angle += sampling_angles[n]
                    sampling_angles.append(Angle(radians=float(calculated_net_angle - partial_net_angle.in_radians())))

                    bridge_config = make_robot_config_from_ee1(x1, y1, sampling_angles, sampling_lengths,
                                                               ee1_grappled=True, ee2_grappled=True)
                    if collison_checking(bridge_config, spec, obstacles):
                        bridge_configs.append(bridge_config)
                        complete_find = True

            else:
                x1, y1 = start_point
                x2, y2 = end_point
                partial_bridge = make_robot_config_from_ee2(x1, y1, sampling_angles, sampling_lengths,
                                                            ee2_grappled=True)

                if (partial_bridge.points[0][0] - x2) ** 2 + (partial_bridge.points[0][1] - y2) ** 2 < \
                        spec.max_lengths[0] ** 2:
                    calculated_net_angle = math.atan((partial_bridge.points[0][1] - y2) /
                                                   (partial_bridge.points[0][0] - x2))
                    sampling_lengths.insert(0, math.sqrt((partial_bridge.points[0][0] - x2) ** 2 +
                                                         (partial_bridge.points[0][1] - y2) ** 2))

                    if y2 > partial_bridge.points[0][1] and x2 < partial_bridge.points[0][0]:
                        calculated_net_angle += math.pi
                    elif y2 < partial_bridge.points[0][1] and x2 < partial_bridge.points[0][0]:
                        calculated_net_angle -= math.pi

                    partial_net_angle = Angle(radians=0)
                    for n in range(len(sampling_angles)):
                        partial_net_angle += sampling_angles[n]
                    sampling_angles.append(Angle(radians=float(calculated_net_angle - partial_net_angle.in_radians())))

                    bridge_config = make_robot_config_from_ee2(x1, y1, sampling_angles, sampling_lengths,
                                                               ee1_grappled=True, ee2_grappled=True)

                    if collison_checking(bridge_config, spec, obstacles):
                        bridge_configs.append(bridge_config)
                        complete_find = True

    return bridge_configs


def main(arglist):
    input_file = arglist[0]
    output_file = arglist[1]

    spec = ProblemSpec(input_file)

    obstacles = __get_lenient_obstacles(spec)

    steps = []

    # single grapple point no need bridge config
    if spec.num_grapple_points == 1:
        steps += find_graph_path(spec, obstacles, 0)
    else:  # for grapple points at least two
        for phase in range(spec.num_grapple_points):
            if phase == 0:
                bridge_configs = generate_bridge_configs(spec, obstacles, phase)
                partial_steps_1 = find_graph_path(spec, obstacles, phase, bridge_configs=bridge_configs)
                if partial_steps_1 is None:
                    print("no bridge_configs in stage1")
                else:
                    steps += partial_steps_1
            elif 0 < phase < spec.num_grapple_points - 1:
                bridge_configs = generate_bridge_configs(spec, obstacles, phase)
                partial_steps_2 = find_graph_path(spec, obstacles, phase, partial_init=steps[-1],
                                                  bridge_configs=bridge_configs)
                if partial_steps_2 is None:
                    print("no bridge_configs in stage2")
                else:
                    steps += partial_steps_2
            else:
                partial_steps_3 = find_graph_path(spec, obstacles, phase, partial_init=steps[-1])
                if partial_steps_3 is None:
                    print("no bridge_configs in stage3")
                else:
                    steps += partial_steps_3

    if len(arglist) > 1:
        print("steps:", len(steps))
        write_robot_config_list_to_file(output_file, steps)

    #
    # You may uncomment this line to launch visualiser once a solution has been found. This may be useful for debugging.
    # *** Make sure this line is commented out when you submit to Gradescope ***
    #
    # v = Visualiser(spec, steps)


if __name__ == '__main__':
    start_time = time.time()
    main(sys.argv[1:])
    print("finished: ", time.time() - start_time)
