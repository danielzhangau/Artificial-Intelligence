import sys
import math
from laser_tank import LaserTankMap
from solver import Solver
import matplotlib.pyplot as plt

DEBUG_MODE = True      # set to True to disable time limit checks
TOLERANCE = 0.01

CRASH = 255
OVERTIME = 254

ACTION_LOOKUP = {LaserTankMap.MOVE_FORWARD: 0,
                 LaserTankMap.TURN_LEFT: 1,
                 LaserTankMap.TURN_RIGHT: 2,
                 LaserTankMap.SHOOT_LASER: 3}

def main(arglist):
    """
    Test whether the given output file is a valid solution to the given map file.

    This test script uses a 'trapdoor function' approach to comparing your computed values and policy to a reference
    solution without revealing the reference solution - 3 different results are computed based on your values and policy
    and compared to the results computed for the reference solution.

    :param arglist: [map file name]
    """
    input_file = arglist[0]
    input_file_1 = arglist[1]
    game_map = LaserTankMap.process_input_file(input_file)
    game_map_1 = LaserTankMap.process_input_file(input_file_1)
    simulator = game_map.make_clone()
    simulator_1 = game_map_1.make_clone()
    solver = Solver(0.01)
    solver_1 = Solver(0.01)

    if game_map.method == 'q-learning':
        solver.train_q_learning(simulator)
    total_reward = 0
    num_trials = 50
    max_steps = 60
    for _ in range(num_trials):
        state = game_map.make_clone()
        for i in range(max_steps):
            action = solver.get_policy(state)
            r, f = state.apply_move(action)
            total_reward += r
            if f:
                break
    total_reward /= num_trials

    # compute score based on how close episode reward is to optimum
    print(f"Avg Episode Reward = {str(total_reward)}, Benchmark = {str(game_map.benchmark)}")
    diff = game_map.benchmark - total_reward  # amount by which benchmark score is better
    if diff < 0:
        diff = 0
    if diff > 20:
        diff = 20
    below = math.ceil(diff / 2)
    mark = 10 - below

    if below == 0:
        print("Testcase passed, policy matches or exceeds benchmark")
    elif mark > 0:
        print(f"Testcase passed, {below} marks below solution quality benchmark")
    Aveg_0 = solver.get_list()


    if game_map_1.method == 'sarsa':
        solver_1.train_sarsa(simulator_1)
    total_reward = 0
    num_trials = 50
    max_steps = 60
    for _ in range(num_trials):
        state = game_map_1.make_clone()
        for i in range(max_steps):
            action = solver_1.get_policy(state)
            r, f = state.apply_move(action)
            total_reward += r
            if f:
                break
    total_reward /= num_trials

    # compute score based on how close episode reward is to optimum
    print(f"Avg Episode Reward = {str(total_reward)}, Benchmark = {str(game_map_1.benchmark)}")
    diff = game_map_1.benchmark - total_reward  # amount by which benchmark score is better
    if diff < 0:
        diff = 0
    if diff > 20:
        diff = 20
    below = math.ceil(diff / 2)
    mark = 10 - below

    if below == 0:
        print("Testcase passed, policy matches or exceeds benchmark")
    elif mark > 0:
        print(f"Testcase passed, {below} marks below solution quality benchmark")
    Aveg_1 = solver_1.get_list()


    x = range(len(Aveg_0))
    x_1 = range(len(Aveg_1))

    plt.plot(x, Aveg_0, '--r', label='q_learning')
    plt.plot(x_1, Aveg_1, '-b', label='sarsa')
    plt.xlabel('eqisode')
    plt.ylabel('Average Reward')
    plt.title('learned policy against iteration number under \n Q-learning and SARSA with lr rate 0.01')
    plt.legend()
    plt.savefig('q4.png')
    plt.show()


if __name__ == '__main__':
    main(sys.argv[1:])