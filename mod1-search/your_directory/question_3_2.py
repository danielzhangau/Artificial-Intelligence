#!/usr/bin/env python3
from gridworld import GridWorld
from breath_first_search import search_bfs
from astar_search import search_astar
search_ucs = search_astar

def main():
    cost_map = []
    cost_map.append([1, 1,  1,  5,  5,  5,  5, 1, None])
    cost_map.append([1, 1,  1,  5,  5,  5,  5, 1, 1])
    cost_map.append([1, 1, 10, 10, 10, 10, 10, 1, 1])
    cost_map.append([1, 1,  1, 10, 10, 10, 10, 1, 1])
    cost_map.append([1, 1,  1,  1,  1, 10, 10, 1, 1])
    cost_map.append([1, 1,  1,  1,  1, 10, 10, 1, 1])
    cost_map.append([1, 1,  1,  1, 10, 10, 10, 1, 1])
    cost_map.append([1, 1,  1, 10, 10, 10, 10, 1, 1])
    cost_map.append([0, 1,  1,  1,  1,  1,  1, 1, 1])
    env_config = {'nrow': 9, 'ncol': 9, 'obstacle_coords': [],
        'start_coord': (8, 0), 'goal_coord': (0, 8), 'cost_map': cost_map}
    env = GridWorld(env_config)
    start = env.start_state
    goal = env.goal_state

    question_3_2_a(start, goal, env)
    question_3_2_b(start, goal, env)
    question_3_2_c(start, goal, env)
    question_3_2_d_and_e(start, goal, env)
    question_3_2_f(start, goal, env)

def question_3_2_a(start, goal, env, printing=True):
    if printing:
        print('a) Run BFS for this problem, reusing your answer from Question 1')
        print('(nb. it should not use the costs on the grid).')
        print()

    log = search_bfs(start, goal, env)
    for k, v in log.items():
        if printing: print(k, v)
    return log

def question_3_2_b(start, goal, env, printing=True):
    if printing:
        print('b) Implement UCS for this problem using a priority queue.')
        print()

    log = search_ucs(start, goal, env)
    for k, v in log.items():
        if printing: print(k, v)
    return log

def question_3_2_c(start, goal, env):
    print('c) Compare the performance of BFS and UCS in terms of')
    print('(i) the number of nodes generated,')
    print('(ii) the number of nodes on the fringe when the search terminates')
    print('(iii) the cost of the solution path.')
    print('Discuss how and why these results differ from those for Question 1.')
    print()

    log = dict()
    log['bfs'] = question_3_2_a(start, goal, env, printing=False)
    log['ucs'] = question_3_2_b(start, goal, env, printing=False)

    log['bfs']['solution_path_cost'] = start.cost
    state = start
    for action in log['bfs']['action_path']:
        next_state = env.step(state, action)
        if next_state==goal:
            break
        log['bfs']['solution_path_cost'] += next_state.cost
        state = next_state

    keys = ['nvextex_explored_(with_duplicates)', 'nvertex_in_fringe_at_termination']
    keys.append('solution_path_cost')
    algos = ['bfs', 'ucs']
    for key in keys:
        msg = [key]
        for algo in algos:
            msg.append( '[{}: {}]'.format(algo, str(log[algo][key])) )
        print('...'.join(msg))

def question_3_2_d_and_e(start, goal, env, printing=True):
    if printing:
        print('d) Now derive an admissible heuristic for this path planning problem.')
        print('e) Using your heuristic, implement A* search for solving this path planning problem.')
        print()

    heuristic_mode = 'manhattan'
    print('heuristic_mode', heuristic_mode)

    log = search_astar(start, goal, env, heuristic_mode)
    for k, v in log.items():
        if printing: print(k, v)
    return log

def question_3_2_f(start, goal, env):
    print('f) Compare the performance of UCS and A* search in terms of')
    print('(i) the number of nodes generated, ')
    print('(ii) the number of nodes on the fringe when the search terminates')
    print('(iii) the cost of the solution path.')
    print('Explain these results.')
    print()

    log = dict()
    log['ucs'] = question_3_2_b(start, goal, env, printing=False)
    log['astar'] = question_3_2_d_and_e(start, goal, env, printing=False)

    keys = ['nvextex_explored_(with_duplicates)', 'nvertex_in_fringe_at_termination']
    keys.append('solution_path_cost')
    algos = ['ucs', 'astar']
    for key in keys:
        msg = [key]
        for algo in algos:
            msg.append( '[{}: {}]'.format(algo, str(log[algo][key])) )
        print('...'.join(msg))

if __name__ == '__main__':
    main()
