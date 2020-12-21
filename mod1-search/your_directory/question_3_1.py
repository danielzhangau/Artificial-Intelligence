#!/usr/bin/env python3
from gridworld import GridWorld
from breath_first_search import search_bfs
from astar_search import search_astar
from iterative_deepening_search import search_ids
search_ucs = search_astar

def main():
    question_3_1_a()
    question_3_1_b()
    question_3_1_c()
    question_3_1_d()

def question_3_1_a(printing=True):
    if printing:
        print('a) Develop a state graph representation for this search problem, and')
        print('develop a step() method for finding the next legal steps this problem,')
        print('i.e. for generating successor nodes (vertices).')
        print()

    obstacle_coords = [(2, 2), (2, 3), (2, 4), (2, 5), (2, 6),
        (3, 6), (4, 6), (5, 6), (6, 6), (7, 6), (7, 3), (7, 4), (7, 5)]
    env_config = {'nrow': 9, 'ncol': 9, 'obstacle_coords': obstacle_coords,
        'start_coord': (8, 0), 'goal_coord': (0, 8), 'cost_map': None}
    env = GridWorld(env_config)
    if printing: print([i for i in dir(env) if '__' not in i], '\n')
    for k, v in  vars(env).items():
        if k=='cfg': continue
        if printing: print(k, v, '\n')
    return env

def question_3_1_b(printing=True):
    print('b) Implement BFS for this problem (using a FIFO queue) using your step() function.')
    env = question_3_1_a(printing=False)
    start, goal = env.start_state, env.goal_state

    log = search_bfs(start, goal, env)
    for k, v in log.items():
        if printing: print(k, v)
    return log

def question_3_1_c(printing=True):
    print('c) Implement iterative-deepening DFS (IDDFS) for this problem ')
    print('using a length-limited LIFO queue, and reusing step().')
    print()

    env = question_3_1_a(printing=False)
    start, goal = env.start_state, env.goal_state

    log = search_ids(start, goal, env)
    for k, v in log.items():
        if printing: print(k, v)
    return log

def question_3_1_d():
    print('Compare the performance of BFS and IDDFS in terms of')
    print('(i) the number of nodes generated,')
    print('(ii) the number of nodes on the fringe when the search terminates')
    print('(iii) the number of nodes on the explored set (if there is one) when')
    print('the search terminates, and')
    print('(iv) the run time of the algorithm (e.g. in units such as mins:secs).')
    print('Discuss your findings.')
    print()

    log = dict()
    log['bfs'] = question_3_1_b(printing=False)
    log['iddfs'] = question_3_1_c(printing=False)

    keys = []
    keys.append('nvextex_explored_(with_duplicates)')
    keys.append('nvertex_in_fringe_at_termination')
    keys.append('nvextex_explored')
    keys.append('elapsed_time_in_minutes')
    algos = ['bfs', 'iddfs']
    for key in keys:
        msg = [key]
        for algo in algos:
            msg.append( '[{}: {}]'.format(algo, str(log[algo][key])) )
        print('...'.join(msg))

if __name__ == '__main__':
    main()
