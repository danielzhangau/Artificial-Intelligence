import time
from queue import PriorityQueue

def search_astar(start, goal, env, heuristic_mode='zeroed'):
    begin_clock = time.time()
    log = dict()
    log['nvextex_explored_(with_duplicates)'] = 0

    fringe = PriorityQueue()
    fringe.put(start)
    explored = {start.id: start.cost} # a dict of `vertex: cost_so_far`
    path = {start.id: []} # a dict of `vertex: actions`
    log['nvextex_explored_(with_duplicates)'] += 1

    while not fringe.empty():
        current = fringe.get()
        if current==goal:
            log['nvertex_in_fringe_at_termination'] = fringe.qsize()
            log['nvextex_explored'] = len(explored)
            log['action_path'] = path[current.id]
            log['solution_path_cost'] = explored[current.id]
            log['elapsed_time_in_minutes'] = (time.time() - begin_clock)/60.
            return log

        for neighbor, action in env.get_neighborlist(current):
            cost_so_far = explored[current.id] + current.cost
            if (neighbor.id not in explored) or (cost_so_far < explored[neighbor.id]):
                explored[neighbor.id] = cost_so_far
                path[neighbor.id] = path[current.id] + [action]
                log['nvextex_explored_(with_duplicates)'] += 1

                vfp = cost_so_far + env.estimate_cost_to_go(neighbor, heuristic_mode)
                neighbor.value_for_priority = vfp
                fringe.put(neighbor)

    raise RuntimeError('No solution!')
