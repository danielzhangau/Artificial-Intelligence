import time
import queue as queuelib

def search_bfs(start, goal, env):
    begin_clock = time.time()
    log = dict()
    log['nvextex_explored_(with_duplicates)'] = 0

    explored = set() # need hashable objects!
    fringe = queuelib.Queue()
    fringe.put((start, []))

    while fringe.qsize() > 0:
        current, path = fringe.get()
        explored.add(current.id)
        log['nvextex_explored_(with_duplicates)'] += 1

        for neighbor, action in env.get_neighborlist(current):
            if neighbor==goal:
                log['nvertex_in_fringe_at_termination'] = fringe.qsize()
                log['nvextex_explored'] = len(explored)
                log['action_path'] = path + [action]
                log['elapsed_time_in_minutes'] = (time.time() - begin_clock)/60.
                return log
            if neighbor.id in explored:
                continue
            fringe.put((neighbor, path + [action]))

    raise RuntimeError('No solution!')
