import time
import queue as queuelib

def search_dfs(start, goal, env, max_depth=None):
    begin_clock = time.time()
    log = dict()
    log['nvextex_explored_(with_duplicates)'] = 0

    explored = set()
    fringe = queuelib.LifoQueue()
    fringe.put((start, []))

    while fringe.qsize() > 0:
        current, path = fringe.get()
        if current==goal:
            log['nvertex_in_fringe_at_termination'] = fringe.qsize()
            log['nvextex_explored'] = len(explored)
            log['action_path'] = path
            log['elapsed_time_in_minutes'] = (time.time() - begin_clock)/60.
            return log
        if current.id in explored:
            continue
        if (max_depth is not None) and (len(path)==max_depth):
            continue

        explored.add(current.id)
        log['nvextex_explored_(with_duplicates)'] += 1

        for neighbor, action in env.get_neighborlist(current):
            if neighbor.id not in explored:
                fringe.put((neighbor, path + [action]))
    return None
