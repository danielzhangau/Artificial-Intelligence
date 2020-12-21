# Iterative Deepening Search (IDS) or Iterative Deepening Depth First Search (IDDFS)
import time
from depth_first_search import search_dfs

def search_ids(start, goal, env):
    begin_clock = time.time()
    log = dict()
    log['nvextex_explored_(with_duplicates)'] = 0

    max_depth = 1000
    log['max_depth'] = max_depth

    for max_depth_i in range(1, max_depth + 1):
        log_i = search_dfs(start, goal, env, max_depth=max_depth_i)
        if log_i is not None:
            log = {**log, **log_i} # copy the latest successful log
            log['max_depth_i_reached'] = max_depth_i
            log['elapsed_time_in_minutes'] = (time.time() - begin_clock)/60.
            return log

    raise RuntimeError()
