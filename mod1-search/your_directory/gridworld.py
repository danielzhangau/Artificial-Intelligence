class GridWorld():
    def __init__(self, cfg):
        self.cfg = cfg
        self.nrow = nrow = cfg['nrow']; self.ncol = ncol = cfg['ncol']
        self.actionset = ['L', 'R', 'U', 'D']
        self.obstacle_coords = cfg['obstacle_coords']
        self.start_state = start_state = self._get_startstate()
        self.goal_state = goal_state= self._get_goalstate()

        self.cost_map = cfg['cost_map']
        if self.cost_map is None:
            self.cost_map = [[None]*ncol]*nrow
        else:
            assert len(self.cost_map)==self.nrow
            assert all([len(i)==self.ncol for i in self.cost_map])
            assert self.cost_map[start_state.coord[0]][start_state.coord[1]]==start_state.cost
            assert self.cost_map[goal_state.coord[0]][goal_state.coord[1] ]==goal_state.cost

    def get_neighborlist(self, state):
        neighborlist = []
        for action in self.actionset:
            neighbor = self.step(state, action)
            neighborlist.append((neighbor, action))
        return neighborlist

    def step(self, state, action):
        row, col = state.coord
        next_row, next_col = row, col

        if action == 'L': next_col = max(col-1, 0)
        elif action == 'R': next_col = min(col+1, self.ncol-1)
        elif action == 'U': next_row = max(row-1, 0)
        elif action == 'D': next_row = min(row+1, self.nrow-1)
        else: raise NotImplementedError(action)

        next_coord = (next_row, next_col)
        if next_coord in self.obstacle_coords:
            next_coord = state.coord
        next_cost = self.cost_map[next_row][next_col]

        next_state = GridWorldState(next_coord, next_cost)
        return next_state

    def estimate_cost_to_go(self, state, heuristic_mode):
        if heuristic_mode=='zeroed':
            cost_to_go_estimate = 0
        elif heuristic_mode=='manhattan':
            cost_to_go_estimate = abs(self.goal_state.coord[0] - state.coord[0])
            cost_to_go_estimate += abs(self.goal_state.coord[1] - state.coord[1])
        else:
            raise NotImplementedError(heuristic_mode)
        return cost_to_go_estimate

    def _get_startstate(self):
        row, col = self.cfg['start_coord']
        cost = 0
        return GridWorldState((row, col), cost)

    def _get_goalstate(self):
        row, col = self.cfg['goal_coord']
        cost = None
        return GridWorldState((row, col), cost)

class GridWorldState():
    def __init__(self, coord, cost):
        assert all([i>=0 for i in coord])
        self.id = self.coord = coord # coordinate is in (ith row, jth col)
        self.cost = cost # a cost of arriving at this state
        self.value_for_priority = 0

    def __eq__(self, other):
        return self.id==other.id

    def __lt__(self, other):
        # Used by PriorityQueue()
        # Assume: the lower the value, the higher the priority
        return self.value_for_priority < other.value_for_priority
