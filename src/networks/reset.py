def clear_memory(self):
    """Clears the memory of all cells to allow for changes in cell
       parameters"""
    for macro in self.macro_cells:
        macro['attached_users'] = []
        macro['small_interactions'] = []
        macro['potential_users'] = []
        macro['sum_log_R'] = 0
        macro['SINR_frame'] = [[] for _ in range(40)]
        macro['sum_SINR'] = [0 for _ in range(40)]
        macro['rank_sum'] = 0
        macro['min_rank'] = 9999999999999
        macro['max_rank'] = -9999999999999
        macro['max_SINR'] = [-9999999999999 for _ in range(40)]
        macro['min_SINR'] = [9999999999999 for _ in range(40)]
        macro['percentage_rank_sum'] = 0
        macro['ABS_MSUEs'] = 0
    for i, small in enumerate(self.small_cells):
        small['attached_users'] = []
        small['potential_users'] = []
        small['macro_interactions'] = []
        small['sum_log_R'] = 0
        small['SINR_frame'] = [[] for _ in range(40)]
        small['sum_SINR'] = [0 for _ in range(40)]
        small['rank_sum'] = 0
        small['min_rank'] = 9999999999999
        small['max_rank'] = -9999999999999
        small['max_SINR'] = [-9999999999999 for _ in range(40)]
        small['min_SINR'] = [9999999999999 for _ in range(40)]
        small['percentage_rank_sum'] = 0
    self.helpless_UEs = []
    self.MC_UES = []
    self.SC_UES = []
    
    return self
