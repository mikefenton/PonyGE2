from random import uniform


def random_power_bias(self):
    """ Random powers and biases for all SCs.
    """
    
    for i, small in enumerate(self.small_cells):
        small['power'] = uniform(self.power_limits[0], self.power_limits[1])
        small['bias'] = uniform(self.CSB_limits[0], self.CSB_limits[1])
    
    return self


def max_power_bias(self):
    """ Random powers and biases for all SCs.
    """
    
    for i, small in enumerate(self.small_cells):
        small['power'] = self.power_limits[1]
        small['bias'] = self.CSB_limits[1]

    return self


def max_power_no_bias(self):
    """ Random powers and biases for all SCs.
    """
    
    for i, small in enumerate(self.small_cells):
        small['power'] = self.power_limits[1]
        small['bias'] = self.CSB_limits[0]

    return self


def set_benchmark_pb(self):
    """ Sets the power and bias of all SCs to static levels for the benchmark.
    """
        
    for i, small in enumerate(self.small_cells):
        small['power'] = self.power_limits[1]
        small['bias'] = 7

    return self


def reset_to_zero(self):
    """ Re-sets the powers and biases of SCs to minimum levels so that the
        optimisation algorithm can test its mettle on a fresh network.
    """
    
    for i, small in enumerate(self.small_cells):
        small['power'] = self.power_limits[0]
        small['bias'] = self.CSB_limits[0]

    return self


def balance_bias(self):
    """Ensures the powers of SCs are maximised before the CSBs are used."""
    
    for small in self.small_cells:
        if small['power'] < self.power_limits[1]:
            if small['bias'] > 0:
                remainder = self.power_limits[1] - small['power']
                balance = remainder - small['bias']
                if balance >= 0:
                    small['power'] += small['bias']
                    small['bias'] = 0
                else:
                    small['power'] += remainder
                    small['bias'] -= remainder
    
    return self


def balance_network(self):
    """Optimise the network for the current distribution of UEs using
    evolved power & bias algorithm."""
    
    p_lim_0 = self.power_limits[0]
    b_lim_0 = self.CSB_limits[0]
    total_min = p_lim_0 + b_lim_0
    p_lim_1 = self.power_limits[1]
    b_lim_1 = self.CSB_limits[1]
    total_max = p_lim_1 + b_lim_1
    
    for small in self.small_cells:
        all_UEs = small['attached_users']
        macro_list = {}
        if all_UEs:
            N_s = len(small['attached_users'])
            R_s_avg = small['average_downlink']
            s_log_R = small['sum_log_R']
            # For each MC sector that the SC overlaps with find the number
            # of UEs that would attach to these MCs if no SC were present.
            for ind in all_UEs:
                user = self.users[ind]
                macro = user['macro']
                if macro in macro_list:
                    macro_list[macro][0] += 1
                    macro_list[macro][1].append(user['downlink'])
                    if user['downlink']:
                        macro_list[macro][2] += log(user['downlink'])
                else:
                    if user['downlink']:
                        macro_list[macro] = [1, [user['downlink']],
                                             log(user['downlink'])]
                    else:
                        macro_list[macro] = [1, [user['downlink']], 0]
            
            correction = 0
            
            # Look at each MC that the SC overlaps with and let the most-
            # overlapped-with-MC influence the correction most strongly
            for i in macro_list:
                N_ms = macro_list[i][0]
                """
                number is the number of UEs who have the same macro cell
                as their governing macro.
                """
                macro = self.macro_cells[i]
                N_m = len(macro['attached_users'])
                R_ms_avg = ave(macro_list[i][1])
                R_m_avg = self.macro_cells[i]['average_downlink']
                m_log_R = macro['sum_log_R']
                ms_log_R = macro_list[i][2]
                
                correction += eval(self.pb_algorithm)
            total = small['power'] + small['bias']
            if correction > 0:
                if total < total_max:
                    # Can increase something
                    new_total = total + correction
                    if new_total > total_max:
                        small['power'] = p_lim_1
                        small['bias'] = b_lim_1
                    else:
                        if new_total > p_lim_1:
                            small['power'] = p_lim_1
                            small['bias'] = new_total - p_lim_1
                        else:
                            small['power'] = new_total
                            small['bias'] = b_lim_0
                else:
                    # Everything is at its limits, can't increase anything
                    pass
            elif correction < 0:
                if total > total_min:
                    # Can decrease something
                    new_total = total + correction
                    if new_total < total_min:
                        small['power'] = p_lim_0
                        small['bias'] = b_lim_0
                    else:
                        if new_total > p_lim_1:
                            small['power'] = p_lim_1
                            small['bias'] = new_total - p_lim_1
                        else:
                            small['power'] = new_total
                            small['bias'] = b_lim_0
                else:
                    # Everything is at its limits, can't decrease anything
                    pass
    if not self.ALL_TOGETHER and (self.ABS_algorithm or self.BENCHMARK_ABS):
        self.update_network(FIST=True)
    else:
        self.update_network()
