from __future__ import annotations
import math
import platform
import random
from cost_oracle import AlgorithmConfig, PublicProblem

# Adapted from Dorigo and Gambardella (1997).

ALGORITHM_ID = 'ACS97-3opt-Geo-v1'

SOURCE_ID = 'Dorigo_Gambardella_1997_IEEE_TEC_10.1109_4235.585892'

SOURCE_VERSION = 'Section_VI_ATSP_architecture__ADAPTATION_op_spec_1.0.0'

FIXED = {'m': 10, 'beta': 2, 'q0': 0.98, 'alpha': 0.1, 'rho': 0.1,
         'candidate_length': 20, 'operational_spec_version': '1.0.0'}

ADAPTATIONS = ('geometric_NN_reference_with_actual_length', 'fixed_outgoing_geometric_candidates',
               'all_three_new_arcs_candidate_membership', 'declared_first_improvement_and_flags',
               'declared_serial_RNG_and_tie_conventions', 'explicit_iteration_budget')

def rng_runtime():
    return f'{platform.python_implementation()}-{platform.python_version()}-random.Random-MT19937-state3'

def make_config(*, seed, max_iterations):
    if type(seed) is not int or type(max_iterations) is not int or max_iterations <= 0:
        raise ValueError('Explicit integer seed and positive max_iterations are mandatory')
    return AlgorithmConfig(ALGORITHM_ID, SOURCE_ID, SOURCE_VERSION, 'G', 'ADAPTATION', seed,
        'geometric_NN_actual_reference_length', 'outgoing_geometric_20_all_three_new_arcs',
        'explicit_completed_iteration_budget', {**FIXED, 'max_iterations': max_iterations,
                                               'rng_runtime': rng_runtime()}, ADAPTATIONS)

def validate_inputs(public, config, rng):
    expected = {'node_ids', 'coordinates', 'depot', 'scenario', 'cost_unit'}
    if not isinstance(public, dict) or set(public) != expected:
        raise ValueError('Only the declared public problem fields are permitted')
    problem = PublicProblem(tuple(public['node_ids']), tuple(map(tuple, public['coordinates'])),
                            public['depot'], public['scenario'], public['cost_unit'])
    fields = {'algorithm_id': ALGORITHM_ID, 'source_id': SOURCE_ID, 'source_version': SOURCE_VERSION,
              'mode': 'G', 'fidelity': 'ADAPTATION',
              'initialization_policy': 'geometric_NN_actual_reference_length',
              'candidate_policy': 'outgoing_geometric_20_all_three_new_arcs',
              'stopping_policy': 'explicit_completed_iteration_budget'}
    if any(config.get(k) != v for k, v in fields.items()):
        raise ValueError('Algorithm identity, source, fidelity and policies are fixed')
    if tuple(config.get('adaptations', ())) != ADAPTATIONS:
        raise ValueError('Every declared adaptation must remain visible')
    params = config.get('parameters', {})
    if not isinstance(params, dict) or set(params) != set(FIXED) | {'max_iterations', 'rng_runtime'}:
        raise ValueError('No omitted or extra operational parameters permitted')
    if any(params[k] != v for k, v in FIXED.items()):
        raise ValueError('Section VI-A settings and operational version are fixed')
    if any(type(params[k]) is not int for k in ('m', 'beta', 'candidate_length')):
        raise ValueError('Integer fixed parameters required')
    if type(params['max_iterations']) is not int or params['max_iterations'] <= 0:
        raise ValueError('Positive explicit iteration budget required')
    if len(problem.node_ids) < params['m']:
        raise ValueError('m=10 exceeds n; repeated ant starting cities are not authorized')
    if params['rng_runtime'] != rng_runtime():
        raise ValueError('RNG runtime differs from the serialized configuration')
    seed = config.get('seed')
    if type(seed) is not int or type(rng) is not random.Random:
        raise ValueError('Integer seed and fresh random.Random required')
    if rng.getstate() != random.Random(seed).getstate():
        raise ValueError('RNG is not in the configured fresh initial state')
    return problem, params

def canonical(cycle, depot):
    cycle = tuple(cycle)
    if len(cycle) < 3 or len(set(cycle)) != len(cycle) or depot not in cycle:
        raise ValueError('Distinct cycle vertices and declared depot required')
    if any(type(v) is not int for v in cycle):
        raise ValueError('Integer node IDs required')
    i = cycle.index(depot)
    return cycle[i:] + cycle[:i]

def actual_cost(client, i, j, stage, context):
    value = client.cost(i, j, stage=stage, context=context)
    if type(value) is not int or value <= 0:
        raise ValueError('Oracle must return positive integer actual ticks')
    return value

def cycle_cost(cycle, client, stage, context):
    return sum(actual_cost(client, a, cycle[(i+1) % len(cycle)], stage,
                           {**context, 'tour_arc': i}) for i, a in enumerate(cycle))

def geometric_distances(problem):
    coordinates = dict(zip(problem.node_ids, problem.coordinates))
    distance = {}
    for i in sorted(problem.node_ids):
        for j in sorted(problem.node_ids):
            if i == j: continue
            value = math.hypot(coordinates[i][0]-coordinates[j][0], coordinates[i][1]-coordinates[j][1])
            if not math.isfinite(value): raise ValueError('Unrepresentable geometric distance')
            distance[i, j] = value
    return distance

def geometric_nn(problem, distance):
    cycle = [problem.depot]
    remaining = set(problem.node_ids)-{problem.depot}
    while remaining:
        selected = min(remaining, key=lambda j: (distance[cycle[-1], j], j))
        cycle.append(selected); remaining.remove(selected)
    return tuple(cycle)

def candidate_lists(problem, distance):
    width = min(20, len(problem.node_ids)-1)
    return {i: tuple(sorted((j for j in problem.node_ids if j != i),
                            key=lambda j: (distance[i, j], j))[:width])
            for i in sorted(problem.node_ids)}

def positive_finite(value):
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ArithmeticError('Pheromone must be finite and strictly positive')
    return float(value)

class Pheromone:
    def __init__(self, tau0):
        self.tau0 = positive_finite(tau0)
        self.values = {}

    def get(self, i, j):
        if i == j: raise ValueError('No self-arc pheromone')
        return self.values.get((i, j), self.tau0)

    def update(self, i, j, rate, target):
        if not 0 < rate < 1: raise ValueError('Update rate must be in (0,1)')
        old = self.get(i, j)
        target = positive_finite(target)
        new = positive_finite((1-rate)*old + rate*target)
        self.values[i, j] = new
        return old, new

    def local(self, i, j):
        return self.update(i, j, 0.1, self.tau0)

    def global_best(self, cycle, cost):
        if type(cost) is not int or cost <= 0: raise ValueError('Positive actual best-tour objective required')
        target = positive_finite(1.0/cost)
        return [(a, cycle[(i+1) % len(cycle)], *self.update(a, cycle[(i+1) % len(cycle)], 0.1, target))
                for i, a in enumerate(cycle)]

def describe_move(cycle, tails):
    cycle = tuple(cycle)
    if len(cycle) < 3 or len(set(cycle)) != len(cycle) or any(type(x) is not int for x in cycle):
        raise ValueError('Distinct integer cycle required')
    if len(tails) != 3 or any(type(x) is not int or x not in cycle for x in tails) or len(set(tails)) != 3:
        raise ValueError('Three distinct existing cut tails required')
    k, p, r = tails; n = len(cycle); anchor = cycle.index(k)
    off_p, off_r = (cycle.index(p)-anchor) % n, (cycle.index(r)-anchor) % n
    if not 0 < off_p < off_r < n: raise ValueError('Cuts must follow the declared cyclic order')
    successor = dict(zip(cycle, cycle[1:]+cycle[:1]))
    l, q, s = successor[k], successor[p], successor[r]
    removed = ((k, l), (p, q), (r, s))
    added = ((k, q), (p, s), (r, l))
    if len(set(added)) != 3 or any(a == b for a, b in added) or set(added) == set(removed):
        raise ValueError('Degenerate or no-op rewire')
    return removed, added

def validate_successors(nodes, successor, start):
    nodes = tuple(nodes)
    if set(successor) != set(nodes) or set(successor.values()) != set(nodes):
        raise ValueError('Successor map does not have exactly one incoming/outgoing arc per node')
    walked = [start]
    for _ in range(len(nodes)-1):
        walked.append(successor[walked[-1]])
    if len(set(walked)) != len(nodes) or successor[walked[-1]] != start:
        raise ValueError('Disconnected subtours or missing tour closure')
    return tuple(walked)

def apply_move(cycle, tails):
    removed, added = describe_move(cycle, tails)
    cycle = tuple(cycle)
    successor = dict(zip(cycle, cycle[1:]+cycle[:1]))
    for a, b in added: successor[a] = b
    result = validate_successors(cycle, successor, cycle[0])
    if result == cycle: raise ValueError('No-op move')
    return result

def directed_delta(cycle, tails, client, context=None):
    removed, added = describe_move(cycle, tails)
    apply_move(cycle, tails)
    total = 0
    for index, (new, old) in enumerate(zip(added, removed)):
        total += actual_cost(client, *new, 'search', {**(context or {}), 'boundary': index, 'role': 'added'})
        total -= actual_cost(client, *old, 'search', {**(context or {}), 'boundary': index, 'role': 'removed'})
    return total

def cut_triples(cycle, anchor):
    position = cycle.index(anchor); n = len(cycle)
    for b in range(1, n-1):
        for c in range(b+1, n):
            yield anchor, cycle[(position+b) % n], cycle[(position+c) % n]

def eligible(added, candidates):
    return all(b in candidates[a] for a, b in added)

def local_search(cycle, candidates, client, *, depot, context):
    cycle = canonical(cycle, depot)
    total = cycle_cost(cycle, client, 'search', {**context, 'phase': 'local_initial_objective'})
    flags = {v: False for v in sorted(cycle)}
    counters = {'evaluated': 0, 'filter_rejected': 0, 'delta_rejected': 0, 'accepted': 0, 'geometric_excluded': 0}
    client.record('start', {**context, 'phase': 'local_search', 'tour': list(cycle)+[depot],
                             'actual_cost_ticks': total, 'dont_look_on': []})
    while not all(flags.values()):
        anchor = min(v for v in flags if not flags[v])
        accepted = False
        for tails in cut_triples(cycle, anchor):
            removed, added = describe_move(cycle, tails)
            changed = apply_move(cycle, tails)
            if not eligible(added, candidates):
                counters['geometric_excluded'] += 1
                continue
            counters['evaluated'] += 1
            details = {**context, 'phase': 'local_search', 'move': counters['evaluated'], 'cut_tails': list(tails)}
            client.record('candidate', details)
            new0 = actual_cost(client, *added[0], 'search', {**details, 'boundary': 0, 'role': 'added'})
            old0 = actual_cost(client, *removed[0], 'search', {**details, 'boundary': 0, 'role': 'removed'})
            if new0 >= old0:
                counters['filter_rejected'] += 1
                client.record('move_rejected', {**details, 'reason': 'actual_source_filter'})
                continue
            delta = new0-old0
            for boundary in (1, 2):
                delta += actual_cost(client, *added[boundary], 'search', {**details, 'boundary': boundary, 'role': 'added'})
                delta -= actual_cost(client, *removed[boundary], 'search', {**details, 'boundary': boundary, 'role': 'removed'})
            if delta >= 0:
                counters['delta_rejected'] += 1
                client.record('move_rejected', {**details, 'reason': 'nonnegative_actual_delta', 'delta': delta})
                continue
            before = total; old_cycle = cycle; before_flags = sorted(v for v in flags if flags[v])
            cycle = canonical(changed, depot); total += delta
            affected = sorted({x for arc in removed+added for x in arc})
            for v in affected: flags[v] = False
            counters['accepted'] += 1
            client.record('move_accepted', {**details, 'delta': delta, 'old_cost': before, 'new_cost': total,
                'old_tour': list(old_cycle)+[depot], 'new_tour': list(cycle)+[depot],
                'reset_nodes': affected, 'flags_on_before': before_flags,
                'flags_on_after': sorted(v for v in flags if flags[v])})
            client.record('restart', {**context, 'phase': 'local_search', 'smallest_active_anchor': min(v for v in flags if not flags[v])})
            accepted = True
            break
        if not accepted:
            flags[anchor] = True
            client.record('candidate', {**context, 'phase': 'dont_look', 'action': 'set_on',
                'node': anchor, 'flags_on': sorted(v for v in flags if flags[v])})
    client.record('stop', {**context, 'phase': 'local_search', 'reason': 'all_dont_look_bits_set_under_declared_policy',
                           'flags_on': sorted(flags), 'tour': list(cycle)+[depot], 'actual_cost_ticks': total})
    return cycle, total, counters

def distribution(ordered, costs, pheromones):
    ordered = tuple(ordered)
    if not ordered or len(set(ordered)) != len(ordered): raise ValueError('Distinct nonempty alternatives required')
    logs = []
    for j in ordered:
        c, tau = costs[j], pheromones[j]
        if type(c) is not int or c <= 0 or not math.isfinite(tau) or tau <= 0:
            raise ValueError('Positive actual costs and finite positive pheromone required')
        logs.append(math.log(tau)-2*math.log(c))
    maximum = max(logs)
    scaled = [math.exp(v-maximum) for v in logs]
    denominator = math.fsum(scaled)
    probabilities = [v/denominator for v in scaled]
    return logs, scaled, probabilities

def select_next(current, unvisited, candidates, pheromone, client, rng, context):
    remaining = set(unvisited)
    if not remaining: raise ValueError('Transition requires an unvisited node')
    eligible = tuple(j for j in candidates[current] if j in remaining)
    if eligible:
        costs = {j: actual_cost(client, current, j, 'construction',
                                {**context, 'decision': 'candidate_score'}) for j in eligible}
        logs, _, probabilities = distribution(eligible, costs, {j: pheromone.get(current, j) for j in eligible})
        q = rng.random()
        if q <= 0.98:
            winner = min(range(len(eligible)), key=lambda k: (-logs[k], eligible[k]))
            reason = 'exploitation'
        else:
            u = rng.random()
            winner = len(eligible)-1
            for k in range(len(eligible)):
                if u < math.fsum(probabilities[:k+1]): winner = k; break
            reason = 'proportional_exploration'
        selected = eligible[winner]
        client.record('candidate', {**context, 'phase': 'transition', 'origin': current,
            'ordered_alternatives': list(eligible), 'probabilities': probabilities, 'q': q,
            'selected': selected, 'decision': reason})
        return selected, costs[selected]
    costs = {j: actual_cost(client, current, j, 'fallback',
                            {**context, 'decision': 'actual_nearest_outside_list'}) for j in sorted(remaining)}
    selected = min(remaining, key=lambda j: (costs[j], j))
    client.record('candidate', {**context, 'phase': 'transition', 'origin': current,
        'ordered_alternatives': sorted(remaining), 'selected': selected, 'decision': 'actual_nearest_fallback'})
    return selected, costs[selected]

def solve(public_problem, config, client, rng):
    problem, parameters = validate_inputs(public_problem, config, rng)
    ids = tuple(sorted(problem.node_ids)); n = len(ids)
    distance = geometric_distances(problem)
    reference = geometric_nn(problem, distance)
    reference_cost = cycle_cost(reference, client, 'initialization', {'phase': 'geometric_NN_actual_length'})
    tau0 = 1.0/(n*reference_cost)
    pheromone = Pheromone(tau0)
    candidates = candidate_lists(problem, distance)
    client.record('start', {'phase': 'initialization_complete', 'algorithm_id': ALGORITHM_ID,
        'fidelity': 'ADAPTATION', 'reference_tour': list(reference)+[problem.depot],
        'actual_L_nn': reference_cost, 'tau0': tau0, 'rng_runtime': rng_runtime(),
        'candidate_length_effective': min(20, n-1)})
    global_best = None
    selected = None
    totals = {'evaluated': 0, 'filter_rejected': 0, 'delta_rejected': 0, 'accepted': 0,
              'geometric_excluded': 0, 'ants_locally_searched': 0, 'local_pheromone_updates': 0,
              'global_pheromone_updates': 0}
    for iteration in range(parameters['max_iterations']):
        starts = rng.sample(list(ids), parameters['m'])
        cycles = [[v] for v in starts]
        unvisited = [set(ids)-{v} for v in starts]
        built_cost = [0]*parameters['m']
        client.record('start', {'phase': 'iteration', 'iteration': iteration, 'ant_start_cities': starts})
        for step in range(n-1):
            for ant in range(parameters['m']):
                context = {'iteration': iteration, 'ant': ant, 'step': step}
                previous = cycles[ant][-1]
                node, value = select_next(previous, unvisited[ant], candidates, pheromone, client, rng, context)
                built_cost[ant] += value
                cycles[ant].append(node); unvisited[ant].remove(node)
                old, new = pheromone.local(previous, node)
                totals['local_pheromone_updates'] += 1
                client.record('candidate', {**context, 'phase': 'pheromone_local', 'origin': previous,
                    'destination': node, 'old_tau': old, 'new_tau': new})
        for ant in range(parameters['m']):
            a, b = cycles[ant][-1], starts[ant]
            built_cost[ant] += actual_cost(client, a, b, 'construction',
                {'iteration': iteration, 'ant': ant, 'phase': 'close_constructed_tour'})
            old, new = pheromone.local(a, b)
            totals['local_pheromone_updates'] += 1
            client.record('candidate', {'iteration': iteration, 'ant': ant, 'phase': 'pheromone_local',
                'origin': a, 'destination': b, 'old_tau': old, 'new_tau': new, 'closure': True})
        for ant in range(parameters['m']):
            cycle = canonical(cycles[ant], problem.depot)
            client.record('start', {'iteration': iteration, 'ant': ant, 'phase': 'constructed_ant',
                'tour': list(cycle)+[problem.depot], 'actual_cost_ticks': built_cost[ant]})
            improved, value, stats = local_search(cycle, candidates, client, depot=problem.depot,
                                                   context={'iteration': iteration, 'ant': ant})
            totals['ants_locally_searched'] += 1
            for key, number in stats.items(): totals[key] += number
            if value > built_cost[ant]: raise AssertionError('Strict local search worsened an ant')
            key = (value, improved)
            if global_best is None or key < global_best:
                global_best, selected = key, {'iteration': iteration, 'ant': ant}
            client.record('stop', {'iteration': iteration, 'ant': ant, 'phase': 'ant_local_complete',
                'tour': list(improved)+[problem.depot], 'actual_cost_ticks': value, 'local_counts': stats})
        value, best_cycle = global_best
        updates = pheromone.global_best(best_cycle, value)
        totals['global_pheromone_updates'] += len(updates)
        client.record('candidate', {'phase': 'pheromone_global', 'iteration': iteration,
            'global_best_tour': list(best_cycle)+[problem.depot], 'actual_cost_ticks': value,
            'updates': [list(row) for row in updates]})
    client.record('stop', {'phase': 'solver', 'reason': 'explicit_completed_iteration_budget',
        'iterations_completed': parameters['max_iterations'], 'actual_cost_ticks': global_best[0]})
    return {'tour': list(global_best[1])+[problem.depot], 'stopping_reason': 'explicit_completed_iteration_budget',
            'diagnostics': {'algorithm_id': ALGORITHM_ID, 'fidelity': 'ADAPTATION', 'mode': 'G',
                'reference_actual_L_nn': reference_cost, 'reference_tau0': tau0,
                'iterations_completed': parameters['max_iterations'], 'selected_ant': selected,
                'actual_objective_ticks': global_best[0], 'counts': totals,
                'local_certificate_scope': 'declared_geometric_filter_and_dont_look_procedure_only',
                'rng_runtime': rng_runtime()}}
