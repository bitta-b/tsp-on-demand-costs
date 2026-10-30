from __future__ import annotations
from cheapest_insertion_relocation import (canonical_cycle, geographic_distances,
    cheapest_insertion, neighbor_ranks, candidate_moves, relocation_delta, apply_relocation)
from multistart_cheapest_insertion_relocation import initial_tours

def block_vertices(cycle, node, length):
    n = len(cycle)
    if not 1 <= length <= n - 2:
        raise ValueError("A nondegenerate move leaves at least two vertices")
    i = cycle.index(node)
    return tuple(cycle[(i + offset) % n] for offset in range(length))

def valid_block_relocations(cycle, length):
    n = len(cycle)
    if length < 1:
        raise ValueError("Block length must be positive")
    if length > n - 2:
        return []
    moves = []
    for node in sorted(cycle):
        block = block_vertices(cycle, node, length)
        previous = cycle[(cycle.index(node) - 1) % n]
        moves.extend((node, after) for after in sorted(cycle)
                     if after not in block and after != previous)
    return moves

def boundary_arcs(cycle, node, after, length):
    block = block_vertices(cycle, node, length)
    n = len(cycle)
    i = cycle.index(node)
    previous = cycle[(i - 1) % n]
    if after in block or after == previous:
        raise ValueError("Overlapping or no-op block relocation")
    following = cycle[(i + length) % n]
    successor = cycle[(cycle.index(after) + 1) % n]
    return (((previous, following), (after, block[0]), (block[-1], successor)),
            ((previous, block[0]), (block[-1], following), (after, successor)))

def block_delta(cycle, node, after, length, costs):
    added, removed = boundary_arcs(cycle, node, after, length)
    return sum(costs.cost(i, j) for i, j in added) - sum(costs.cost(i, j) for i, j in removed)

def apply_block(cycle, node, after, length):
    boundary_arcs(cycle, node, after, length)
    block = block_vertices(cycle, node, length)
    remainder = [x for x in cycle if x not in block]
    cut = remainder.index(after) + 1
    return canonical_cycle(remainder[:cut] + list(block) + remainder[cut:])

def block_candidates(cycle, distances, ranks, k, seen, length):
    successor = {cycle[i]: cycle[(i + 1) % len(cycle)] for i in range(len(cycle))}
    ranked = []
    for node, after in valid_block_relocations(cycle, length):
        if (node, after) in seen:
            continue

        if min(ranks[node][after], ranks[node][successor[after]]) <= k:
            added, removed = boundary_arcs(cycle, node, after, length)
            delta = sum(distances[i][j] for i, j in added) - sum(distances[i][j] for i, j in removed)
            ranked.append((delta, node, after))
    return sorted(ranked)

def solve_augmented(coords, costs, initial_cycle=None):
    distances = geographic_distances(coords)
    n = len(coords)
    ranks = neighbor_ranks(distances)
    cycle = cheapest_insertion(distances) if initial_cycle is None else tuple(initial_cycle)
    initial_tour = cycle + cycle[:1]
    actual_cost = sum(costs.cost(cycle[i], cycle[(i + 1) % n]) for i in range(n))
    initial_cost = actual_cost
    accepted = []
    evaluated = {1: 0, 2: 0, 3: 0}
    positive = 0
    expansions = 0
    cap = min(12, n - 1)
    while True:
        k = min(3, n - 1)
        seen = {length: set() for length in (1, 2, 3)}
        terminal = {1: 0, 2: 0, 3: 0}
        evaluated_this_restart = 0
        while True:
            improved = False

            for length in (1, 2, 3):
                moves = (candidate_moves(cycle, distances, ranks, k, seen[1]) if length == 1
                         else block_candidates(cycle, distances, ranks, k, seen[length], length))
                for geometric_delta, node, after in moves:
                    seen[length].add((node, after))
                    evaluated[length] += 1
                    terminal[length] += 1
                    evaluated_this_restart += 1
                    positive += int(geometric_delta > 0)
                    delta = (relocation_delta(cycle, node, after, costs) if length == 1
                             else block_delta(cycle, node, after, length, costs))
                    if delta < 0:
                        old = actual_cost
                        actual_cost += delta
                        block = list(block_vertices(cycle, node, length))
                        accepted.append({"node": node, "after": after, "length": length,
                                         "block": block, "actual_delta": delta,
                                         "euclidean_delta": geometric_delta, "old_cost": old,
                                         "new_cost": actual_cost, "k": k,
                                         "evaluated_since_restart": evaluated_this_restart})
                        cycle = (apply_relocation(cycle, node, after) if length == 1
                                 else apply_block(cycle, node, after, length))
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
            if k == cap:
                return {"initial_tour": initial_tour, "final_tour": cycle + cycle[:1],
                        "initial_cost": initial_cost, "final_cost": actual_cost,
                        "accepted": accepted,
                        "accepted_by_type": {str(l): sum(m["length"] == l for m in accepted) for l in (1, 2, 3)},
                        "evaluated_by_type": {str(l): evaluated[l] for l in (1, 2, 3)},
                        "terminal_by_type": {str(l): terminal[l] for l in (1, 2, 3)},
                        "evaluated_moves": sum(evaluated.values()),
                        "positive_euclidean_evaluations": positive, "expansion_events": expansions,
                        "final_pass_moves": sum(terminal.values()), "terminal_k": k,
                        "config": {"initial_k": 3, "expansion_step": 3, "cap": cap, "move_order": [1, 2, 3]},
                        "stopping_reason": "no_improvement_at_cap_all_three_types"}
            k = min(cap, k + 3)
            expansions += 1

def solve(coordinates, costs):
    starts = []
    for spec in initial_tours(coordinates):
        result = solve_augmented(coordinates, costs, spec["cycle"])
        starts.append({"specification": spec, "search": result})
    best = min(range(len(starts)), key=lambda i: (starts[i]["search"]["final_cost"],
                            tuple(starts[i]["search"]["final_tour"]), i))
    tours = [tuple(s["search"]["initial_tour"]) for s in starts]
    return {"variant": "MO", "starts": starts, "starts_attempted": len(starts),
            "distinct_starting_tours": len(set(tours)),
            "duplicated_start_indices": [i for i, tour in enumerate(tours) if tour in tours[:i]],
            "selected_start_index": best,
            "final_tour": starts[best]["search"]["final_tour"],
            "final_cost": starts[best]["search"]["final_cost"]}
