from __future__ import annotations
import random
from dataclasses import asdict
from typing import Sequence
from cheapest_insertion_relocation import (AcceptedMove, SearchConfig, SearchResult,
    apply_relocation, candidate_moves, cheapest_insertion, geographic_distances,
    neighbor_ranks, relocation_delta, canonical_cycle, normalize_original)

ADDITIONAL_START_SEEDS = (91001, 91009, 91019, 91033)

def cheapest_insertion_from_pair(distances, first, second):
    if first == second:
        raise ValueError("The two seed vertices must differ")
    cycle = [first, second]
    unvisited = set(range(len(distances))) - set(cycle)
    while unvisited:
        _, node, after = min(
            (distances[a][u] + distances[u][cycle[(i + 1) % len(cycle)]]
             - distances[a][cycle[(i + 1) % len(cycle)]], u, a)
            for u in sorted(unvisited) for i, a in enumerate(cycle)
        )
        cycle.insert(cycle.index(after) + 1, node)
        unvisited.remove(node)
    return canonical_cycle(cycle)

def initial_tours(coords):
    distances = geographic_distances(coords)
    n = len(coords)
    original_other = min(range(1, n), key=lambda j: (distances[0][j], j))
    starts = [{"index": 0, "seed": None, "seed_pair": [0, original_other],
               "cycle": cheapest_insertion(distances)}]
    for index, seed in enumerate(ADDITIONAL_START_SEEDS, 1):
        first, second = random.Random(seed).sample(range(n), 2)
        starts.append({"index": index, "seed": seed, "seed_pair": [first, second],
                       "cycle": cheapest_insertion_from_pair(distances, first, second)})

    return starts

def solve_capped_from_initial(coords: Sequence[Sequence[float]], costs: CostProvider,
                 configured_max_k: int | None = 12, *, initial_cycle) -> SearchResult:
    if configured_max_k not in (3, 6, 12, None):
        raise ValueError("This controlled experiment permits caps 3, 6, 12, or full only")
    effective_max_k = len(coords) - 1 if configured_max_k is None else min(configured_max_k, len(coords) - 1)
    config = SearchConfig()
    distances = geographic_distances(coords)
    n = len(coords)
    ranks = neighbor_ranks(distances)
    cycle = tuple(initial_cycle)
    initial_tour = cycle + cycle[:1]
    actual_cost = sum(costs.cost(cycle[i], cycle[(i + 1) % n]) for i in range(n))
    initial_cost = actual_cost
    accepted, evaluated, positive_evaluations, expansion_events = [], 0, 0, 0
    while True:
        k = min(config.initial_k, n - 1)
        seen = set()
        evaluated_this_restart = 0
        while True:
            improved = False
            for euclidean_delta, u, a in candidate_moves(cycle, distances, ranks, k, seen):
                seen.add((u, a))
                evaluated += 1
                evaluated_this_restart += 1
                positive_evaluations += int(euclidean_delta > 0)
                delta = relocation_delta(cycle, u, a, costs)
                if delta < 0:
                    old_cost = actual_cost
                    actual_cost += delta
                    accepted.append(AcceptedMove(u, a, delta, euclidean_delta,
                                                 old_cost, actual_cost, k, evaluated_this_restart))
                    cycle = apply_relocation(cycle, u, a)
                    improved = True
                    break
            if improved:
                break
            if k == effective_max_k:
                return SearchResult(initial_tour, cycle + cycle[:1], initial_cost, actual_cost,
                                    evaluated, positive_evaluations, expansion_events,
                                    evaluated_this_restart, accepted, asdict(config))
            k = min(effective_max_k, k + config.expansion_step)
            expansion_events += 1

def solve(coordinates, costs):
    starts = []
    cap = min(12, len(coordinates) - 1)
    for spec in initial_tours(coordinates):
        result = normalize_original(solve_capped_from_initial(coordinates, costs, 12, initial_cycle=spec["cycle"]), cap)
        starts.append({"specification": spec, "search": result})
    best = min(range(len(starts)), key=lambda i: (starts[i]["search"]["final_cost"],
                            tuple(starts[i]["search"]["final_tour"]), i))
    tours = [tuple(s["search"]["initial_tour"]) for s in starts]
    return {"variant": "M", "starts": starts, "starts_attempted": len(starts),
            "distinct_starting_tours": len(set(tours)),
            "duplicated_start_indices": [i for i, tour in enumerate(tours) if tour in tours[:i]],
            "selected_start_index": best,
            "final_tour": starts[best]["search"]["final_tour"],
            "final_cost": starts[best]["search"]["final_cost"]}
