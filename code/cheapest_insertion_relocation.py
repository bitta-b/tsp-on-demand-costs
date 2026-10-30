from __future__ import annotations
from dataclasses import dataclass, asdict
from math import hypot, isfinite
from typing import Sequence

@dataclass(frozen=True)
class SearchConfig:
    initial_k: int = 3
    expansion_step: int = 3

    def __post_init__(self):
        if self.initial_k < 1 or self.expansion_step < 1:
            raise ValueError("Candidate sizes must be positive")

@dataclass(frozen=True)
class AcceptedMove:
    node: int
    after: int
    actual_delta: int
    euclidean_delta: float
    old_cost: int
    new_cost: int
    k: int
    evaluated_since_restart: int

@dataclass
class SearchResult:
    initial_tour: tuple[int, ...]
    final_tour: tuple[int, ...]
    initial_cost: int
    final_cost: int
    evaluated_moves: int
    positive_euclidean_evaluations: int
    expansion_events: int
    final_pass_moves: int
    accepted: list[AcceptedMove]
    config: dict

def canonical_cycle(cycle: Sequence[int]) -> tuple[int, ...]:
    cycle = tuple(cycle)
    pivot = cycle.index(0)
    return cycle[pivot:] + cycle[:pivot]

def geographic_distances(coords: Sequence[Sequence[float]]) -> tuple:
    if len(coords) < 3 or any(len(p) != 2 or not all(isfinite(x) for x in p) for p in coords):
        raise ValueError("Use at least three finite 2D destination coordinates")
    return tuple(tuple(hypot(x[0] - y[0], x[1] - y[1]) for y in coords) for x in coords)

def cheapest_insertion(distances: Sequence[Sequence[float]]) -> tuple[int, ...]:

    n = len(distances)
    other = min(range(1, n), key=lambda j: (distances[0][j], j))
    cycle = [0, other]
    unvisited = set(range(n)) - set(cycle)
    while unvisited:
        best = min(
            (distances[a][u] + distances[u][cycle[(position + 1) % len(cycle)]]
             - distances[a][cycle[(position + 1) % len(cycle)]], u, a)
            for u in sorted(unvisited) for position, a in enumerate(cycle)
        )
        _, u, a = best
        cycle.insert(cycle.index(a) + 1, u)
        unvisited.remove(u)
    return canonical_cycle(cycle)

def valid_relocations(cycle: Sequence[int]) -> list[tuple[int, int]]:
    predecessor = {cycle[i]: cycle[i - 1] for i in range(len(cycle))}
    return [(u, a) for u in sorted(cycle) for a in sorted(cycle)
            if a != u and a != predecessor[u]]

def apply_relocation(cycle: Sequence[int], node: int, after: int) -> tuple[int, ...]:
    predecessor = cycle[(cycle.index(node) - 1) % len(cycle)]
    if after == node or after == predecessor:
        raise ValueError("Invalid/no-op relocation")
    updated = list(cycle)
    updated.remove(node)
    updated.insert(updated.index(after) + 1, node)
    return canonical_cycle(updated)

def affected_arcs(cycle: Sequence[int], node: int, after: int) -> tuple:
    position = cycle.index(node)
    previous, following = cycle[position - 1], cycle[(position + 1) % len(cycle)]
    if after == node or after == previous:
        raise ValueError("Invalid/no-op relocation")
    insertion_successor = cycle[(cycle.index(after) + 1) % len(cycle)]
    added = ((previous, following), (after, node), (node, insertion_successor))
    removed = ((previous, node), (node, following), (after, insertion_successor))
    return added, removed

def relocation_delta(cycle: Sequence[int], node: int, after: int,
                     costs: CostProvider) -> int:
    added, removed = affected_arcs(cycle, node, after)
    return sum(costs.cost(i, j) for i, j in added) - sum(costs.cost(i, j) for i, j in removed)

def _euclidean_delta(cycle, node, after, distances):
    added, removed = affected_arcs(cycle, node, after)
    return sum(distances[i][j] for i, j in added) - sum(distances[i][j] for i, j in removed)

def candidate_moves(cycle, distances, neighbor_rank, k, seen):

    successor = {cycle[i]: cycle[(i + 1) % len(cycle)] for i in range(len(cycle))}
    ranked = []
    for u, a in valid_relocations(cycle):
        if (u, a) in seen:
            continue
        if min(neighbor_rank[u][a], neighbor_rank[u][successor[a]]) <= k:
            ranked.append((_euclidean_delta(cycle, u, a, distances), u, a))
    return sorted(ranked)

def neighbor_ranks(distances):
    ranks = []
    for u, row in enumerate(distances):
        ordered = sorted((v for v in range(len(row)) if v != u), key=lambda v: (row[v], v))
        rank = {v: position + 1 for position, v in enumerate(ordered)}
        rank[u] = len(row) + 1
        ranks.append(rank)
    return ranks

def solve_capped(coords: Sequence[Sequence[float]], costs: CostProvider,
                 configured_max_k: int | None = None) -> SearchResult:
    if configured_max_k not in (3, 6, 12, None):
        raise ValueError("This controlled experiment permits caps 3, 6, 12, or full only")
    effective_max_k = len(coords) - 1 if configured_max_k is None else min(configured_max_k, len(coords) - 1)
    config = SearchConfig()
    distances = geographic_distances(coords)
    n = len(coords)
    ranks = neighbor_ranks(distances)
    cycle = cheapest_insertion(distances)
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

def normalize_original(result, cap):
    raw = asdict(result)
    normalized = dict(raw)
    normalized["accepted"] = [dict(m, length=1, block=[m["node"]]) for m in raw["accepted"]]
    normalized["accepted_by_type"] = {"1": len(result.accepted), "2": 0, "3": 0}
    normalized["evaluated_by_type"] = {"1": result.evaluated_moves, "2": 0, "3": 0}
    normalized["terminal_by_type"] = {"1": result.final_pass_moves, "2": 0, "3": 0}
    normalized["terminal_k"] = cap
    normalized["frozen_search"] = raw
    normalized["stopping_reason"] = "no_improvement_exhaustive_relocation" if cap == len(result.final_tour) - 2 else "no_improvement_at_cap_relocation"
    return normalized

def solve(coordinates, costs):
    n = len(coordinates)
    result = normalize_original(solve_capped(coordinates, costs, 12), min(12, n - 1))
    return {
        "variant": "C",
        "starts": [{"specification": {"index": 0, "seed": None, "cycle": result["initial_tour"][:-1]}, "search": result}],
        "starts_attempted": 1,
        "distinct_starting_tours": 1,
        "duplicated_start_indices": [],
        "selected_start_index": 0,
        "final_tour": result["final_tour"],
        "final_cost": result["final_cost"],
    }
