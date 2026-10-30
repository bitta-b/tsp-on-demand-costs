from __future__ import annotations
import argparse
import gzip
import heapq
import json
import random
from dataclasses import dataclass, asdict
from math import hypot
from pathlib import Path

SEEDS = (17, 41, 73)

SIZES = (8, 10, 12, 20, 50, 100)

SCENARIOS = {"grid": ("bidirectional", "one_way", "obstacles"),
             "random_geometric": ("bidirectional", "directed", "barrier")}

TICKS_PER_SECOND = 1_000_000

@dataclass
class RoadNetwork:
    family: str
    scenario: str
    seed: int
    coordinates: list[tuple[float, float]]

    edges: list[tuple[int, int, float, float, int]]
    destination_order: list[int]
    construction: dict

    def adjacency(self):
        graph = [[] for _ in self.coordinates]
        for i, j, length, speed, ticks in self.edges:
            graph[i].append((j, ticks))
        for row in graph:
            row.sort()
        return graph

def dijkstra(graph, origin):
    distances = [None] * len(graph)
    distances[origin] = 0
    heap = [(0, origin)]
    while heap:
        distance, node = heapq.heappop(heap)
        if distance != distances[node]:
            continue
        for neighbor, weight in graph[node]:
            proposed = distance + weight
            if distances[neighbor] is None or proposed < distances[neighbor]:
                distances[neighbor] = proposed
                heapq.heappush(heap, (proposed, neighbor))
    return distances

def strongly_connected(network):
    graph = network.adjacency()
    reverse = [[] for _ in graph]
    for u, row in enumerate(graph):
        for v, weight in row:
            reverse[v].append((u, weight))
    return all(x is not None for x in dijkstra(graph, 0)) and all(
        x is not None for x in dijkstra(reverse, 0))

def _link(edges, coords, rng, u, v, both, independent_directions=False):
    length = hypot(coords[u][0] - coords[v][0], coords[u][1] - coords[v][1])
    speed = rng.uniform(8.0, 16.0)
    edges.append((u, v, length, speed, max(1, round(TICKS_PER_SECOND * length / speed))))
    if both:
        reverse_speed = rng.uniform(8.0, 16.0) if independent_directions else speed
        edges.append((v, u, length, reverse_speed,
                      max(1, round(TICKS_PER_SECOND * length / reverse_speed))))

def grid_network(seed, scenario, side=12, spacing=300.0):
    if scenario not in SCENARIOS["grid"]:
        raise ValueError(scenario)
    rng = random.Random(seed)
    coords = [(column * spacing, row * spacing) for row in range(side) for column in range(side)]
    edges = []
    for row in range(side):
        for column in range(side):
            u = row * side + column
            for dr, dc in ((0, 1), (1, 0)):
                if row + dr == side or column + dc == side:
                    continue
                if scenario == "obstacles" and dc == 1 and column == side // 2 - 1 and row not in (0, side - 1):
                    continue
                v = (row + dr) * side + column + dc
                boundary = (dr == 0 and row in (0, side - 1)) or (dc == 0 and column in (0, side - 1))
                both = scenario != "one_way" or boundary
                if scenario == "one_way" and not both:
                    forward = (row % 2 == 0) if dc else (column % 2 == 0)
                    if not forward:
                        u_edge, v_edge = v, u
                    else:
                        u_edge, v_edge = u, v
                else:
                    u_edge, v_edge = u, v
                _link(edges, coords, rng, u_edge, v_edge, both, scenario == "one_way")
    order = list(range(len(coords)))
    random.Random(seed + 100_000).shuffle(order)
    result = RoadNetwork("grid", scenario, seed, coords, edges, order,
                         {"side": side, "spacing_m": spacing, "wall_gates": [0, side - 1],
                          "one_way_rule": "alternating row/column direction; bidirectional perimeter"})
    if not strongly_connected(result):
        raise ValueError("Generated grid is not strongly connected; case must be reported as failed")
    return result

def _barrier_allowed(p, q, extent):
    wall = extent / 2
    if (p[0] - wall) * (q[0] - wall) >= 0:
        return True
    fraction = (wall - p[0]) / (q[0] - p[0])
    crossing_y = p[1] + fraction * (q[1] - p[1])
    return crossing_y <= 0.15 * extent or crossing_y >= 0.85 * extent

def random_geometric_network(seed, scenario, node_count=144, extent=3300.0, radius=500.0):
    if scenario not in SCENARIOS["random_geometric"]:
        raise ValueError(scenario)
    rng = random.Random(seed)
    coords = [(rng.uniform(0, extent), rng.uniform(0, extent)) for _ in range(node_count)]
    allowed = {}
    for u in range(node_count):
        for v in range(u + 1, node_count):
            if scenario != "barrier" or _barrier_allowed(coords[u], coords[v], extent):
                allowed[u, v] = hypot(coords[u][0] - coords[v][0], coords[u][1] - coords[v][1])

    visited = {0}
    tree = set()
    while len(visited) < node_count:
        options = [(distance, u, v) for (u, v), distance in allowed.items()
                   if (u in visited) != (v in visited)]
        if not options:
            raise ValueError("Admissible geometric graph disconnected")
        _, u, v = min(options)
        tree.add((u, v))
        visited.update((u, v))
    pairs = tree | {pair for pair, distance in allowed.items() if distance <= radius}
    edges = []
    for u, v in sorted(pairs):
        both = scenario != "directed" or (u, v) in tree
        if not both and rng.random() < 0.5:
            u, v = v, u
        _link(edges, coords, rng, u, v, both, scenario == "directed")
    order = list(range(node_count))
    random.Random(seed + 100_000).shuffle(order)
    result = RoadNetwork("random_geometric", scenario, seed, coords, edges, order,
                         {"node_count": node_count, "extent_m": extent, "radius_m": radius,
                          "backbone": "bidirectional admissible Euclidean minimum spanning tree",
                          "barrier": "x=extent/2, open gates in bottom/top 15 percent"})
    if not strongly_connected(result):
        raise ValueError("Generated geometric network is not strongly connected")
    return result

def generate_network(family, scenario, seed):
    if family == "grid":
        return grid_network(seed, scenario)
    if family == "random_geometric":
        return random_geometric_network(seed, scenario)
    raise ValueError(family)

def destination_instance(network, n):
    if n > len(network.coordinates) or n < 3:
        raise ValueError("Destination count out of range")
    destinations = network.destination_order[:n]
    graph = network.adjacency()
    matrix = []
    for origin in destinations:
        distances = dijkstra(graph, origin)
        row = [distances[j] for j in destinations]
        if any(x is None for x in row):
            raise ValueError("Selected destinations not mutually reachable")
        matrix.append(row)
    return {"id": f"{network.family}_{network.scenario}_s{network.seed}_n{n}",
            "family": network.family, "scenario": network.scenario, "seed": network.seed,
            "n": n, "t0": "fixed_synthetic_reference_t0", "time_unit": "microseconds",
            "destination_road_nodes": destinations,
            "coordinates": [network.coordinates[i] for i in destinations],
            "ground_truth": matrix}

def generate_example(seed=7, cities=15):
    network = generate_network("grid", "one_way", seed)
    return destination_instance(network, cities)
