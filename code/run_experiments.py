import argparse
import csv
import gzip
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

from cost_oracle import CostOracle, PublicProblem
from generate_networks import SCENARIOS, generate_network, destination_instance
import cheapest_insertion_relocation
import multistart_cheapest_insertion_relocation
import multistart_cheapest_insertion_or_opt
import ant_colony_system
import doubly_rooted_stem_and_cycle

METHODS = {
    "C": ("Cheapest Insertion + Relocation", cheapest_insertion_relocation),
    "M": ("Multistart Relocation", multistart_cheapest_insertion_relocation),
    "MO": ("Multistart Or-opt", multistart_cheapest_insertion_or_opt),
    "ACS": ("Ant Colony System", ant_colony_system),
    "DR": ("Doubly-Rooted Stem-and-Cycle", doubly_rooted_stem_and_cycle),
}


def run(instance, seed=7, acs_iterations=1, dr_chain_depth=2, dr_max_rounds=1, methods=None):
    selected = methods or list(METHODS)
    rows = []
    for method in selected:
        name, module = METHODS[method]
        oracle = CostOracle(instance["ground_truth"], instance.get("id", instance.get("instance_id", "example")))
        start_wall, start_cpu = time.perf_counter(), time.process_time()
        if method in ("ACS", "DR"):
            problem = PublicProblem(tuple(range(instance["n"])),
                tuple(map(tuple, instance["coordinates"])), 0, oracle.dataset.scenario)
            config = (module.make_config(seed=seed, max_iterations=acs_iterations) if method == "ACS"
                      else module.make_config(seed=seed, chain_depth=dr_chain_depth, max_rounds=dr_max_rounds))
            result = module.solve(problem.public_json(), asdict(config), oracle, random.Random(seed))
            tour = result["tour"]
        else:
            result = module.solve(instance["coordinates"], oracle)
            tour = result["final_tour"]
        wall_s, cpu_s = time.perf_counter() - start_wall, time.process_time() - start_cpu
        cost = oracle.verify(tour)
        q = oracle.metrics()
        rows.append({"method": method, "name": name, "cost": cost,
            "cost_seconds": round(cost / 1_000_000, 6),
            "unique_od": q["distinct_directed_acquired"],
            "available_od": q["possible_directed_OD"], "calls": q["logical_calls"],
            "acquired_percent": round(q["acquired_percent"], 3),
            "wall_seconds": round(wall_s, 6), "cpu_seconds": round(cpu_s, 6)})
    return rows


def main():
    parser = argparse.ArgumentParser(description="Compare TSP methods on an offline network")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--cities", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--family", choices=tuple(SCENARIOS))
    parser.add_argument("--scenario")
    parser.add_argument("--methods", nargs="+", choices=tuple(METHODS))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    generate = any(x is not None for x in (args.cities, args.seed, args.family, args.scenario))
    if args.input and generate:
        parser.error("Use either --input or network-generation options, not both")
    if args.input:
        with gzip.open(args.input, "rt", encoding="utf8") as file:
            instance = json.load(file)
    elif generate:
        family = args.family or "grid"
        scenario = args.scenario or ("one_way" if family == "grid" else "directed")
        if scenario not in SCENARIOS[family]:
            parser.error(f"Valid scenarios for {family}: {', '.join(SCENARIOS[family])}")
        cities = args.cities if args.cities is not None else 15
        if cities < 3:
            parser.error("--cities must be at least 3")
        seed = args.seed if args.seed is not None else 7
        instance = destination_instance(generate_network(family, scenario, seed), cities)
    else:
        with gzip.open(Path(__file__).parents[1] / "examples" / "sample_instance.json.gz", "rt") as file:
            instance = json.load(file)

    selected = args.methods or list(METHODS)
    if "ACS" in selected and instance["n"] < 10:
        parser.error("ACS needs at least 10 cities in this implementation")
    results = run(instance, seed=args.seed if args.seed is not None else 7, methods=selected)
    base = next((r for r in results if r["method"] == "C"), None)
    print("DEMONSTRATION ONLY: this is not the 200-instance historical benchmark")
    print(f"Cities={instance['n']}; family={instance.get('family', 'saved')}; scenario={instance.get('scenario', 'saved')}")
    print("Q=distinct acquired directed OD; travel cost is in seconds, wall/CPU measure local execution")
    for r in results:
        vs_c = f"; cost vs C={100 * (r['cost']/base['cost']-1):+.2f}%" if base else ""
        print(f"{r['method']:>3}: travel={r['cost_seconds']:.3f} s; Q={r['unique_od']}/{r['available_od']} "
              f"({r['acquired_percent']:.2f}%); CPU={r['cpu_seconds']:.3f} s; wall={r['wall_seconds']:.3f} s{vs_c}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", newline="", encoding="utf8") as file:
            writer = csv.DictWriter(file, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
        print(f"Saved {args.output}")
    print("Custom runs are demonstrations; they are not reproductions of the frozen 200-instance study")


if __name__ == "__main__":
    main()
