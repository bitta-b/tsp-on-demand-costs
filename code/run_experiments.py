import argparse
import csv
import gzip
import json
import random
from dataclasses import asdict
from pathlib import Path

from cost_oracle import CostOracle, PublicProblem
import cheapest_insertion_relocation
import multistart_cheapest_insertion_relocation
import multistart_cheapest_insertion_or_opt
import ant_colony_system
import doubly_rooted_stem_and_cycle

METHODS = {
    "Cheapest Insertion + Relocation": cheapest_insertion_relocation,
    "Multistart Relocation": multistart_cheapest_insertion_relocation,
    "Multistart Or-opt": multistart_cheapest_insertion_or_opt,
    "Ant Colony System": ant_colony_system,
    "Doubly-Rooted Stem-and-Cycle": doubly_rooted_stem_and_cycle,
}

def run(instance, seed=7, acs_iterations=1, dr_chain_depth=2, dr_max_rounds=1):
    rows = []
    for name, module in METHODS.items():
        oracle = CostOracle(instance["ground_truth"], instance.get("instance_id", "example"))
        if name in ("Ant Colony System", "Doubly-Rooted Stem-and-Cycle"):
            problem = PublicProblem(tuple(range(instance["n"])),
                tuple(map(tuple, instance["coordinates"])), 0, oracle.dataset.scenario)
            config = (module.make_config(seed=seed, max_iterations=acs_iterations) if name == "Ant Colony System"
                      else module.make_config(seed=seed, chain_depth=dr_chain_depth, max_rounds=dr_max_rounds))
            result = module.solve(problem.public_json(), asdict(config), oracle, random.Random(seed))
            tour = result["tour"]
        else:
            result = module.solve(instance["coordinates"], oracle)
            tour = result["final_tour"]
        cost = oracle.verify(tour)
        q = oracle.metrics()
        rows.append({"method": name, "cost": cost, "unique_od": q["distinct_directed_acquired"],
            "available_od": q["possible_directed_OD"], "calls": q["logical_calls"],
            "acquired_percent": round(q["acquired_percent"], 3)})
    return rows

def main():
    parser = argparse.ArgumentParser(description="Run five heuristics on one offline sample.")
    parser.add_argument("--input", default=str(Path(__file__).parents[1] / "examples" / "sample_instance.json.gz"))
    parser.add_argument("--output")
    args = parser.parse_args()
    with gzip.open(args.input, "rt", encoding="utf8") as f:
        instance = json.load(f)
    rows = run(instance)
    for row in rows:
        print(f"{row['method']}: cost={row['cost']}, Q={row['unique_od']}/{row['available_od']} ({row['acquired_percent']}%)")
    if args.output:
        with open(args.output, "w", newline="", encoding="utf8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

if __name__ == "__main__":
    main()
