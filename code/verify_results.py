import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import statistics
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results"
INPUTS = ROOT / "examples" / "historical_inputs.zip"
NAMES = {
    "C": "Cheapest insertion + relocation",
    "M": "Multistart cheapest insertion + relocation",
    "MO": "Multistart cheapest insertion + Or-opt",
    "D": "Exhaustive relocation",
    "DR": "Doubly-rooted stem-and-cycle",
}
SIZES = (15, 16, 20, 50, 100)


def read_csv(filename):
    with (DATA / filename).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def historical_rows(records, n, method):
    role = "historical_control" if method in ("C", "M", "MO", "D") else "primary"
    return [r for r in records if int(r["n"]) == n and r["method"] == method and r["role"] == role]


def check_results():
    records = read_csv("all_runs.csv")
    published = read_csv("comparison.csv")
    assert len(records) == 1804, "Unexpected number of historical records"
    instance_ids = {r["instance_id"] for r in records}
    networks = {r["network_id"] for r in records}
    assert len(instance_ids) == 200 and len(networks) == 40
    assert Counter(int(r["n"]) for r in records if r["method"] == "C") == {n: 40 for n in SIZES}
    assert Counter(r["family"] for r in records if r["method"] == "C") == {
        "directed_grid": 50, "barrier_grid": 50,
        "geometric_bottleneck": 50, "clustered_corridors": 50,
    }
    baseline = {(int(r["cities"]), r["method"]): r for r in published}
    verified = {}
    for n in SIZES:
        for method, name in NAMES.items():
            rows = historical_rows(records, n, method)
            expected_count = 40
            assert len(rows) == expected_count, (n, method, "missing runs")
            completed = [r for r in rows if r["status"] == "VERIFIED"]
            expected = baseline[n, name]
            assert len(completed) == int(expected["completed"]), (n, method, "status count")
            values = {
                "mean_od_percent": statistics.mean(float(r["acquired_pct"]) for r in completed),
                "cost_change_vs_baseline_percent": statistics.mean(float(r["relative_C_pct"]) for r in completed),
                "median_cpu_seconds": statistics.median(float(r["historical_cpu_median_s"] if method != "DR"
                                                            else r["solver_cpu_s"]) for r in completed),
            }
            if n in (15, 16):
                values["exact_gap_percent"] = statistics.mean(float(r["gap_exact_pct"]) for r in completed)
                for r in completed:
                    cost = float(r["actual_cost_ticks"])
                    optimum = float(r["optimal_cost_ticks"])
                    assert optimum > 0
                    assert math.isclose(100 * (cost / optimum - 1), float(r["gap_exact_pct"]), abs_tol=1e-8)
            for metric, measured in values.items():
                assert math.isclose(measured, float(expected[metric]), rel_tol=1e-9, abs_tol=1e-8), (
                    n, method, metric, measured, expected[metric]
                )
            for r in completed:
                assert 0 < float(r["Q_acquired"]) <= n * (n - 1)
                assert math.isclose(float(r["acquired_pct"]), 100 * float(r["Q_acquired"]) / (n * (n - 1)), abs_tol=1e-9)
            verified[n, name] = {**expected, **{k: str(v) for k, v in values.items()}}
    acs = [r for r in records if r["method"] == "ACS"]
    dr = [r for r in records if r["method"] == "DR" and r["role"] == "primary"]
    assert Counter(r["status"] for r in acs) == {
        "VERIFIED": 166, "TIMEOUT": 206, "NOT_RUN_SYSTEMATIC_RESOURCE_STOP": 228,
    }
    assert Counter(r["status"] for r in dr) == {"VERIFIED": 198, "TIMEOUT": 2}
    assert len(acs) == 600 and len(dr) == 200
    print("PASS: 200 instances, 40 networks, 1,804 records")
    print("PASS: C, M, MO, D and DR summaries independently calculated from run-level data")
    print("PASS: certified small-instance gaps, acquisition percentages and ACS/DR status totals")
    return verified, records


def check_inputs(records):
    reference = {}
    for r in records:
        identity = r["instance_id"]
        if identity in reference:
            assert reference[identity] == r["input_sha256"], identity
        else:
            reference[identity] = r["input_sha256"]
    with zipfile.ZipFile(INPUTS) as archive:
        names = archive.namelist()
        assert len(names) == 200 and len(set(names)) == 200
        for identity, digest in reference.items():
            raw = archive.read(identity + ".json.gz")
            assert hashlib.sha256(raw).hexdigest() == digest, identity
            instance = json.loads(gzip.decompress(raw))
            n = instance["n"]
            assert instance["id"] == identity and len(instance["coordinates"]) == n
            assert len(instance["ground_truth"]) == n
            assert all(len(row) == n for row in instance["ground_truth"])
    print("PASS: all 200 original input files match the SHA-256 hashes in the historical records")


def replay(records, selection):
    from run_experiments import run
    references = {}
    for r in records:
        if r["method"] in ("C", "M", "MO") and r["role"] == "historical_control":
            references[r["instance_id"], r["method"]] = r
    instances = sorted({key[0] for key in references})
    if selection == "sample":
        instances = [next(i for i in instances if i.startswith(prefix) and i.endswith(suffix)) for prefix, suffix in [
            ("P4B_directed_grid_", "_n15"),
            ("P4B_barrier_grid_", "_n16"),
            ("P4B_geometric_bottleneck_", "_n20"),
            ("P4B_clustered_corridors_", "_n50"),
            ("P4B_clustered_corridors_", "_n100"),
        ]]
    pairs = 0
    with zipfile.ZipFile(INPUTS) as archive:
        for i, identity in enumerate(instances, 1):
            instance = json.loads(gzip.decompress(archive.read(identity + ".json.gz")))
            outcome = run(instance, methods=["C", "M", "MO"])
            for result in outcome:
                ref = references[identity, result["method"]]
                if result["cost"] != int(float(ref["actual_cost_ticks"])):
                    raise AssertionError(f"Tour cost mismatch: {identity}, {result['method']}")
                if result["unique_od"] != int(float(ref["Q_acquired"])):
                    raise AssertionError(f"Acquired OD mismatch: {identity}, {result['method']}")
                pairs += 1
            if selection == "all" and i % 20 == 0:
                print(f"Checked {i}/200 instances", flush=True)
    print(f"PASS: {pairs} C/M/MO historical cost and Q comparisons, across {len(instances)} real study inputs")
    print("CPU and wall times are not compared: they depend on the execution environment")
    print("ACS/DR historical experiments are NOT reproduced by this command")


def main():
    parser = argparse.ArgumentParser(description="Audit and optionally replay the original study data")
    parser.add_argument("--replay", choices=("sample", "all"), help="Rerun C, M and MO on actual saved study inputs")
    args = parser.parse_args()
    _, records = check_results()
    check_inputs(records)
    if args.replay:
        replay(records, args.replay)


if __name__ == "__main__":
    main()
