import gzip
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from cost_oracle import CostOracle
from run_experiments import run

class TestAlgorithms(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with gzip.open(ROOT / "examples" / "sample_instance.json.gz", "rt") as f:
            cls.instance = json.load(f)

    def test_cache(self):
        oracle = CostOracle(self.instance["ground_truth"])
        oracle.cost(0, 1)
        oracle.cost(0, 1)
        oracle.cost(1, 0)
        values = oracle.metrics()
        self.assertEqual(values["logical_calls"], 3)
        self.assertEqual(values["distinct_directed_acquired"], 2)
        self.assertEqual(values["cache_hits"], 1)

    def test_cache_is_separate_for_each_run(self):
        first = CostOracle(self.instance["ground_truth"])
        second = CostOracle(self.instance["ground_truth"])
        first.cost(0, 1)
        first.cost(0, 1)
        second.cost(0, 1)
        self.assertEqual(first.metrics()["distinct_directed_acquired"], 1)
        self.assertEqual(second.metrics()["distinct_directed_acquired"], 1)
        self.assertEqual(first.metrics()["logical_calls"], 2)
        self.assertEqual(second.metrics()["logical_calls"], 1)

    def test_invalid_tour_is_rejected(self):
        oracle = CostOracle(self.instance["ground_truth"])
        n = self.instance["n"]
        self.assertRaises(ValueError, oracle.verify, list(range(n)))
        self.assertRaises(ValueError, oracle.verify, list(range(n - 1)) + [0, 0])

    def test_five_algorithms(self):
        results = run(self.instance)
        self.assertEqual(len(results), 5)
        for result in results:
            self.assertGreater(result["cost"], 0)
            self.assertLessEqual(result["unique_od"], result["available_od"])
            self.assertGreaterEqual(result["calls"], result["unique_od"])
        self.assertEqual([(r["cost"],r["unique_od"]) for r in results[:3]],
            [(2020560674,208),(1785339689,210),(1785339689,210)])
        self.assertEqual([(r["cost"], r["unique_od"]) for r in results[3:]],
                         [(1785339689, 210), (2318999081, 142)])
        self.assertEqual([r["calls"] for r in results[:3]],
            [1527+15,8085+15,19947+15])

    def test_historical_results(self):
        import csv
        with (ROOT / "results" / "all_runs.csv").open(newline="", encoding="utf8") as file:
            records = list(csv.DictReader(file))
        self.assertEqual(len(records), 1804)
        self.assertEqual(len({r["instance_id"] for r in records}), 200)
        for n in (15, 16, 20, 50, 100):
            for method in ("C", "M", "MO", "D"):
                runs = [r for r in records if r["method"] == method and int(r["n"]) == n]
                self.assertEqual(len(runs), 40)
                self.assertTrue(all(r["status"] == "VERIFIED" for r in runs))

    def test_study_summaries_match_raw_records(self):
        from verify_results import check_results
        summaries, records = check_results()
        self.assertEqual(len(summaries), 25)
        self.assertEqual(len(records), 1804)

    def test_original_study_inputs(self):
        from verify_results import check_results, check_inputs
        _, records = check_results()
        check_inputs(records)

    def test_one_original_study_instance_replays(self):
        import csv
        import zipfile
        name = "P4B_clustered_corridors_s4974870_n15"
        with zipfile.ZipFile(ROOT / "examples" / "historical_inputs.zip") as archive:
            instance = json.loads(gzip.decompress(archive.read(name + ".json.gz")))
        with (ROOT / "results" / "all_runs.csv").open(newline="", encoding="utf8") as f:
            records = list(csv.DictReader(f))
        expected = {r["method"]: r for r in records if r["instance_id"] == name and
                    r["method"] in ("C", "M", "MO") and r["role"] == "historical_control"}
        for item in run(instance, methods=["C", "M", "MO"]):
            record = expected[item["method"]]
            self.assertEqual(item["cost"], int(float(record["actual_cost_ticks"])))
            self.assertEqual(item["unique_od"], int(float(record["Q_acquired"])))

    def test_network_generator(self):
        from generate_networks import generate_example
        instance = generate_example(7, 15)
        self.assertEqual(instance["n"], 15)
        self.assertEqual(len(instance["ground_truth"]), 15)

if __name__ == "__main__":
    unittest.main()
