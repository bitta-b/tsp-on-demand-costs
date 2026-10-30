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

    def test_network_generator(self):
        from generate_networks import generate_example
        instance = generate_example(7, 15)
        self.assertEqual(instance["n"], 15)
        self.assertEqual(len(instance["ground_truth"]), 15)

if __name__ == "__main__":
    unittest.main()
