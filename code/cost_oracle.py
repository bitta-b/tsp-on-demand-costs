from __future__ import annotations
import json
import hashlib
import time
import math
from collections import Counter
from dataclasses import dataclass, field

@dataclass(frozen=True)
class PublicProblem:
    node_ids: tuple[int, ...]
    coordinates: tuple[tuple[float, float], ...]
    depot: int
    scenario: str
    cost_unit: str = "integer_microseconds"

    def __post_init__(self):
        if len(self.node_ids) < 3 or len(set(self.node_ids)) != len(self.node_ids):
            raise ValueError("At least three distinct nodes required")
        if any(type(i) is not int for i in self.node_ids) or type(self.depot) is not int:
            raise ValueError("Node IDs must be integers, not booleans")
        if self.depot not in self.node_ids or len(self.coordinates) != len(self.node_ids):
            raise ValueError("Invalid depot/coordinate alignment")
        if any(len(p) != 2 or any(type(x) not in (int, float) or not math.isfinite(x)
                                 for x in p) for p in self.coordinates):
            raise ValueError("Finite planar coordinates required")
        if not self.scenario or self.cost_unit != "integer_microseconds":
            raise ValueError("Explicit scenario and cost unit required")

    def public_json(self):
        return {"node_ids": list(self.node_ids), "coordinates": [list(p) for p in self.coordinates],
                "depot": self.depot, "scenario": self.scenario, "cost_unit": self.cost_unit}

@dataclass(frozen=True)
class AlgorithmConfig:
    algorithm_id: str
    source_id: str
    source_version: str
    mode: str
    fidelity: str
    seed: int
    initialization_policy: str
    candidate_policy: str
    stopping_policy: str
    parameters: dict = field(default_factory=dict)
    adaptations: tuple[str, ...] = ()

    def __post_init__(self):
        if not all((self.algorithm_id, self.source_id, self.source_version,
                    self.initialization_policy, self.candidate_policy, self.stopping_policy)):
            raise ValueError("All method/provenance/policy identifiers required")
        if self.mode not in ("P", "G") or self.fidelity not in ("ORIGINAL", "ADAPTATION", "UNRESOLVED"):
            raise ValueError("Invalid fidelity/mode label")
        if type(self.seed) is not int:
            raise ValueError("Integer random seed required")
        if self.mode == "G" and (self.fidelity != "ADAPTATION" or not self.adaptations):
            raise ValueError("Mode G must disclose its adaptations")
        if self.fidelity == "ORIGINAL" and self.adaptations:
            raise ValueError("An adapted implementation cannot claim ORIGINAL")
        json.dumps(self.parameters, allow_nan=False)

STAGES = frozenset({"preprocessing", "initialization", "candidate_generation", "construction",
                    "search", "fallback", "final_verification"})

class OracleError(Exception):
    def __init__(self, code, message, retryable=False):
        self.code, self.retryable = code, retryable
        super().__init__(message)

class OfflineDataset:

    def __init__(self, node_ids, matrix, label="fixture_t0", transient_failures=None):
        self.node_ids = tuple(node_ids)
        if len(set(self.node_ids)) != len(self.node_ids) or any(type(i) is not int for i in self.node_ids):
            raise ValueError("Distinct integer IDs required")
        n = len(self.node_ids)
        self.__matrix = tuple(tuple(row) for row in matrix)
        if len(self.__matrix) != n or any(len(row) != n for row in self.__matrix):
            raise ValueError("Square matrix required")
        for i, row in enumerate(self.__matrix):
            for j, c in enumerate(row):
                if i != j and c is not None and (type(c) is not int or c <= 0):
                    raise ValueError("Positive integer costs or explicit unreachable None required")
        self.__index = {v: k for k, v in enumerate(self.node_ids)}
        raw = json.dumps([self.node_ids, self.__matrix], separators=(",", ":")).encode()
        self.scenario = label + ":" + hashlib.sha256(raw).hexdigest()
        self.__remaining = dict(transient_failures or {})
        if any(type(v) is not int or v < 0 for v in self.__remaining.values()):
            raise ValueError("Nonnegative transient-failure counts required")
        self.attempts = []

    def fetch(self, scenario, i, j):
        self.attempts.append((scenario, i, j))
        if scenario != self.scenario:
            raise OracleError("SCENARIO_MISMATCH", "Frozen scenario mismatch")
        if i == j or i not in self.__index or j not in self.__index:
            raise OracleError("INVALID_OD", "Invalid directed OD")
        if self.__remaining.get((i, j), 0):
            self.__remaining[(i, j)] -= 1
            raise OracleError("TRANSIENT", "Injected offline failure", retryable=True)
        value = self.__matrix[self.__index[i]][self.__index[j]]
        if value is None:
            raise OracleError("UNREACHABLE", "No directed route in this frozen scenario")
        return value

    def evaluator_cost(self, closed_tour):

        values = [self.__matrix[self.__index[i]][self.__index[j]] for i, j in zip(closed_tour, closed_tour[1:])]
        if any(v is None for v in values):
            raise OracleError("UNREACHABLE", "Tour contains unreachable OD")
        return sum(values)

class OracleLedger:

    def __init__(self, dataset, run_id, arm_id, allowed_pairs=None):
        self.__dataset = dataset
        self.scenario = dataset.scenario
        self.nodes = frozenset(dataset.node_ids)
        self.run_id, self.arm_id = run_id, arm_id
        self.__cache = {}
        self.__failures = {}
        self.__requested = set()
        self.__allowed = None if allowed_pairs is None else frozenset(allowed_pairs)
        self.events = []
        self.oracle_cpu_s = self.oracle_wall_s = 0.0

    def query(self, i, j, stage, context=None):
        event = {"schema_version": 1, "sequence": len(self.events), "run_id": self.run_id,
                 "arm_id": self.arm_id, "scenario": self.scenario, "origin": i, "destination": j,
                 "stage": stage, "context": context or {}, "outcome": None, "cost_ticks": None,
                 "retryable": False, "acquisition_index": None}
        self.events.append(event)
        if (type(i) is not int or type(j) is not int or i == j or i not in self.nodes or j not in self.nodes
                or not isinstance(stage, str) or stage not in STAGES
                or (context is not None and not isinstance(context, dict))):
            event.update(outcome="invalid", error_code="INVALID_REQUEST")
            raise OracleError("INVALID_REQUEST", "Valid directed OD, stage and context required")
        try:
            context_bytes = json.dumps(context or {}, allow_nan=False).encode()
            if len(context_bytes) > 4096: raise ValueError("Context too large")
            event["context"] = json.loads(context_bytes)
        except (ValueError, TypeError):
            event.update(outcome="invalid", error_code="INVALID_CONTEXT", context={})
            raise OracleError("INVALID_CONTEXT", "Small JSON context required")
        key = (self.scenario, i, j)
        self.__requested.add(key)
        if self.__allowed is not None and (i, j) not in self.__allowed:
            event.update(outcome="denied", error_code="POLICY_DENIED")
            raise OracleError("POLICY_DENIED", "OD excluded by explicit test access policy")
        if key in self.__cache:
            event.update(outcome="hit", cost_ticks=self.__cache[key])
            return self.__cache[key]
        if key in self.__failures:
            code = self.__failures[key]
            event.update(outcome="negative_hit", error_code=code)
            raise OracleError(code, "Previously recorded permanent offline failure")
        cpu, wall = time.process_time(), time.perf_counter()
        try:
            value = self.__dataset.fetch(self.scenario, i, j)
            if type(value) is not int or value <= 0:
                raise OracleError("INVALID_COST", "Positive integer actual cost required")
        except OracleError as error:
            event.update(outcome="miss_failure", error_code=error.code, retryable=error.retryable)
            if not error.retryable: self.__failures[key] = error.code
            raise
        except Exception as error:

            event.update(outcome="miss_failure", error_code="BACKEND_ERROR", retryable=True)
            raise OracleError("BACKEND_ERROR", str(error), retryable=True) from error
        else:
            self.__cache[key] = value
            event.update(outcome="miss_success", cost_ticks=value, acquisition_index=len(self.__cache)-1)
            return value
        finally:
            self.oracle_cpu_s += time.process_time()-cpu
            self.oracle_wall_s += time.perf_counter()-wall

    def metrics(self):
        counts = Counter(e["outcome"] for e in self.events)
        stages = Counter(e["stage"] for e in self.events if e["outcome"] == "miss_success")
        return {"logical_calls": len(self.events), "distinct_directed_requested": len(self.__requested),
                "distinct_directed_acquired": len(self.__cache),
                "cache_hits": counts["hit"]+counts["negative_hit"], "positive_cache_hits": counts["hit"],
                "negative_cache_hits": counts["negative_hit"],
                "cache_misses": counts["miss_success"]+counts["miss_failure"],
                "successful_misses": counts["miss_success"], "failed_misses": counts["miss_failure"],
                "invalid_calls": counts["invalid"], "denied_calls": counts["denied"],
                "possible_directed_OD": len(self.nodes)*(len(self.nodes)-1),
                "acquired_percent": 100*len(self.__cache)/(len(self.nodes)*(len(self.nodes)-1)),
                "first_acquisitions_by_stage": dict(sorted(stages.items()))}

def canonical_trace(events):
    return b"".join((json.dumps(e, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n").encode() for e in events)

class CostOracle:
    def __init__(self, matrix, name="example"):
        self.dataset = OfflineDataset(tuple(range(len(matrix))), matrix, name)
        self.ledger = OracleLedger(self.dataset, name, name)
        self.events = []

    def cost(self, origin, destination, *, stage="search", context=None):
        return self.ledger.query(origin, destination, stage, context)

    def record(self, event_type, details):
        self.events.append((event_type, details))

    def verify(self, tour):
        if len(tour) != len(self.dataset.node_ids) + 1 or tour[0] != tour[-1]:
            raise ValueError("Tour is not closed")
        if set(tour[:-1]) != set(self.dataset.node_ids) or len(set(tour[:-1])) != len(self.dataset.node_ids):
            raise ValueError("Tour does not visit every city exactly once")
        for a, b in zip(tour, tour[1:]):
            self.cost(a, b, stage="final_verification")
        return self.dataset.evaluator_cost(tour)

    def metrics(self):
        return self.ledger.metrics()
