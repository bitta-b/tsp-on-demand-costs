# Solving TSP with On-Demand Travel Costs

This repository contains Python implementations, simulation results, and verification tools developed while investigating Question 7 proposed by Dr. Mehdi Nourinejad.

## Research question

How can we solve the Traveling Salesman Problem (TSP) when travel costs are unknown in advance and must be obtained individually from an API?

The central trade-off is between **tour quality** and **information acquisition**. A search may produce a good tour but require too many distinct origin–destination (OD) costs to be practical.

## Approaches investigated

The experiments explore configurations and adaptations of **established methods**, rather than newly invented algorithms:

- **Cheapest Insertion + Relocation:** Construct an initial tour and evaluate relocation moves using actual directed travel costs.
- **Geographically restricted search:** Limit candidate moves to reduce the number of distinct OD costs acquired.
- **Multistart and Or-opt:** Explore alternative starting tours and short-block local-search moves.
- **Ant Colony System (ACS) and DR-core:** Investigate adaptations of published TSP/ATSP methods; DR-core is not a reproduction of the full published algorithm.

## Main findings

- Requesting costs only when needed does **not** automatically save information: an unrestricted search may still acquire the entire cost matrix.
- Geographical restrictions can reduce acquisition substantially, but fewer acquired costs do not guarantee a good tour.
- Additional starting tours and move types may improve tour quality at the cost of more information and computation.
- **No method was established as universally superior.** The choice depends on tour-quality requirements and the information budget.

The acquisition metric counts **distinct directed OD costs**, not HTTP requests, latency, or API charges.

## Repository contents

| Path | Purpose |
| --- | --- |
| `code/` | Algorithm implementations, synthetic-network generation, and analysis/verification scripts |
| `results/` | Saved experiment records, comparison tables, and charts |
| `tests/` | Automated correctness tests |
| `examples/` | Sample instances and historical inputs |
| `requirements.txt` | Python dependencies |
| `RUN.txt` | Setup, demonstration, and verification commands |

## Running and checking the work

See [RUN.txt](RUN.txt) for instructions. It includes setup commands, automated tests, and verification of saved results.

**Reproducibility boundary:** The default `run_experiments.py` settings demonstrate the implementations; they do **not** recreate the original historical benchmark. Selected historical runs can be replayed using the provided verification tools. Historical ACS and DR-core completion/timeout records are audited separately, rather than fully replayed.

## Important limitations

- Experiments used **synthetic directed road networks** and an **offline cost oracle**, not a live routing API.
- The information metric measures acquired distinct directed OD costs, not actual API billing, HTTP request count, or latency.
- Exact optimality was certified only for instances with **15 or 16 destinations**.
- ACS and DR-core are **adaptations**, not complete reproductions of their respective published algorithms.
- The repository is an analysis and reproducibility companion, **not** a production-ready API-based TSP solver.

## Related literature

- Dorigo, M., & Gambardella, L. M. (1997). *Ant colony system: A cooperative learning approach to the traveling salesman problem.* [DOI: 10.1109/4235.585892](https://doi.org/10.1109/4235.585892).
- Rego, C., Gamboa, D., & Glover, F. (2016). *Doubly-rooted stem-and-cycle ejection chain algorithm for the asymmetric traveling salesman problem.* [DOI: 10.1002/net.21676](https://doi.org/10.1002/net.21676).
- Rashidi, Nourinejad, & Roorda (2025, preprint). *Sampling the cost matrix from online APIs in the travelling salesman problem.* [SSRN preprint](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5643071). This sampling-and-estimation approach is discussed as a related alternative, **not** benchmarked against the implementations here.
