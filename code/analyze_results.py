import csv
from pathlib import Path
from verify_results import check_results, read_csv

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results"
METHODS = ["Cheapest insertion + relocation", "Multistart cheapest insertion + relocation", "Multistart cheapest insertion + Or-opt"]
LABELS = ["Single-start relocation", "Multistart relocation", "Multistart + Or-opt"]


def load():
    measured, _ = check_results()
    with (DATA / "comparison.csv").open(newline="", encoding="utf8") as file:
        rows = list(csv.DictReader(file))
    original = {(int(r["cities"]), r["method"]): r for r in rows}
    original.update(measured)
    return original


def value(row, key):
    return float(row[key]) if row.get(key) else float("nan")


def show_table(d):
    print("Historical comparison: 200 instances, 40 for each number of cities")
    print("Travel cost: lower is better. OD: fewer retrieved costs is better.")
    print("Cost difference: negative means an improvement over single-start relocation.\n")
    print(f"{'Cities':>6} {'Approach':<25} {'OD retrieved':>12} {'Cost vs single':>16} {'Median CPU':>12}")
    for n in (15, 16, 20, 50, 100):
        for method, label in zip(METHODS, LABELS):
            r = d[(n, method)]
            print(f"{n:>6} {label:<25} {value(r, 'mean_od_percent'):>11.1f}% "
                  f"{value(r, 'cost_change_vs_baseline_percent'):>+14.1f}% "
                  f"{value(r, 'median_cpu_seconds'):>10.3f} s")
        print()
    print("Across 80 smaller instances (15 and 16 cities), certified mean optimality gaps:")
    for method, label in zip(METHODS, LABELS):
        gap = sum(value(d[(n, method)], "exact_gap_percent") for n in (15,16)) / 2
        print(f"  {label:<25} {gap:.3f}%")
    with (DATA / "all_runs.csv").open(newline="", encoding="utf8") as file:
        runs = list(csv.DictReader(file))
    print(f"\nAll raw records preserved: {len(runs)} across {len(set(r['instance_id'] for r in runs))} instances.")
    summaries = list(d.values())
    acs = [r for r in summaries if r["method"] == "Ant colony system"]
    dr = [r for r in summaries if r["method"] == "Doubly-rooted stem-and-cycle"]
    def total(rows, key):
        return sum(int(r[key]) for r in rows)
    print(f"ACS: {total(acs, 'completed')} completed, {total(acs, 'timeouts')} timed out, "
          f"{total(acs, 'not_run')} not run; all completed ACS runs retrieved 100% of the matrix.")
    print(f"DR: {total(dr, 'completed')}/200 completed; at 100 cities, "
          f"{value(d[(100,'Doubly-rooted stem-and-cycle')], 'mean_od_percent'):.1f}% OD and "
          f"{value(d[(100,'Doubly-rooted stem-and-cycle')], 'cost_change_vs_baseline_percent'):+.1f}% cost vs single-start.")
    print("ACS success-only quality must not be generalized.")
    print("No exact optimum was certified for 20, 50 or 100 cities.")
    print("CPU results shown above come from the same original baseline study.")
    print("CPU numbers for ACS/DR were measured separately and are not directly comparable.")
    print("Actual HTTP calls, monetary billing and consistent memory measurements are unavailable.")


def make_charts(d):
    try:
        import matplotlib.pyplot as plt
    except (ImportError, OSError) as error:
        raise SystemExit("Cannot load Matplotlib. Use the project virtual environment and install the versions in requirements.txt") from error
    charts = DATA / "charts"
    charts.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(9.4, 5.8))
    for method, label in zip(METHODS, LABELS):
        row = d[(100, method)]
        q = value(row, "mean_od_percent")
        gain = -value(row, "cost_change_vs_baseline_percent")
        ax.scatter([q], [gain], s=115, zorder=3)
        ax.annotate((f"{label}\n{q:.1f}% OD  |  baseline cost" if method == METHODS[0] else
                     f"{label}\n{q:.1f}% OD  |  {gain:.1f}% lower cost"), (q, gain),
                    xytext=((9, -38) if method == METHODS[1] else (9, 12)), textcoords="offset points",
                    ha="left", fontsize=10)
    dr = d[(100, "Doubly-rooted stem-and-cycle")]
    dq = value(dr, "mean_od_percent")
    dgain = -value(dr, "cost_change_vs_baseline_percent")
    ax.scatter([dq], [dgain], s=115, zorder=3)
    ax.annotate(f"Stem-and-cycle (38/40 completed)\n{dq:.1f}% OD  |  {-dgain:.1f}% higher cost",
                (dq, dgain), xytext=(-10, 12), textcoords="offset points", ha="right", fontsize=10)
    ax.axhline(0, linewidth=.8, alpha=.35)
    ax.set_xlim(14, 51);ax.set_ylim(-24, 20)
    ax.set_xlabel("Actual directed cost entries retrieved (% of matrix)")
    ax.set_ylabel("Mean paired relative tour-cost reduction vs single-start (%)")
    ax.set_title("100 cities: Tour quality vs retrieved costs")
    ax.grid(alpha=.17)
    fig.text(.5,.02,"Same inputs: 40 for C/M/MO; 38 completed for DR. ACS had no completed 100-city runs.",ha="center",fontsize=9)
    fig.tight_layout(rect=[0,.05,1,1])
    fig.savefig(charts / "cost_vs_queries.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.4, 5.8))
    gaps=[sum(value(d[(n,method)],"exact_gap_percent") for n in (15,16))/2 for method in METHODS]
    bars = ax.bar(LABELS,gaps,width=.55)
    for b,g in zip(bars,gaps):
        ax.text(b.get_x()+b.get_width()/2,g+.45, f"{g:.2f}%",ha="center",fontsize=12)
    ax.set_ylim(0,max(gaps)*1.2)
    ax.set_ylabel("Mean gap above certified optimum (%)")
    ax.set_title("80 small instances: How close are the tours to optimal?")
    ax.grid(axis='y',alpha=.17)
    ax.set_axisbelow(True)
    records = read_csv("all_runs.csv")
    max_gaps = [max(float(r["gap_exact_pct"]) for r in records if
                r["method"] == code and r["role"] == "historical_control" and
                r["status"] == "VERIFIED" and int(r["n"]) in (15, 16))
                for code in ("C", "M", "MO")]
    fig.text(.5,.02, f"80 certified small instances. Worst observed gaps: C {max_gaps[0]:.1f}%, "
             f"M {max_gaps[1]:.1f}%, MO {max_gaps[2]:.1f}%.", ha="center",fontsize=9)
    fig.tight_layout(rect=[0,.05,1,1])
    fig.savefig(charts / "optimality_gap.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.4, 5.8))
    values = [value(d[(100,method)],"median_cpu_seconds") for method in METHODS]
    bars = ax.bar(LABELS,values,width=.55)
    for b,t in zip(bars,values):
        ax.text(b.get_x()+b.get_width()/2,t+.055,f"{t:.2f} s",ha="center",fontsize=12)
    ax.set_ylim(0,max(values)*1.22)
    ax.set_ylabel("Median algorithm CPU time (seconds)")
    ax.set_title("100 cities: Median algorithm CPU time")
    ax.grid(axis='y',alpha=.17)
    ax.set_axisbelow(True)
    fig.text(.5,.02,"Same historical execution environment; 40 instances for each method.",ha="center",fontsize=9)
    fig.tight_layout(rect=[0,.05,1,1])
    fig.savefig(charts / "computation_time.png", dpi=170)
    plt.close(fig)
    print("\nThree charts saved in results/charts.")


if __name__ == "__main__":
    data = load()
    show_table(data)
    make_charts(data)
