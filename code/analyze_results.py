import csv
from pathlib import Path

def main():
    file = Path(__file__).parents[1] / "results" / "paired_summary.csv"
    with file.open(newline="", encoding="utf8") as f:
        for row in csv.DictReader(f):
            if row["reference"] == "C" and row["target"] in ("M", "MO", "ACS", "DR"):
                print("n=", row["n"], "method=", row["target"],
                      "mean change=", row["mean_delta_cost_pct"],
                      "extra Q pp=", row["mean_extra_Q_pp"])

if __name__ == "__main__":
    main()
