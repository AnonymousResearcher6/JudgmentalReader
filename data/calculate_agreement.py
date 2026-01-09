import pandas as pd
import argparse
from sklearn.metrics import cohen_kappa_score


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--human_files", nargs="+", required=True)
    args = parser.parse_args()
    return args


def main():
    args = config()

    df1, df2 = pd.read_csv(args.human_files[0]), pd.read_csv(args.human_files[1])
    merged = pd.merge(df1, df2, on="id", suffixes=("_h1", "_h2"))

    report = []

    print("\n--- 1. INTER-RATER AGREEMENT (Human vs Human) ---")
    hh_list = []
    for ds in sorted(merged["dataset_h1"].unique()):
        sub = merged[merged["dataset_h1"] == ds]
        k = cohen_kappa_score(sub["human_correct_h1"], sub["human_correct_h2"])
        hh_list.append(k)
        print(f"{ds:8} | Kappa: {k:.3f} (n={len(sub)})")
        report.append({"Metric": "Inter-Rater", "Dataset": ds, "Kappa": round(k, 3)})

    print("\n--- 2. VALIDATION AGREEMENT (Consensus vs LLM) ---")
    consensus = merged[merged["human_correct_h1"] == merged["human_correct_h2"]].copy()
    hl_list = []
    for ds in sorted(consensus["dataset_h1"].unique()):
        sub = consensus[consensus["dataset_h1"] == ds]
        k = cohen_kappa_score(sub["human_correct_h1"], sub["llm_judge_correct_h1"])
        hl_list.append(k)
        print(f"{ds:8} | Kappa: {k:.3f} (n={len(sub)})")
        report.append({"Metric": "Validation", "Dataset": ds, "Kappa": round(k, 3)})

    print(f"\nGlobal Mean Validation Kappa: {sum(hl_list) / len(hl_list):.3f}")
    report.append(
        {
            "Metric": "Validation",
            "Dataset": "MEAN",
            "Kappa": round(sum(hl_list) / len(hl_list), 3),
        }
    )

    pd.DataFrame(report).to_csv("agreement_report.csv", index=False)
    print("\n✅ Report saved to agreement_report.csv")


if __name__ == "__main__":
    main()
