import json
import random
import argparse
import os
import pandas as pd
from collections import defaultdict
import glob


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["sample", "annotate"],
        default="annotate",
        help="Sample to CSV or start UI",
    )
    parser.add_argument(
        "--input_dir",
        default="/Users/maximedassen/Documents/PhD/Papers/Paper 2/Anonymous/JudgementalReader/data/filtered_data",
    )
    parser.add_argument(
        "--csv_file", default="annotation_sample.csv", help="CSV to save or load"
    )
    parser.add_argument("--sample_size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return args


class Color:
    PURPLE, CYAN, BLUE, GREEN, YELLOW, RED, BOLD, END = (
        "\033[95m",
        "\033[96m",
        "\033[94m",
        "\033[92m",
        "\033[93m",
        "\033[91m",
        "\033[1m",
        "\033[0m",
    )
    BG_MAGENTA, BG_BLUE, BG_YELLOW = "\033[45m", "\033[44m", "\033[43m"


def get_behavioral_cohort(item):
    pk, ctx = item.get("parametric_knowledge_label", ""), item.get("context_label", "")
    scen, is_corr = (
        item.get("scenario_type", ""),
        item.get("labeled_info", {}).get("is_correct", False),
    )

    if pk == "PK_correct" and ctx == "Ctx_correct" and not is_corr:
        return "SCR"
    if (
        pk == "PK_correct"
        and ctx == "Ctx_incorrect"
        and not is_corr
        and ("Coherent" in scen or "Narrative" in scen)
    ):
        return "NHR"
    if pk == "PK_incorrect" and ctx == "Ctx_incorrect" and is_corr:
        return "PKR"
    return "Control"


def load_stratified_sample(input_dir, sample_size, seed):
    """Deterministic stratified sampling across all models and cohorts."""
    random.seed(seed)
    files = sorted(glob.glob(os.path.join(input_dir, "*.jsonl")))
    pool = defaultdict(list)

    print(f"Scanning {len(files)} files for deterministic stratification...")
    for f_path in files:
        model_name = os.path.basename(f_path).split("_labeled")[0]
        with open(f_path, "r") as f:
            for line in f:
                item = json.loads(line)
                if "scenario_type" not in item:
                    continue
                cohort = get_behavioral_cohort(item)
                key = (model_name, cohort)
                pool[key].append(item)

    unique_keys = sorted(list(pool.keys()))
    if not unique_keys:
        return []

    per_strata = sample_size // len(unique_keys)
    final_sample = []

    for key in unique_keys:
        count = min(len(pool[key]), per_strata)
        final_sample.extend(random.sample(pool[key], count))

    if len(final_sample) < sample_size:
        remainder_pool = []
        for key in unique_keys:
            remainder_pool.extend([i for i in pool[key] if i not in final_sample])
        final_sample.extend(
            random.sample(
                remainder_pool,
                min(sample_size - len(final_sample), len(remainder_pool)),
            )
        )

    random.shuffle(final_sample)
    return final_sample


def main():
    args = config()

    # Mode: sample
    if args.mode == "sample":
        sample = load_stratified_sample(args.input_dir, args.sample_size, args.seed)
        rows = []
        for item in sample:
            rows.append(
                {
                    "id": item["id"],
                    "model": item.get("model", "unknown").split("/")[-1],
                    "cohort": get_behavioral_cohort(item),
                    "prompt": item.get("prompt", "N/A"),
                    "ground_truth": item["question_details"]["correct_answer"],
                    "distractor": item["question_details"].get(
                        "distractor_answer", "N/A"
                    ),
                    "original_response": item["original_response"],
                    "human_correct": "",
                    "llm_judge_correct": item["labeled_info"]["is_correct"],
                }
            )
        pd.DataFrame(rows).to_csv(args.csv_file, index=False)
        print(
            f"✅ Deterministic sample of {len(rows)} (Seed {args.seed}) saved to {args.csv_file}"
        )
        return

    # Mode: annotate
    name = (
        input(f"{Color.BOLD}Enter Annotator Name: {Color.END}")
        .strip()
        .replace(" ", "_")
    )
    output_file = f"labels_{name}.csv"

    if os.path.exists(args.csv_file):
        print(f"Loading sample from {args.csv_file}...")
        sample_df = pd.read_csv(args.csv_file)
        sample = sample_df.to_dict("records")
    else:
        print("No CSV found. Sampling directly from raw data...")
        raw_sample = load_stratified_sample(args.input_dir, args.sample_size, args.seed)
        sample = []
        for item in raw_sample:
            sample.append(
                {
                    "id": item["id"],
                    "model": item.get("model", "unknown").split("/")[-1],
                    "cohort": get_behavioral_cohort(item),
                    "prompt": item.get("prompt", "N/A"),
                    "ground_truth": item["question_details"]["correct_answer"],
                    "distractor": item["question_details"].get(
                        "distractor_answer", "N/A"
                    ),
                    "original_response": item["original_response"],
                    "llm_judge_correct": item["labeled_info"]["is_correct"],
                }
            )

    results = []
    completed_ids = set()
    if os.path.exists(output_file):
        results = pd.read_csv(output_file).to_dict("records")
        completed_ids = {str(r["id"]) for r in results}
        print(f"Resuming progress: {len(completed_ids)}/{len(sample)} done.")

    for i, item in enumerate(sample):
        if str(item["id"]) in completed_ids:
            continue

        os.system("clear" if os.name == "posix" else "cls")
        print(
            f"{Color.BG_MAGENTA}{Color.BOLD} ITEM {i + 1}/{len(sample)} {Color.END} | {Color.CYAN}{item['cohort']}{Color.END} | {item['model']}"
        )
        print(
            f"{Color.BOLD}{Color.GREEN}GROUND TRUTH:{Color.END} {{{item['ground_truth']}}}"
        )
        print(f"{Color.BOLD}{Color.RED}DISTRACTOR:  {Color.END} [{item['distractor']}]")
        print("-" * 60)

        # Display prompt
        print(f"{Color.BOLD}{Color.YELLOW}PROMPT / CONTEXT:{Color.END}")
        print(f"{item['prompt']}")
        print("-" * 60)

        # Display response given by model
        print(f"{Color.BOLD}{Color.BG_BLUE} MODEL RESPONSE: {Color.END}")
        print(f"{Color.BLUE}{item['original_response']}{Color.END}")
        print("-" * 60)

        is_corr = input(f"{Color.BOLD}Is response CORRECT? (t/f): {Color.END}").lower()

        results.append({**item, "human_correct": (is_corr == "t")})

        pd.DataFrame(results).to_csv(output_file, index=False)
        if input("\nPress Enter for next or 'q' to quit: ").lower() == "q":
            break

    print(f"Annotation saved to {output_file}")


if __name__ == "__main__":
    main()
