import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import re
import json
import os
from tqdm import tqdm
from scipy.stats import norm, mannwhitneyu, chi2_contingency, fisher_exact
from statsmodels.stats.multitest import multipletests

# Optional: Statsmodels for rigorous CIs
try:
    from statsmodels.stats.proportion import confint_proportions_2indep

    HAS_STATSMODELS_CI = True
except Exception:
    HAS_STATSMODELS_CI = False

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
# UPDATE THIS PATH TO YOUR DATA DIRECTORY
DATA_DIR = "/Users/maximedassen/Documents/PhD/Papers/Paper 2/code/knowledge-conflicts/data/filtered_data_logprobs"

# Visual Settings
sns.set_theme(
    style="whitegrid",
    font_scale=1.1,
    rc={"font.family": "serif", "axes.titlesize": 14, "axes.labelsize": 12},
)

PALETTE = {"Base": "#8da0cb", "Instruct": "#fc8d62"}
PALETTE_MECH = {"Teachability": "#3498db", "Stickiness": "#e74c3c"}

# Statistics Config
ALPHA = 0.05
CI_LEVEL = 0.95
DATASET_ORDER = ["NQ (0-hop)", "SQuAD (1-hop)", "MuSiQue (N-hop)"]
SORTED_CORE_MODELS = []  # Will be populated dynamically
SORTED_DISPLAY_NAMES = []  # Will be populated dynamically


# =============================================================================
# 2. UTILITY FUNCTIONS
# =============================================================================
def get_family(clean_name):
    if "Llama" in clean_name:
        return "Llama"
    if "Qwen" in clean_name:
        return "Qwen"
    if "gemma" in clean_name.lower():
        return "Gemma"
    if "Mistral" in clean_name:
        return "Mistral"
    if "OLMo" in clean_name:
        return "OLMo"
    return "Other"


def get_size_param(clean_name):
    match = re.search(r"(\d+\.?\d*)[Bb]", clean_name)
    if match:
        return float(match.group(1))
    if "Nemo" in clean_name:
        return 12.0
    if "OLMo" in clean_name:
        return 7.0
    return 0.0


def get_type(clean_name):
    return "Instruct" if "Instruct" in clean_name else "Base"


def get_core_name(clean_name):
    return clean_name.replace("-Instruct", "").replace("-Base", "").strip("-")


def length_bucket(n_words: int) -> str:
    """Buckets context length for ablation study."""
    if n_words < 50:
        return "Very Short (<50)"
    if n_words < 150:
        return "Short (<150)"
    if n_words < 400:
        return "Medium (<400)"
    return "Long (400+)"


def p_to_stars(p: float) -> str:
    if p is None or np.isnan(p):
        return ""
    if p < 0.0001:
        return "***"  # Strict threshold for Chi2
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def bh_adjust(pvals):
    """Benjamini-Hochberg FDR correction."""
    pvals = np.array(pvals, dtype=float)
    mask = ~np.isnan(pvals)
    adj = np.full_like(pvals, np.nan, dtype=float)
    if mask.sum() == 0:
        return adj.tolist()
    _, p_adj, _, _ = multipletests(pvals[mask], alpha=ALPHA, method="fdr_bh")
    adj[mask] = p_adj
    return adj.tolist()


def diff_proportion_stats(s1, n1, s2, n2, method="wald"):
    """Calculates difference in proportions, CI, and p-value."""
    if n1 == 0 or n2 == 0:
        return np.nan, (np.nan, np.nan), np.nan

    p1 = s1 / n1
    p2 = s2 / n2
    diff = p2 - p1

    p_pool = (s1 + s2) / (n1 + n2)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))

    if se_pool == 0:
        p_val = 1.0
    else:
        z = diff / se_pool
        p_val = 2 * (1 - norm.cdf(abs(z)))

    ci_low, ci_high = np.nan, np.nan
    ci_alpha = 1.0 - CI_LEVEL

    if HAS_STATSMODELS_CI:
        try:
            ci_low, ci_high = confint_proportions_2indep(
                s1, n1, s2, n2, method=method, compare="diff", alpha=ci_alpha
            )
        except:
            pass

    if np.isnan(ci_low):
        se_unpooled = np.sqrt((p1 * (1 - p1)) / n1 + (p2 * (1 - p2)) / n2)
        zc = norm.ppf(1 - ci_alpha / 2)
        ci_low, ci_high = diff - zc * se_unpooled, diff + zc * se_unpooled

    return diff, (ci_low, ci_high), p_val


def add_significance_bars(ax, data, x, y, hue):
    """Adds statistical significance annotations to bar plots using Mann-Whitney U."""
    datasets = data[x].unique()
    for i, ds in enumerate(datasets):
        subset = data[data[x] == ds]
        base = subset[subset[hue] == "Base"][y].values
        inst = subset[subset[hue] == "Instruct"][y].values

        if len(base) > 0 and len(inst) > 0:
            stat, p = mannwhitneyu(base, inst, alternative="two-sided")
            if p < 0.05:
                bar_max = max(base.mean(), inst.mean())
                offset = abs(bar_max) * 0.1 if bar_max != 0 else 0.05
                label = p_to_stars(p)
                ax.text(
                    i,
                    bar_max + offset,
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=12,
                    fontweight="bold",
                    color="black",
                )


# =============================================================================
# 3. DATA LOADING & PROCESSING
# =============================================================================
def load_and_process_data(directory):
    print(f"🔄 Loading data from {directory}...")
    data = []

    try:
        files = [f for f in os.listdir(directory) if f.endswith("_logprobs.jsonl")]
    except FileNotFoundError:
        print(f"❌ Error: Directory '{directory}' not found.")
        sys.exit(1)

    for filename in tqdm(files, desc="Processing Files"):
        # Name Cleaning
        raw_model = filename.replace("_logprobs.jsonl", "")
        clean_name = (
            raw_model.replace("models--", "")
            .replace("google--", "")
            .replace("meta-llama--", "")
            .replace("mistralai--", "")
            .replace("Qwen--", "")
            .replace("allenai--", "")
        )
        clean_name = (
            clean_name.replace("Meta-Llama-", "Llama-")
            .replace("-hf", "")
            .replace("-It", "-Instruct")
            .replace("-it", "-Instruct")
        )
        clean_name = clean_name.replace(
            "Mistral-Nemo-Base-2407", "Mistral-Nemo-Base"
        ).replace("Mistral-Nemo-Instruct-2407", "Mistral-Nemo-Instruct")

        file_path = os.path.join(directory, filename)

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if "scenario_type" not in entry or "final_label" not in entry:
                        continue

                    # Dataset ID
                    source_id = entry.get("source_id", "")
                    if source_id.startswith("juice"):
                        dataset = "NQ (0-hop)"
                    elif source_id.startswith("musique"):
                        dataset = "MuSiQue (N-hop)"
                    elif source_id.startswith("squad"):
                        dataset = "SQuAD (1-hop)"
                    else:
                        continue

                    # Filter for High Confidence Only (0 or 10)
                    pk_score = entry.get("parametric_knowledge_score", -1)
                    if pk_score not in [0, 10]:
                        continue

                    # Scenario Logic
                    scenario = entry.get("scenario_type", "Unknown")
                    if "Coherent" in scenario or "Narrative" in scenario:
                        ctx_class = "Long / Natural"
                    elif "Substitute" in scenario or "Repetitive" in scenario:
                        ctx_class = "Short / Synthetic"
                    else:
                        ctx_class = "Other"

                    # Outcome Logic
                    ans_label = entry.get("answer_label")
                    is_success = 1 if ans_label == "Ans_Success" else 0
                    is_failure = 1 if ans_label == "Ans_Failure" else 0

                    prompt = entry.get("prompt", "")
                    wc = len(prompt.split()) if isinstance(prompt, str) else 0

                    # --- LOGPROB EXTRACTION ---
                    ft_logprob = np.nan
                    ft_prob_gap = np.nan
                    all_logprobs = []
                    all_prob_gaps = []

                    lp_data = entry.get("_logprob_data", {})
                    if lp_data and "logprobs" in lp_data:
                        logs = lp_data["logprobs"]
                        for i, obj in enumerate(logs):
                            if obj is None:
                                continue
                            val = obj.get("val")
                            top_dict = obj.get("top", {})

                            if val is not None:
                                all_logprobs.append(val)

                            if len(top_dict) >= 2:
                                vals = sorted(top_dict.values(), reverse=True)
                                p1, p2 = np.exp(vals[0]), np.exp(vals[1])
                                gap = p1 - p2
                                all_prob_gaps.append(gap)

                                if i == 0:
                                    ft_logprob = val
                                    ft_prob_gap = gap

                    row = {
                        "Display_Name": clean_name,
                        "Dataset_Nice": dataset,
                        "Scenario": scenario,
                        "Context_Class": ctx_class,
                        "PK_Score": pk_score,
                        "Context_Label": entry.get("context_label"),
                        "Answer_Label": ans_label,
                        "Is_Success": is_success,
                        "Is_Failure": is_failure,
                        "Context_Length_Words": wc,
                        "Ctx_Len_Bucket": length_bucket(wc),
                        "FT_LogProb": ft_logprob,
                        "FT_Prob_Gap": ft_prob_gap,
                        "Avg_LogProb": np.mean(all_logprobs)
                        if all_logprobs
                        else np.nan,
                        "FS_Prob_Gap": np.mean(all_prob_gaps)
                        if all_prob_gaps
                        else np.nan,
                    }
                    data.append(row)
                except Exception:
                    continue

    return pd.DataFrame(data)


# Load Data
df = load_and_process_data(DATA_DIR)
if df.empty:
    print("❌ Error: No data found.")
    sys.exit(1)

# Metadata Enrichment
df["Family"] = df["Display_Name"].apply(get_family)
df["Type"] = df["Display_Name"].apply(get_type)
df["Core_Model"] = df["Display_Name"].apply(get_core_name)

# Sorting Helpers
model_meta = (
    df[["Display_Name", "Core_Model", "Family", "Type"]].drop_duplicates().copy()
)
model_meta["Size_Param"] = model_meta["Display_Name"].apply(get_size_param)
model_meta["Type_Rank"] = model_meta["Type"].apply(
    lambda x: 1 if x == "Instruct" else 0
)
model_meta = model_meta.sort_values(by=["Family", "Size_Param", "Type_Rank"])

SORTED_DISPLAY_NAMES = model_meta["Display_Name"].tolist()
SORTED_CORE_MODELS = (
    model_meta[["Family", "Size_Param", "Core_Model"]]
    .sort_values(by=["Family", "Size_Param"])["Core_Model"]
    .unique()
    .tolist()
)
dataset_order = ["NQ (0-hop)", "SQuAD (1-hop)", "MuSiQue (N-hop)"]

# =============================================================================
# 4. METRIC AGGREGATION
# =============================================================================
# Merging for plots (reusing logic)
subset_ffr = df[(df["PK_Score"] == 10) & (df["Context_Label"] == "Ctx_correct")]
ffr_stats = (
    subset_ffr.groupby(["Display_Name", "Dataset_Nice", "Context_Class"])[
        "Is_Failure"
    ].mean()
    * 100
)
ffr_df = ffr_stats.reset_index(name="Synergy Collapse Rate (%)")

subset_por = df[(df["PK_Score"] == 0) & (df["Context_Label"] == "Ctx_incorrect")]
por_stats = (
    subset_por.groupby(["Display_Name", "Dataset_Nice", "Context_Class"])[
        "Is_Success"
    ].mean()
    * 100
)
por_df = por_stats.reset_index(name="Phantom Knowledge Rate (PKR) (%)")

subset_teach = df[(df["PK_Score"] == 0) & (df["Context_Label"] == "Ctx_correct")]
teach_stats = (
    subset_teach.groupby(["Display_Name", "Dataset_Nice", "Context_Class"])[
        "Is_Success"
    ].mean()
    * 100
)
teach_df = teach_stats.reset_index(name="Teachability")

subset_sticky = df[(df["PK_Score"] == 10) & (df["Context_Label"] == "Ctx_incorrect")]
sticky_stats = (
    subset_sticky.groupby(["Display_Name", "Dataset_Nice", "Context_Class"])[
        "Is_Success"
    ].mean()
    * 100
)
stubborn_df = sticky_stats.reset_index(name="Stickiness")

merged_df = ffr_df.merge(
    por_df, on=["Display_Name", "Dataset_Nice", "Context_Class"], how="outer"
)
merged_df = merged_df.merge(
    model_meta[["Display_Name", "Type", "Core_Model", "Family"]], on="Display_Name"
)

mech_df = teach_df.merge(
    stubborn_df, on=["Display_Name", "Dataset_Nice", "Context_Class"], how="outer"
)
mech_df = mech_df.merge(
    model_meta[["Display_Name", "Type", "Core_Model"]], on="Display_Name"
)


# =============================================================================
# 5. PLOTTING FUNCTIONS
# =============================================================================


def plot_split_lines(metric, filename, title, ylabel, split_col="Context_Class"):
    """Generates the X-Crossing Line Plots (Fig 1 & 4)."""
    split_values = ["Short / Synthetic", "Long / Natural"]
    agg = (
        merged_df.groupby(["Dataset_Nice", "Type", split_col])[metric]
        .mean()
        .reset_index()
    )
    agg["Dataset_Nice"] = pd.Categorical(
        agg["Dataset_Nice"], categories=DATASET_ORDER, ordered=True
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for i, val in enumerate(split_values):
        subset = agg[agg[split_col] == val]
        sns.lineplot(
            ax=axes[i],
            data=subset,
            x="Dataset_Nice",
            y=metric,
            hue="Type",
            style="Type",
            markers=True,
            dashes=False,
            linewidth=3,
            markersize=10,
            palette=PALETTE,
        )
        axes[i].set_title(f"{val}", fontsize=14, fontweight="bold")
        axes[i].set_xlabel("Reasoning Complexity")
        axes[i].grid(True, linestyle="--", alpha=0.6)
        if i == 1:
            axes[i].legend(title="Model Type")
        else:
            axes[i].get_legend().remove()

    axes[0].set_ylabel(ylabel)
    plt.suptitle(title, fontsize=18, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close()


def plot_split_lines_with_traces(
    metric_col, filename, title, ylabel, split_col="Context_Class"
):
    """
    Generates the X-Crossing Line Plots (Fig 1 & 4) with individual model traces.
    This visually proves the universal nature of the trend across the fleet.
    """
    split_values = ["Short / Synthetic", "Long / Natural"]

    # Ensure dataset order
    df_plot = merged_df.copy()
    df_plot["Dataset_Nice"] = pd.Categorical(
        df_plot["Dataset_Nice"], categories=DATASET_ORDER, ordered=True
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

    for i, val in enumerate(split_values):
        ax = axes[i]
        subset = df_plot[df_plot[split_col] == val]

        # 1. Plot individual traces (The Spaghetti)
        sns.lineplot(
            ax=ax,
            data=subset,
            x="Dataset_Nice",
            y=metric_col,
            hue="Type",
            units="Display_Name",
            estimator=None,
            alpha=0.15,
            palette=PALETTE,
            legend=False,
            linewidth=1.5,
        )

        # 2. Plot aggregate mean (The Iconic Visual)
        sns.lineplot(
            ax=ax,
            data=subset,
            x="Dataset_Nice",
            y=metric_col,
            hue="Type",
            markers=True,
            markersize=14,
            linewidth=5,
            palette=PALETTE,
            errorbar=None,
            style="Type",
            dashes=False,
        )

        ax.set_title(f"{val}", fontsize=16, fontweight="bold")
        ax.set_xlabel("Reasoning Complexity", fontsize=14)
        ax.grid(True, linestyle="--", alpha=0.4)

        if i == 1:
            ax.legend(title="Model Type", fontsize=12, title_fontsize=13)
        else:
            ax.get_legend().remove()

    axes[0].set_ylabel(ylabel, fontsize=14)
    plt.suptitle(title, fontsize=22, fontweight="bold", y=1.05)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close()


def plot_iconic_with_se_split(
    metric_col, filename, title, ylabel, split_col="Context_Class"
):
    """
    Generates the Iconic Split Plots with fat lines and narrow SE bands.
    Synthetic on left, Narrative on right.
    """
    split_values = ["Short / Synthetic", "Long / Natural"]

    # Ensure dataset order and copy data
    df_plot = merged_df.copy()
    df_plot["Dataset_Nice"] = pd.Categorical(
        df_plot["Dataset_Nice"], categories=DATASET_ORDER, ordered=True
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

    for i, val in enumerate(split_values):
        ax = axes[i]
        subset = df_plot[df_plot[split_col] == val]

        # errorbar=('se', 1) provides the narrower, cleaner uncertainty band
        sns.lineplot(
            ax=ax,
            data=subset,
            x="Dataset_Nice",
            y=metric_col,
            hue="Type",
            palette=PALETTE,
            errorbar=("se", 1),  # Narrow Standard Error band
            err_style="band",  # Shaded band
            linewidth=5,  # Fat color line
            markers=True,
            markersize=14,
            style="Type",
            dashes=False,
        )

        ax.set_title(f"{val}", fontsize=16, fontweight="bold")
        ax.set_xlabel("Reasoning Complexity", fontsize=14)
        ax.grid(True, linestyle="--", alpha=0.4)

        if i == 1:
            ax.legend(title="Model Type", fontsize=12, title_fontsize=13)
        else:
            ax.get_legend().remove()

    axes[0].set_ylabel(ylabel, fontsize=14)
    plt.suptitle(title, fontsize=22, fontweight="bold", y=1.05)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close()


def plot_heatmap(metric, filename, title, cmap):
    """Generates full heatmaps (Fig 2 & 3)."""
    merged_df["Condition"] = (
        merged_df["Dataset_Nice"] + "\n" + merged_df["Context_Class"]
    )
    pivot = merged_df.pivot_table(
        index="Display_Name", columns="Condition", values=metric
    )
    pivot = pivot.reindex(SORTED_DISPLAY_NAMES)
    cols = [
        f"{ds}\n{ctx}"
        for ds in DATASET_ORDER
        for ctx in ["Short / Synthetic", "Long / Natural"]
    ]
    pivot = pivot[cols]
    plt.figure(figsize=(16, 12), dpi=300)
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1f",
        cmap=cmap,
        linewidths=0.5,
        cbar_kws={"label": "Rate (%)"},
    )
    plt.title(title, fontsize=16, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close()


def create_dumbbell_grid(metric_col, title_prefix, filename):
    """Generates Per-Model Gap Charts (Fig 5) with fixed axis limits to prevent cutoff."""
    fig, axes = plt.subplots(3, 2, figsize=(16, 22), sharex=False)

    for i, ds in enumerate(DATASET_ORDER):
        for j, ctx in enumerate(["Short / Synthetic", "Long / Natural"]):
            ax = axes[i, j]
            subset = merged_df[
                (merged_df["Dataset_Nice"] == ds) & (merged_df["Context_Class"] == ctx)
            ]
            pivot = subset.pivot_table(
                index="Core_Model", columns="Type", values=metric_col
            ).dropna()

            # Use the global model order defined earlier
            pivot = pivot.reindex([m for m in SORTED_CORE_MODELS if m in pivot.index])

            if pivot.empty:
                ax.set_title(f"{ds} - {ctx} (No Data)")
                continue

            y_pos = np.arange(len(pivot))

            # Draw the lines connecting the dots
            ax.hlines(
                y=y_pos,
                xmin=pivot["Base"],
                xmax=pivot["Instruct"],
                color="grey",
                alpha=0.5,
                linewidth=2,
            )

            # Plot dots
            ax.scatter(
                pivot["Base"],
                y_pos,
                color=PALETTE["Base"],
                s=100,
                label="Base",
                zorder=3,
            )
            ax.scatter(
                pivot["Instruct"],
                y_pos,
                color=PALETTE["Instruct"],
                s=100,
                label="Instruct",
                zorder=3,
            )

            # Add text labels with the % gap
            for k, (idx, row) in enumerate(pivot.iterrows()):
                diff = row["Instruct"] - row["Base"]
                color = "red" if diff > 0 else "green"

                # Place text slightly to the right of the furthest point
                x_pos = max(row["Base"], row["Instruct"])
                ax.text(
                    x_pos + 0.5,
                    k,
                    f"{diff:+.1f}%",
                    va="center",
                    color=color,
                    fontweight="bold",
                    fontsize=10,
                )

            # --- THE FIX: DYNAMIC X-AXIS BUFFER ---
            # Calculate the furthest point reached by data or text
            current_max = pivot[["Base", "Instruct"]].max().max()
            current_min = pivot[["Base", "Instruct"]].min().min()

            # Add 25% padding to the right for labels, and 5% to the left for breathing room
            x_range = current_max - current_min
            if x_range == 0:
                x_range = 5  # Prevent division by zero

            ax.set_xlim(
                max(0, current_min - (x_range * 0.1)), current_max + (x_range * 0.3)
            )

            # Formatting
            ax.set_yticks(y_pos)
            ax.set_yticklabels(pivot.index, fontsize=11)
            ax.set_title(f"{ds} — {ctx}", fontsize=13, fontweight="bold")
            ax.grid(True, axis="x", linestyle="--", alpha=0.7)

    # Global Legend
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Base",
            markerfacecolor=PALETTE["Base"],
            markersize=12,
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Instruct",
            markerfacecolor=PALETTE["Instruct"],
            markersize=12,
        ),
    ]

    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.96),
        ncol=2,
        fontsize=14,
    )
    plt.suptitle(
        f"{title_prefix} Gap (Per Model)", fontsize=22, fontweight="bold", y=0.99
    )

    # Adjust layout to prevent clipping of model names on the left
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close()


def plot_mech_full_grid():
    mech_melt = pd.melt(
        mech_df,
        id_vars=["Display_Name", "Dataset_Nice", "Context_Class"],
        value_vars=["Teachability", "Stickiness"],
        var_name="Metric",
        value_name="Rate",
    )

    # Ensure categorical sorting
    mech_melt["Display_Name"] = pd.Categorical(
        mech_melt["Display_Name"], categories=SORTED_DISPLAY_NAMES, ordered=True
    )

    fig, axes = plt.subplots(2, 3, figsize=(24, 15), sharey=True)

    for i, ctx in enumerate(["Short / Synthetic", "Long / Natural"]):
        for j, ds in enumerate(DATASET_ORDER):
            ax = axes[i, j]
            subset = mech_melt[
                (mech_melt["Context_Class"] == ctx) & (mech_melt["Dataset_Nice"] == ds)
            ].sort_values(by="Display_Name")

            sns.barplot(
                data=subset,
                x="Display_Name",
                y="Rate",
                hue="Metric",
                palette=PALETTE_MECH,
                edgecolor="black",
                ax=ax,
            )

            ax.set_title(f"{ds} — {ctx}", fontsize=16, fontweight="bold")

            # --- X-AXIS FORMATTING ---
            if i == 1:
                # Bottom Row: Show labels and set axis title to "Model"
                ax.set_xlabel("Model", fontsize=14, fontweight="bold")

                # Fix for the UserWarning: Set ticks explicitly before labels
                ax.set_xticks(range(len(subset["Display_Name"].unique())))
                ax.set_xticklabels(
                    subset["Display_Name"].unique(), rotation=45, ha="right", fontsize=9
                )
            else:
                # Top Row: Hide everything related to X-axis
                ax.set_xlabel("")
                ax.set_xticklabels([])
                ax.tick_params(axis="x", which="both", bottom=False, top=False)

            # --- Y-AXIS FORMATTING ---
            if j == 0:
                ax.set_ylabel("Rate (%)", fontsize=12)
            else:
                ax.set_ylabel("")

            # Remove individual legends
            if ax.get_legend():
                ax.get_legend().remove()

    # Global Legend logic
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.96),
        ncol=2,
        fontsize=16,
        title="Cognitive Mode",
        title_fontsize=18,
    )

    plt.suptitle("Stickiness vs. Teachability", fontsize=24, fontweight="bold", y=0.99)

    # Adjust layout to make room for rotated labels at the bottom
    plt.tight_layout(rect=[0, 0.05, 1, 0.94])

    plt.savefig(
        "Fig8_Grid_Stickiness_Teachability_Full.png", bbox_inches="tight", dpi=300
    )
    plt.close()


def plot_bucket_ablation(metric_name, col, fname, title):
    sub = df[
        (df["PK_Score"] == (10 if metric_name == "SCR" else 0))
        & (
            df["Context_Label"]
            == ("Ctx_correct" if metric_name == "SCR" else "Ctx_incorrect")
        )
    ].copy()
    sub["Metric"] = sub[col] * 100
    agg = (
        sub.groupby(["Dataset_Nice", "Type", "Ctx_Len_Bucket"])["Metric"]
        .mean()
        .reset_index()
    )
    agg["Dataset_Nice"] = pd.Categorical(
        agg["Dataset_Nice"], categories=DATASET_ORDER, ordered=True
    )

    buckets = ["Very Short (<50)", "Short (<150)", "Medium (<400)", "Long (400+)"]
    valid_buckets = [b for b in buckets if b in agg["Ctx_Len_Bucket"].unique()]

    # CHANGE 1: Increased width multiplier to 5.5 per plot for better spacing
    fig, axes = plt.subplots(
        1, len(valid_buckets), figsize=(5.5 * len(valid_buckets), 5), sharey=True
    )
    if len(valid_buckets) == 1:
        axes = [axes]

    for i, b in enumerate(valid_buckets):
        d = agg[agg["Ctx_Len_Bucket"] == b]
        if not d.empty:
            sns.lineplot(
                ax=axes[i],
                data=d,
                x="Dataset_Nice",
                y="Metric",
                hue="Type",
                style="Type",
                markers=True,
                dashes=False,
                palette=PALETTE,
                markersize=10,
                linewidth=2.5,
            )

        axes[i].set_title(b, fontsize=14, fontweight="bold")
        axes[i].set_xlabel("")
        axes[i].grid(True, alpha=0.5)

        # CHANGE 2: Fix alignment.
        # ha='right' + rotation_mode='anchor' pins the text end to the tick.
        # We must explicitly get the current labels and reset them with the new parameters.
        current_ticks = axes[i].get_xticks()
        current_labels = [l.get_text() for l in axes[i].get_xticklabels()]

        axes[i].set_xticks(current_ticks)
        axes[i].set_xticklabels(
            current_labels, rotation=30, ha="right", rotation_mode="anchor", fontsize=12
        )

        axes[i].tick_params(axis="y", labelsize=12)

        # Handle Legend
        if i == len(valid_buckets) - 1:
            axes[i].legend(fontsize=10, title="Model Type", title_fontsize=11)
        else:
            if axes[i].get_legend():
                axes[i].get_legend().remove()

    axes[0].set_ylabel(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()


def run_full_sequence_audit():
    print("\n" + "=" * 60)
    print("🧠 LOGPROB AUDIT: FIRST TOKEN VS. WHOLE SEQUENCE")
    print("=" * 60)

    # Filter for cases where model accepted the narrative lie (PK+ Ctx- Nat- Failure)
    hook_filter = (
        (df["PK_Score"] == 10)
        & (df["Context_Label"] == "Ctx_incorrect")
        & (df["Context_Class"] == "Long / Natural")
        & (df["Answer_Label"] == "Ans_Failure")
    )

    subset = df[hook_filter].copy()

    # Calculate difference for table
    for ds in DATASET_ORDER:
        ds_sub = subset[subset["Dataset_Nice"] == ds]
        print(f"\n--- Dataset: {ds} ---")
        print(f"{'Metric':<25} | {'Base Mean':<12} | {'Inst Mean':<12} | {'Delta':<8}")
        print("-" * 65)

        for name, col in [
            ("First Token LogProb", "FT_LogProb"),
            ("Total Sequence LogProb", "Total_Seq_LogProb"),
        ]:
            b_val = ds_sub[ds_sub["Type"] == "Base"][col].mean()
            i_val = ds_sub[ds_sub["Type"] == "Instruct"][col].mean()
            delta = i_val - b_val
            print(f"{name:<25} | {b_val:>12.4f} | {i_val:>12.4f} | {delta:>+8.4f}")

    # Plot Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    sns.barplot(
        data=subset,
        x="Dataset_Nice",
        y="FT_LogProb",
        hue="Type",
        palette=PALETTE,
        ax=ax1,
        errorbar="sd",
    )
    ax1.set_title("First Token Confidence (Accepted Lies)")

    sns.barplot(
        data=subset,
        x="Dataset_Nice",
        y="Total_Seq_LogProb",
        hue="Type",
        palette=PALETTE,
        ax=ax2,
        errorbar="sd",
    )
    ax2.set_title("Whole Sequence Confidence (Accepted Lies)")

    plt.tight_layout()
    plt.savefig("LogProb_Comparison_First_vs_Seq.png")
    plt.close()


# =============================================================================
# 9. SYNERGY COLLAPSE STATISTICS (CHI-SQUARE)
# =============================================================================


def run_synergy_collapse_stats():
    print("\n🔬 RUNNING SYNERGY COLLAPSE STATS (CHI-SQUARE) ...")

    # Filter for Synergy Quadrant (PK+, Ctx+) on Synthetic Data
    # Outcome of interest: FAILURE to use context
    subset = df[
        (df["PK_Score"] == 10)
        & (df["Context_Label"] == "Ctx_correct")
        & (df["Context_Class"] == "Short / Synthetic")
    ]

    # --- 1. AGGREGATE CHI-SQUARE (All Models Pooled) ---
    print("\n--- AGGREGATE CHI-SQUARE (Pooled 14 Models) ---")
    rows = []

    for ds in DATASET_ORDER:
        ds_sub = subset[subset["Dataset_Nice"] == ds]

        # Contingency Table: [ [Base_Fail, Base_Success], [Inst_Fail, Inst_Success] ]
        # Note: We count FAILURES as the "event of interest" for SCR
        b_data = ds_sub[ds_sub["Type"] == "Base"]
        i_data = ds_sub[ds_sub["Type"] == "Instruct"]

        b_fail = b_data["Is_Failure"].sum()
        b_succ = b_data[
            "Is_Success"
        ].sum()  # Success is NOT failure here (it's Is_Success==1)
        b_total = len(b_data)

        i_fail = i_data["Is_Failure"].sum()
        i_succ = i_data["Is_Success"].sum()
        i_total = len(i_data)

        # Verify totals
        if b_total == 0 or i_total == 0:
            continue

        # Construct Matrix
        # Row 0: Base, Row 1: Instruct
        # Col 0: Fail, Col 1: Success
        observed = [[b_fail, b_succ], [i_fail, i_succ]]

        # Run Chi2
        chi2, p, dof, expected = chi2_contingency(observed)

        # Calculate Rates
        b_rate = (b_fail / b_total) * 100
        i_rate = (i_fail / i_total) * 100

        sig = p_to_stars(p)
        p_str = f"{p:.1e}" if p < 0.001 else f"{p:.4f}"

        rows.append((ds, b_total, b_fail, b_rate, i_total, i_fail, i_rate, p, sig))

    # Print LaTeX Table for Aggregate
    print(r"\begin{table}[h] \centering \small")
    print(
        r"\caption{\textbf{Quantifying the Collapse (Aggregate).} Pooled failure rates in the \textbf{Unanimous Truth} quadrant ($PK^+ \cap Ctx_{cor}$) on Synthetic data. Chi-Square ($\chi^2$) tests for independence.}"
    )
    print(r"\label{tab:synergy_stats_agg}")
    print(r"\begin{tabular}{l c c c c c l} \toprule")
    print(
        r"\textbf{Dataset} & \textbf{Base $N$} & \textbf{Base SCR} & \textbf{Inst $N$} & \textbf{Inst SCR} & \textbf{Sig.} \\ \midrule"
    )

    for r in rows:
        # Highlight Red if Instruct SCR is significantly higher
        ffr_cell = (
            f"\\textbf{{{r[6]:.1f}\\%}}"
            if (r[6] > r[3] and r[7] < 0.05)
            else f"{r[6]:.1f}\\%"
        )
        if r[6] > r[3] and r[7] < 0.05:
            ffr_cell = f"\\cellcolor{{badred!15}}{ffr_cell}"

        print(
            f"{r[0]} & {r[1]} & {r[3]:.1f}\\% & {r[4]} & {ffr_cell} & {r[8]} ($p={r[7]:.1e}$) \\\\"
        )

    print(r"\bottomrule \end{tabular} \end{table}")

    # --- 2. PER-MODEL CHI-SQUARE (Appendix) ---
    print("\n--- PER-MODEL CHI-SQUARE (For Appendix) ---")

    pm_rows = []

    for ds in DATASET_ORDER:
        ds_sub = subset[subset["Dataset_Nice"] == ds]

        for core_model in SORTED_CORE_MODELS:
            m_sub = ds_sub[ds_sub["Core_Model"] == core_model]

            b_data = m_sub[m_sub["Type"] == "Base"]
            i_data = m_sub[m_sub["Type"] == "Instruct"]

            if len(b_data) == 0 or len(i_data) == 0:
                continue

            b_fail = b_data["Is_Failure"].sum()
            b_succ = b_data["Is_Success"].sum()
            i_fail = i_data["Is_Failure"].sum()
            i_succ = i_data["Is_Success"].sum()

            observed = [[b_fail, b_succ], [i_fail, i_succ]]

            # Use Fisher Exact if counts are low (<5), else Chi2
            if np.min(observed) < 5:
                res = fisher_exact(observed)
                p = res[1]
                test_type = "Fisher"
            else:
                chi2, p, dof, ex = chi2_contingency(observed)
                test_type = "Chi2"

            b_rate = (b_fail / len(b_data)) * 100
            i_rate = (i_fail / len(i_data)) * 100

            gap = i_rate - b_rate
            sig = p_to_stars(p)

            pm_rows.append((ds, core_model, b_rate, i_rate, gap, p, sig))

    # Print LaTeX Table for Per-Model
    print(r"\begin{table*}[h] \centering \scriptsize")
    print(
        r"\caption{\textbf{Detailed Synergy Statistics.} Chi-Square/Fisher tests per model on Synthetic Data. Positive Gap = Instruct Fails More.}"
    )
    print(r"\label{tab:synergy_stats_detailed}")
    print(r"\begin{tabular}{l l c c c l} \toprule")
    print(
        r"\textbf{Dataset} & \textbf{Model} & \textbf{Base SCR} & \textbf{Inst SCR} & \textbf{Gap} & \textbf{Sig.} \\ \midrule"
    )

    last_ds = None
    for r in pm_rows:
        ds = r[0]
        ds_cell = ds.split(" ")[0] if ds != last_ds else ""
        last_ds = ds
        name = r[1].replace("_", r"\_")

        gap_str = (
            f"\\textcolor{{badred}}{{{r[4]:+.1f}\\%}}"
            if r[4] > 0
            else f"{r[4]:+.1f}\\%"
        )
        if r[4] > 5 and r[5] < 0.05:
            gap_str = f"\\textbf{{{gap_str}}}"  # Bold significant failures

        print(
            f"{ds_cell} & {name} & {r[2]:.1f}\\% & {r[3]:.1f}\\% & {gap_str} & {r[6]} \\\\"
        )

    print(r"\bottomrule \end{tabular} \end{table*}")


def create_granular_bar_grid(metric_col, title_prefix, filename):
    merged_df["Core_Model"] = pd.Categorical(
        merged_df["Core_Model"], categories=SORTED_CORE_MODELS, ordered=True
    )
    target = merged_df.sort_values("Core_Model")
    fig, axes = plt.subplots(3, 2, figsize=(20, 18), sharex=False)
    for i, ds in enumerate(dataset_order):
        for j, ctx in enumerate(["Short / Synthetic", "Long / Natural"]):
            ax = axes[i, j]
            subset = target[
                (target["Dataset_Nice"] == ds) & (target["Context_Class"] == ctx)
            ]
            sns.barplot(
                data=subset,
                x="Core_Model",
                y=metric_col,
                hue="Type",
                hue_order=["Base", "Instruct"],
                palette=PALETTE,
                ax=ax,
                edgecolor="black",
                width=0.7,
            )
            ax.set_title(f"{ds} — {ctx}", fontsize=13, fontweight="bold")
            if i == 2:
                ax.set_xticklabels(
                    ax.get_xticklabels(), rotation=45, ha="right", fontsize=10
                )
            else:
                ax.set_xticklabels([])
            ax.get_legend().remove() if ax.get_legend() else None
            ax.set_ylabel("Rate (%)" if j == 0 else "")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=2,
        fontsize=14,
        title="Model Type",
    )
    plt.suptitle(
        f"{title_prefix} Detailed Breakdown", fontsize=20, fontweight="bold", y=0.99
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close()


def run_main_text_aggregate_chi2():
    """
    Generates high-level aggregate statistical tables for the Main Paper.
    Updated to align with Sections 4, 5, and 6.
    """
    print("\n" + "=" * 60)
    print("📈 MAIN PAPER: AGGREGATE FLEET-WIDE STATISTICAL AUDIT")
    print("=" * 60)

    # Scenarios redefined to perfectly match Sections 4, 5, and 6
    scenarios = [
        {
            "section": "Section 4: Synergy Collapse",
            "title": "Truth-on-Truth Failure",
            # Quadrant: PK+ (Known) AND Ctx_Correct.
            # Logic: Both sources are right. Do they fail anyway?
            "filter": (df["PK_Score"] == 10) & (df["Context_Label"] == "Ctx_correct"),
            "ctx_class": "Short / Synthetic",
            "metric": "Is_Failure",  # Measuring the collapse in accuracy
            "label_a": "Faithful",
            "label_b": "Failed (SCR)",
        },
        {
            "section": "Section 5: Narrative Hook (long)",
            "title": "Lie-as-Truth Adoption",
            # Quadrant: PK+ (Known) AND Ctx_Incorrect (Lie).
            # Logic: I know the truth, but context lies. Do I follow the lie?
            "filter": (df["PK_Score"] == 10) & (df["Context_Label"] == "Ctx_incorrect"),
            "ctx_class": "Long / Natural",
            "metric": "Is_Failure",
            "label_a": "Skeptical",
            "label_b": "Hooked (NHR)",
        },
        {
            "section": "Section 5: Narrative Hook (short)",
            "title": "Lie-as-Truth Adoption",
            # Quadrant: PK+ (Known) AND Ctx_Incorrect (Lie).
            # Logic: I know the truth, but context lies. Do I follow the lie?
            "filter": (df["PK_Score"] == 10) & (df["Context_Label"] == "Ctx_incorrect"),
            "ctx_class": "Short / Synthetic",
            "metric": "Is_Failure",
            "label_a": "Skeptical",
            "label_b": "Hooked (NHR)",
        },
        {
            "section": "Section 6: Phantom Knowledge",
            "title": "Truth-from-Lie Reconstruction",
            # Quadrant: PK- (Unknown) AND Ctx_Incorrect (Lie).
            # Logic: Neither knows it. Do I guess/reconstruct the truth?
            "filter": (df["PK_Score"] == 0) & (df["Context_Label"] == "Ctx_incorrect"),
            "ctx_class": "Long / Natural",
            "metric": "Is_Success",  # Measuring hallucinating the truth
            "label_a": "Followed Lie",
            "label_b": "Phantom Truth (PKR)",
        },
    ]

    for sc in scenarios:
        print(f"\n🚀 {sc['section']} - {sc['title']} ({sc['ctx_class']})")
        print(
            f"{'Dataset':<15} | {'Type':<10} | {sc['label_a']:<15} | {sc['label_b']:<18} | {'Rate %':<8} | {'P-Value'}"
        )
        print("-" * 95)

        for ds in DATASET_ORDER:
            # Filter the dataframe for the specific scenario and reasoning complexity
            subset = df[
                sc["filter"]
                & (df["Dataset_Nice"] == ds)
                & (df["Context_Class"] == sc["ctx_class"])
            ]

            base = subset[subset["Type"] == "Base"]
            inst = subset[subset["Type"] == "Instruct"]

            def get_counts(data):
                if len(data) == 0:
                    return 0, 0, 0
                count_b = int(data[sc["metric"]].sum())
                count_a = len(data) - count_b
                rate = (count_b / len(data)) * 100
                return count_a, count_b, rate

            ba, bb, br = get_counts(base)
            ia, ib, ir = get_counts(inst)

            # Statistical Significance
            contingency = [[ba, bb], [ia, ib]]
            try:
                # If N is small, chi2 might be unstable, but at your scale, it's perfect.
                chi2, p, _, _ = chi2_contingency(contingency)
                p_str = f"{p:.2e}"
            except:
                p_str = "N/A"

            print(f"{ds:<15} | Base       | {ba:<15} | {bb:<18} | {br:>7.1f}% | -")
            print(
                f"{'':<15} | Instruct   | {ia:<15} | {ib:<18} | {ir:>7.1f}% | {p_str}"
            )
            print("-" * 95)


# =============================================================================
# 6. LOGPROB PLOTTING LOGIC (CONFIDENCE & GAP)
# =============================================================================


def plot_synergy_friction_and_gap():
    subset = df[
        (df["PK_Score"] == 10)
        & (df["Context_Label"] == "Ctx_correct")
        & (df["Context_Class"] == "Short / Synthetic")
    ].copy()

    # DECISIVENESS: Use FT_Prob_Gap (First Token Gap)
    plt.figure(figsize=(10, 6))
    ax1 = sns.barplot(
        data=subset, x="Dataset_Nice", y="FT_Prob_Gap", hue="Type", palette=PALETTE
    )
    add_significance_bars(ax1, subset, "Dataset_Nice", "FT_Prob_Gap", "Type")
    plt.title("Epistemic Friction: Decision Gap")
    plt.savefig("Fig_Prob_Gap_FT_Synergy.png")
    plt.close()

    # REASONING PATH: Use FS_Prob_Gap (Average Sequence Gap)
    plt.figure(figsize=(10, 6))
    ax2 = sns.barplot(
        data=subset, x="Dataset_Nice", y="FS_Prob_Gap", hue="Type", palette=PALETTE
    )
    add_significance_bars(ax2, subset, "Dataset_Nice", "FS_Prob_Gap", "Type")
    plt.title("Epistemic Friction: Full Sequence Decisiveness")
    plt.savefig("Fig_Prob_Gap_FS_Synergy.png")
    plt.close()


def plot_narrative_hook_confidence():
    subset = df[
        (df["PK_Score"] == 10)
        & (df["Context_Label"] == "Ctx_incorrect")
        & (df["Context_Class"] == "Long / Natural")
        & (df["Is_Failure"] == 1)
    ].copy()

    # CONFIDENCE: Use FT_LogProb
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=subset, x="Dataset_Nice", y="FT_LogProb", hue="Type", palette=PALETTE
    )
    add_significance_bars(ax, subset, "Dataset_Nice", "FT_LogProb", "Type")
    plt.title("Narrative Hook: Confidence in Lies")
    plt.savefig("Fig_LogProb_Hook_Confidence.png")
    plt.close()


def plot_narrative_hook_gap():
    subset = df[
        (df["PK_Score"] == 10)
        & (df["Context_Label"] == "Ctx_incorrect")
        & (df["Context_Class"] == "Long / Natural")
        & (df["Is_Failure"] == 1)
    ].copy()

    # DECISIVENESS: Use FS_Prob_Gap
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=subset, x="Dataset_Nice", y="FS_Prob_Gap", hue="Type", palette=PALETTE
    )
    add_significance_bars(ax, subset, "Dataset_Nice", "FS_Prob_Gap", "Type")
    plt.title("Narrative Hook: Decisiveness in Accepted Lies")
    plt.savefig("Fig_LogProb_Hook_Gap.png")
    plt.close()


def plot_narrative_hook_gap():
    print("\n📊 Plotting Narrative Hook Gap (Decisiveness in Failures)...")
    # Filter: PK+, Ctx-, NATURAL, FAILURE
    # The model knew the truth, saw a lie in narrative prose, and accepted the lie.
    subset = df[
        (df["PK_Score"] == 10)
        & (df["Context_Label"] == "Ctx_incorrect")
        & (df["Context_Class"] == "Long / Natural")
        & (df["Answer_Label"] == "Ans_Failure")
    ].copy()

    if subset.empty:
        print("Skipping Hook Gap Plot - No failure cases found.")
        return

    plt.figure(figsize=(10, 6), dpi=300)
    ax = sns.barplot(
        data=subset,
        x="Dataset_Nice",
        y="FS_Prob_Gap",
        hue="Type",
        palette=PALETTE,
        edgecolor="black",
        errorbar=("ci", 95),
        order=DATASET_ORDER,
    )
    add_significance_bars(ax, subset, "Dataset_Nice", "FS_Prob_Gap", "Type")

    plt.title(
        "The Narrative Hook: Decisiveness in Accepted Lies",
        fontsize=14,
        fontweight="bold",
    )
    plt.ylabel("Confidence Gap (1st - 2nd LogProb)\n(Higher = More Decisively Wrong)")
    plt.legend(title="Model Type")
    plt.tight_layout()
    plt.savefig("Fig_LogProb_Hook_Gap.png")
    plt.close()


def run_mechanistic_audit_rq3_rq4():
    """
    Analyzes the internal confidence (LogProbs) to explain the
    Synergy Collapse (RQ3) and Latent Priming (RQ4).
    """
    print("\n" + "=" * 60)
    print("🧠 MECHANISTIC EXPLANATION: LOGPROB AUDIT FOR RQ3 & RQ4")
    print("=" * 60)

    # SCENARIO A: RQ3 Synergy Friction (Why they collapse on Raw Data)
    # We look at cases where PK+ and Ctx+ agree on Synthetic data.
    rq3_filter = (
        (df["PK_Score"] == 10)
        & (df["Context_Label"] == "Ctx_correct")
        & (df["Context_Class"] == "Short / Synthetic")
    )

    # SCENARIO B: RQ4 Priming Decisiveness (Why they 'guess' truth on Narrative Lies)
    # We look specifically at the SUCCESSES in the Phantom Quadrant on Natural data.
    rq4_filter = (
        (df["PK_Score"] == 0)
        & (df["Context_Label"] == "Ctx_incorrect")
        & (df["Context_Class"] == "Long / Natural")
        & (df["Is_Success"] == 1)
    )

    scenarios = [
        {
            "rq": "RQ3",
            "title": "Epistemic Friction (Synergy)",
            "filter": rq3_filter,
            "metric": "FS_Prob_Gap",
            "higher_is_better": True,
        },
        {
            "rq": "RQ4",
            "title": "Priming Decisiveness (Phantom)",
            "filter": rq4_filter,
            "metric": "FS_Prob_Gap",
            "higher_is_better": False,
        },
    ]

    for sc in scenarios:
        print(f"\n🔍 {sc['rq']} audit: {sc['title']}")
        print(
            f"{'Dataset':<15} | {'Base Mean':<10} | {'Inst Mean':<10} | {'Delta':<8} | {'Sig.'}"
        )
        print("-" * 70)

        for ds in DATASET_ORDER:
            subset = df[sc["filter"] & (df["Dataset_Nice"] == ds)]
            base_vals = subset[subset["Type"] == "Base"][sc["metric"]].dropna()
            inst_vals = subset[subset["Type"] == "Instruct"][sc["metric"]].dropna()

            if len(base_vals) > 5 and len(inst_vals) > 5:
                b_m = base_vals.mean()
                i_m = inst_vals.mean()
                diff = i_m - b_m
                _, p = mannwhitneyu(base_vals, inst_vals)

                sig = p_to_stars(p)
                # For RQ3, Negative Delta is BAD (Friction)
                # For RQ4, Positive Delta is 'INTERESTING' (Authoritative Hallucination)
                color = "red" if (diff < 0 and sc["rq"] == "RQ3") else "blue"

                print(f"{ds:<15} | {b_m:>10.2f} | {i_m:>10.2f} | {diff:>+8.2f} | {sig}")
            else:
                print(f"{ds:<15} | Insufficient Data")


# =============================================================================
# 10. GENERATE TABLE GENERATION WRAPPERS
# =============================================================================


def format_gap_cell(gap_pct: float):
    color = (
        "red" if gap_pct > 0 else "teal"
    )  # Red = Worse/More Failure/More Hallucination
    if abs(gap_pct) < 0.1:
        color = "black"
    return f"\\textcolor{{{color}}}{{{gap_pct:+.1f}\\%}}"


def print_audit_table(audit):
    print(f"\nGENERATING TABLE: {audit['title']}")
    rows, pvals = [], []
    for ds_name in DATASET_ORDER:
        subset = df[
            (df["Dataset_Nice"] == ds_name)
            & (df["Context_Class"] == audit["context_class"])
            & (df["PK_Score"] == audit["filter_pk"])
            & (df["Context_Label"] == audit["filter_ctx_label"])
        ]
        if subset.empty:
            continue

        for core_model in SORTED_CORE_MODELS:
            model_data = subset[subset["Core_Model"] == core_model]
            base_data = model_data[model_data["Type"] == "Base"]
            inst_data = model_data[model_data["Type"] == "Instruct"]

            if base_data.empty or inst_data.empty:
                continue

            # Calculate Rates
            b_t = int(base_data[audit["target_outcome"]].sum())
            b_n = len(base_data)
            i_t = int(inst_data[audit["target_outcome"]].sum())
            i_n = len(inst_data)

            diff, (ci_lo, ci_hi), p = diff_proportion_stats(b_t, b_n, i_t, i_n)

            rows.append(
                (
                    ds_name,
                    core_model,
                    {
                        "b_rate": (b_t / b_n * 100),
                        "i_rate": (i_t / i_n * 100),
                        "gap": ((i_t / i_n) - (b_t / b_n)) * 100,
                        "p": p,
                    },
                )
            )
            pvals.append(p)

    # Apply BH Correction
    p_adj = bh_adjust(pvals)

    # Print LaTeX
    print(r"\begin{table*}[h] \centering \small")
    print(f"\\caption{{{audit['caption']}}}")
    print(r"\begin{tabular}{l l c c c l} \toprule")
    print(
        r"\textbf{Dataset} & \textbf{Model} & \textbf{Base} & \textbf{Inst} & \textbf{Gap} & \textbf{Sig.} \\ \midrule"
    )

    last_ds = None
    for idx, (ds, model, data) in enumerate(rows):
        ds_cell = ds.split(" ")[0] if ds != last_ds else ""
        last_ds = ds
        name = model.replace("_", r"\_")

        gap_str = format_gap_cell(data["gap"])
        sig = p_to_stars(p_adj[idx])

        print(
            f"{ds_cell} & {name} & {data['b_rate']:.1f}\\% & {data['i_rate']:.1f}\\% & {gap_str} & {sig} \\\\"
        )
    print(r"\bottomrule \end{tabular} \end{table*}")


def get_did_table():
    print("\nGENERATING TABLE: The Alignment Effect (DiD on Conflict Stickiness)")
    subset = df[(df["PK_Score"] == 10) & (df["Context_Label"] == "Ctx_incorrect")]

    rows, pvals = [], []
    for ds_name in DATASET_ORDER:
        ds_subset = subset[subset["Dataset_Nice"] == ds_name]
        for core_model in SORTED_CORE_MODELS:
            model_data = ds_subset[ds_subset["Core_Model"] == core_model]

            # Helper to get success/total for Short vs Long
            def get_stats(type_str):
                d = model_data[model_data["Type"] == type_str]
                s = d[d["Context_Class"] == "Short / Synthetic"]
                n = d[d["Context_Class"] == "Long / Natural"]
                if len(s) == 0 or len(n) == 0:
                    return None
                return (s["Is_Success"].sum(), len(s), n["Is_Success"].sum(), len(n))

            b = get_stats("Base")
            i = get_stats("Instruct")

            if b and i:
                # Rates
                pb_s, nb_s, pb_n, nb_n = b[0] / b[1], b[1], b[2] / b[3], b[3]
                pi_s, ni_s, pi_n, ni_n = i[0] / i[1], i[1], i[2] / i[3], i[3]

                # Deltas (Natural - Synthetic)
                d_b = pb_n - pb_s
                d_i = pi_n - pi_s

                # Alignment Effect (DiD)
                eff = d_i - d_b

                # Z-test for interaction
                # SE = sqrt(sum of variances)
                se = np.sqrt(
                    (pb_s * (1 - pb_s) / nb_s)
                    + (pb_n * (1 - pb_n) / nb_n)
                    + (pi_s * (1 - pi_s) / ni_s)
                    + (pi_n * (1 - pi_n) / ni_n)
                )

                p = 2 * (1 - norm.cdf(abs(eff / se))) if se > 0 else 1.0

                rows.append(
                    (ds_name, core_model, {"db": d_b, "di": d_i, "eff": eff, "p": p})
                )
                pvals.append(p)

    p_adj = bh_adjust(pvals)

    print(r"\begin{table*}[t] \centering \small")
    print(
        r"\caption{\textbf{Does Alignment Amplify Style Sensitivity?} Comparison of the \textbf{Defense Collapse}.}"
    )
    print(r"\begin{tabular}{l l c c c l} \toprule")
    print(
        r"\textbf{Dataset} & \textbf{Family} & \textbf{Base $\Delta$} & \textbf{Inst $\Delta$} & \textbf{Align. Effect} & \textbf{Sig.} \\ \midrule"
    )

    last_ds = None
    for idx, (ds, model, data) in enumerate(rows):
        ds_cell = ds.split(" ")[0] if ds != last_ds else ""
        last_ds = ds
        name = model.replace("_", r"\_")

        # Color coding for Effect (Negative is BAD/RED)
        c = "badred" if data["eff"] < -0.05 else "goodblue"
        if abs(data["eff"]) < 0.05:
            c = "black"
        eff_str = f"\\textcolor{{{c}}}{{{data['eff'] * 100:+.1f}\\%}}"

        # Bold Instruct Delta if collapse is huge
        di_str = (
            f"\\textbf{{{data['di'] * 100:+.1f}\\%}}"
            if data["di"] < -0.15
            else f"{data['di'] * 100:+.1f}\\%"
        )

        sig = p_to_stars(p_adj[idx])
        print(
            f"{ds_cell} & {name} & {data['db'] * 100:+.1f}\\% & {di_str} & {eff_str} & {sig} \\\\"
        )
    print(r"\bottomrule \end{tabular} \end{table*}")


def get_detailed_mechanistic_table():
    print("\nGENERATING TABLE: Detailed Mechanistic LogProb Audit (4 Metrics)")

    # SCENARIO 1: Synergy (Raw Data).
    # We look at ALL attempts (Success + Failure) to see general friction.
    # Hypothesis: Instruct models are less confident/decisive on raw data.
    syn_filter = (
        (df["PK_Score"] == 10)
        & (df["Context_Label"] == "Ctx_correct")
        & (df["Context_Class"] == "Short / Synthetic")
    )

    # SCENARIO 2: Narrative Hook (Accepted Lies).
    # We look ONLY at FAILURES (where they swallowed the lie).
    # Hypothesis: Instruct models are MORE confident/decisive when tricked by prose.
    hook_filter = (
        (df["PK_Score"] == 10)
        & (df["Context_Label"] == "Ctx_incorrect")
        & (df["Context_Class"] == "Long / Natural")
        & (df["Answer_Label"] == "Ans_Failure")
    )

    rows = []

    MIN_N = 5  # Minimum samples to calculate stats

    for ds in DATASET_ORDER:
        for core_model in SORTED_CORE_MODELS:
            # Get Subsets
            syn_sub = df[
                syn_filter
                & (df["Dataset_Nice"] == ds)
                & (df["Core_Model"] == core_model)
            ]
            hook_sub = df[
                hook_filter
                & (df["Dataset_Nice"] == ds)
                & (df["Core_Model"] == core_model)
            ]

            # Prepare Data Dictionary
            res = {}

            # --- CALCULATE SYNERGY METRICS (Expect Negative Delta) ---
            for metric, col in [("LP", "FT_LogProb"), ("Gap", "FS_Prob_Gap")]:
                b_vals = syn_sub[syn_sub["Type"] == "Base"][col].dropna()
                i_vals = syn_sub[syn_sub["Type"] == "Instruct"][col].dropna()

                if len(b_vals) >= MIN_N and len(i_vals) >= MIN_N:
                    _, p = mannwhitneyu(b_vals, i_vals, alternative="two-sided")
                    diff = i_vals.mean() - b_vals.mean()

                    # Bad = Negative (Instruct is struggling/hesitating on raw data)
                    color = "badred" if diff < -0.05 else "black"
                    sig = p_to_stars(p)
                    res[f"Syn_{metric}"] = f"\\textcolor{{{color}}}{{{diff:+.2f}}}{sig}"
                else:
                    res[f"Syn_{metric}"] = "-"

            # --- CALCULATE HOOK METRICS (Expect Positive Delta) ---
            for metric, col in [("LP", "FT_LogProb"), ("Gap", "FS_Prob_Gap")]:
                b_vals = hook_sub[hook_sub["Type"] == "Base"][col].dropna()
                i_vals = hook_sub[hook_sub["Type"] == "Instruct"][col].dropna()

                # Special Case: If Base is too robust (no failures), we can't compare
                if len(i_vals) >= MIN_N and len(b_vals) < MIN_N:
                    res[f"Hook_{metric}"] = "\\textcolor{badred}{Base Robust}"
                elif len(b_vals) >= MIN_N and len(i_vals) >= MIN_N:
                    _, p = mannwhitneyu(b_vals, i_vals, alternative="two-sided")
                    diff = i_vals.mean() - b_vals.mean()

                    # Bad = Positive (Instruct is confidently wrong)
                    color = "badred" if diff > 0.05 else "black"
                    sig = p_to_stars(p)
                    res[f"Hook_{metric}"] = (
                        f"\\textcolor{{{color}}}{{{diff:+.2f}}}{sig}"
                    )
                else:
                    res[f"Hook_{metric}"] = "-"

            # Only add row if we have at least some data
            if any(v != "-" for v in res.values()):
                rows.append((ds, core_model, res))

    # Print LaTeX
    print(r"\begin{table*}[t] \centering \scriptsize")
    print(
        r"\caption{\textbf{Mechanistic Audit of Confidence:} Comparison of $\Delta$ (Instruct - Base) for Avg LogProb and Decision Gap. \textbf{Synergy (Left):} \textcolor{badred}{Negative} values indicate Instruct models experience higher friction (lower confidence) on raw data. \textbf{Hook (Right):} \textcolor{badred}{Positive} values indicate Instruct models are more decisively wrong when accepting narrative lies.}"
    )
    print(r"\begin{tabular}{l l c c | c c} \toprule")
    print(
        r"& & \multicolumn{2}{c|}{\textbf{Synergy Friction (Raw Data)}} & \multicolumn{2}{c}{\textbf{The Narrative Hook (Lie Acceptance)}} \\"
    )
    print(
        r"\textbf{Dataset} & \textbf{Model} & \textbf{$\Delta$ LogProb} & \textbf{$\Delta$ Gap} & \textbf{$\Delta$ LogProb} & \textbf{$\Delta$ Gap} \\ \midrule"
    )

    last_ds = None
    for ds, model, data in rows:
        ds_cell = ds.split(" ")[0] if ds != last_ds else ""
        last_ds = ds
        name = model.replace("_", r"\_")
        print(
            f"{ds_cell} & {name} & {data.get('Syn_LP', '-')} & {data.get('Syn_Gap', '-')} & {data.get('Hook_LP', '-')} & {data.get('Hook_Gap', '-')} \\\\"
        )

    print(r"\bottomrule \end{tabular} \end{table*}")


def print_dual_mechanistic_table(
    scenario_name, filter_mask, higher_is_better=True, min_n=5
):
    """
    Prints a LaTeX table comparing First Token Delta and Sequence Delta.
    higher_is_better=True (Synergy): Negative Delta is Bad (Red)
    higher_is_better=False (Hook): Positive Delta is Bad (Red)
    """
    print(f"\n--- GENERATING DUAL LOGPROB TABLE: {scenario_name} ---")

    rows = []
    subset = df[filter_mask]

    for ds in DATASET_ORDER:
        for model in SORTED_CORE_MODELS:
            m_sub = subset[
                (subset["Dataset_Nice"] == ds) & (subset["Core_Model"] == model)
            ]

            b_data = m_sub[m_sub["Type"] == "Base"]
            i_data = m_sub[m_sub["Type"] == "Instruct"]

            if len(b_data) < min_n or len(i_data) < min_n:
                continue

            res = {}
            # Ensure keys match the print statement below: "First" and "Seq"
            for metric_label, col in [("First", "FT_Prob_Gap"), ("Seq", "FS_Prob_Gap")]:
                b_vals = b_data[col].dropna()
                i_vals = i_data[col].dropna()

                # Stats
                stat, p = mannwhitneyu(b_vals, i_vals)
                diff = i_vals.mean() - b_vals.mean()
                sig = p_to_stars(p)

                # Color logic
                is_bad = (diff < -0.02) if higher_is_better else (diff > 0.02)
                color = "badred" if (is_bad and p < 0.05) else "black"

                res[metric_label] = f"\\textcolor{{{color}}}{{{diff:+.2f}}}{sig}"

            rows.append((ds, model, res))

    # Print LaTeX
    print(r"\begin{table*}[t] \centering \small")
    print(f"\\caption{{Mechanistic Audit: {scenario_name} ($\Delta$ Instruct - Base)}}")
    print(r"\begin{tabular}{l l c c} \toprule")
    # Updated header to reflect the actual metrics being printed
    print(
        r"\textbf{Dataset} & \textbf{Model} & \textbf{$\Delta$ FT Prob Gap} & \textbf{$\Delta$ FS Prob Gap} \\ \midrule"
    )

    last_ds = None
    for ds, model, data in rows:
        ds_cell = ds.split(" ")[0] if ds != last_ds else ""
        last_ds = ds
        name = model.replace("_", r"\_")
        # FIXED: Access 'Seq' instead of 'Gap'
        print(f"{ds_cell} & {name} & {data['First']} & {data['Seq']} \\\\")

    print(r"\bottomrule \end{tabular} \end{table*}")


def generate_synhook_tables():
    print("\n📊 GENERATING 4 SEPARATE LOGPROB TABLES...")

    # Define the 4 Configurations
    configs = [
        {
            "name": "Synergy_Confidence",
            "title": "Mechanistic Audit: Epistemic Friction (Confidence)",
            "caption": "Avg First Token LogProb on Raw Data (Synergy). Lower (more negative) means higher friction/uncertainty.",
            "filter": (df["PK_Score"] == 10)
            & (df["Context_Label"] == "Ctx_correct")
            & (df["Context_Class"] == "Short / Synthetic"),
            "metric": "FT_LogProb",
            "higher_is_better": True,  # Closer to 0 is better (more confident)
        },
        {
            "name": "Synergy_Gap",
            "title": "Mechanistic Audit: Epistemic Friction (Decision Gap)",
            "caption": "Avg LogProb Gap (1st - 2nd token) on Raw Data (Synergy). Smaller gap means higher friction/hesitation.",
            "filter": (df["PK_Score"] == 10)
            & (df["Context_Label"] == "Ctx_correct")
            & (df["Context_Class"] == "Short / Synthetic"),
            "metric": "FS_Prob_Gap",
            "higher_is_better": True,  # Larger gap is better (more decisive)
        },
        {
            "name": "Hook_Confidence",
            "title": "Mechanistic Audit: Narrative Hook (Confidence in Lies)",
            "caption": "Avg First Token LogProb when Accepting Lies (Narrative Hook). Higher (closer to 0) means more confidently wrong.",
            "filter": (df["PK_Score"] == 10)
            & (df["Context_Label"] == "Ctx_incorrect")
            & (df["Context_Class"] == "Long / Natural")
            & (df["Answer_Label"] == "Ans_Failure"),
            "metric": "FT_LogProb",
            "higher_is_better": False,  # Lower (more negative) is better (less confident in the lie)
        },
        {
            "name": "Hook_Gap",
            "title": "Mechanistic Audit: Narrative Hook (Decisiveness in Lies)",
            "caption": "Avg LogProb Gap when Accepting Lies (Narrative Hook). Larger gap means the model decisively chose the lie over the truth.",
            "filter": (df["PK_Score"] == 10)
            & (df["Context_Label"] == "Ctx_incorrect")
            & (df["Context_Class"] == "Long / Natural")
            & (df["Answer_Label"] == "Ans_Failure"),
            "metric": "FS_Prob_Gap",
            "higher_is_better": False,  # Smaller gap is better (less decisive about the lie)
        },
    ]

    for conf in configs:
        print(f"\n--- Processing Table: {conf['title']} ---")
        rows = []
        subset = df[conf["filter"]]

        for ds in DATASET_ORDER:
            ds_sub = subset[subset["Dataset_Nice"] == ds]
            for core_model in SORTED_CORE_MODELS:
                m_sub = ds_sub[ds_sub["Core_Model"] == core_model]

                b_vals = m_sub[m_sub["Type"] == "Base"][conf["metric"]].dropna()
                i_vals = m_sub[m_sub["Type"] == "Instruct"][conf["metric"]].dropna()

                MIN_N = 5

                # Default values
                b_str, i_str, gap_str, sig = "-", "-", "-", ""

                if len(b_vals) >= MIN_N and len(i_vals) >= MIN_N:
                    # Calculate Stats
                    b_mean = b_vals.mean()
                    i_mean = i_vals.mean()
                    diff = i_mean - b_mean
                    stat, p = mannwhitneyu(b_vals, i_vals, alternative="two-sided")

                    # Formatting
                    b_str = f"{b_mean:.2f}"
                    i_str = f"{i_mean:.2f}"
                    sig = p_to_stars(p)

                    # Color Coding the Gap
                    is_bad = (diff < 0) if conf["higher_is_better"] else (diff > 0)

                    if abs(diff) > 0.05:
                        color = "badred" if is_bad else "goodblue"
                    else:
                        color = "black"

                    gap_str = f"\\textcolor{{{color}}}{{{diff:+.2f}}}"

                    rows.append((ds, core_model, b_str, i_str, gap_str, sig))

                elif len(i_vals) >= MIN_N and len(b_vals) < MIN_N:
                    rows.append(
                        (ds, core_model, "Robust", f"{i_vals.mean():.2f}", "N/A", "")
                    )

        # Print LaTeX
        print(r"\begin{table}[h] \centering \small")
        print(f"\\caption{{{conf['caption']}}}")
        print(r"\begin{tabular}{l l c c c l} \toprule")
        print(
            r"\textbf{Dataset} & \textbf{Model} & \textbf{Base} & \textbf{Inst} & \textbf{Gap} & \textbf{Sig.} \\ \midrule"
        )

        last_ds = None
        for ds, model, b, i, g, s in rows:
            ds_cell = ds.split(" ")[0] if ds != last_ds else ""
            last_ds = ds
            name = model.replace("_", r"\_")
            print(f"{ds_cell} & {name} & {b} & {i} & {g} & {s} \\\\")

        print(r"\bottomrule \end{tabular} \end{table}")


def run_all_tables():
    audits = [
        {
            "title": "SCR Synthetic",
            "metric": "SCR",
            "filter_pk": 10,
            "filter_ctx_label": "Ctx_correct",
            "target_outcome": "Is_Failure",
            "context_class": "Short / Synthetic",
            "caption": "SCR Synthetic",
        },
        {
            "title": "SCR Narrative",
            "metric": "SCR",
            "filter_pk": 10,
            "filter_ctx_label": "Ctx_correct",
            "target_outcome": "Is_Failure",
            "context_class": "Long / Natural",
            "caption": "SCR Narrative",
        },
        {
            "title": "Stickiness Synthetic",
            "metric": "Stickiness",
            "filter_pk": 10,
            "filter_ctx_label": "Ctx_incorrect",
            "target_outcome": "Is_Success",
            "context_class": "Short / Synthetic",
            "caption": "Stickiness Synthetic",
        },
        {
            "title": "Stickiness Natural",
            "metric": "Stickiness",
            "filter_pk": 10,
            "filter_ctx_label": "Ctx_incorrect",
            "target_outcome": "Is_Success",
            "context_class": "Long / Natural",
            "caption": "Stickiness Natural",
        },
        {
            "title": "PKR Synthetic",
            "metric": "PKR",
            "filter_pk": 0,
            "filter_ctx_label": "Ctx_incorrect",
            "target_outcome": "Is_Success",
            "context_class": "Short / Synthetic",
            "caption": "PKR Synthetic",
        },
        {
            "title": "PKR Natural",
            "metric": "PKR",
            "filter_pk": 0,
            "filter_ctx_label": "Ctx_incorrect",
            "target_outcome": "Is_Success",
            "context_class": "Long / Natural",
            "caption": "PKR Natural",
        },
    ]
    for a in audits:
        print_audit_table(a)
    get_did_table()
    run_synergy_collapse_stats()  # NEW CALL HERE


# =============================================================================
# 8. EXECUTION
# =============================================================================
print("Generating Plots...")

# 1. Main Performance Splits
plot_split_lines(
    "Synergy Collapse Rate (%)",
    "Fig1_SCR_Split.png",
    "Synergy Collapse Rate",
    "SCR (%)",
)
plot_split_lines(
    "Phantom Knowledge Rate (PKR) (%)",
    "Fig4_PKR_Split.png",
    "Phantom Knowledge Rate",
    "PKR (%)",
)

plot_split_lines_with_traces(
    "Synergy Collapse Rate (%)",
    "Fig1_SCR_Iconic.png",
    "Synergy Collapse Rate",
    "SCR (%)",
)
plot_split_lines_with_traces(
    "Phantom Knowledge Rate (PKR) (%)",
    "Fig4_PKR_Iconic.png",
    "Phantom Knowledge Rate",
    "PKR (%)",
)

print("Generating Iconic Split Plots with SE Bands...")

# 1. Performance Metrics (SCR and PKR)
plot_iconic_with_se_split(
    "Synergy Collapse Rate (%)",
    "Fig1_SCR_Iconic_SE.png",
    "Synergy Collapse Rate",
    "SCR (%)",
)
plot_iconic_with_se_split(
    "Phantom Knowledge Rate (PKR) (%)",
    "Fig4_PKR_Iconic_SE.png",
    "Phantom Knowledge Rate",
    "PKR (%)",
)


# Assuming Stickiness is Is_Success where PK=10 and Ctx=incorrect
subset_sticky = df[(df["PK_Score"] == 10) & (df["Context_Label"] == "Ctx_incorrect")]
# Stickiness: Model says the TRUTH (Success)
sticky_stats = (
    subset_sticky.groupby(["Display_Name", "Dataset_Nice", "Context_Class"])[
        "Is_Success"
    ].mean()
    * 100
)
# NHR: Model says the LIE (Failure)
nhr_stats = (
    subset_sticky.groupby(["Display_Name", "Dataset_Nice", "Context_Class"])[
        "Is_Failure"
    ].mean()
    * 100
)

sticky_df = sticky_stats.reset_index(name="Stickiness (%)")
nhr_df = nhr_stats.reset_index(name="Narrative Hook Rate (%)")

# Merge into your main plotting dataframe
merged_df = merged_df.merge(
    sticky_df, on=["Display_Name", "Dataset_Nice", "Context_Class"], how="left"
)
merged_df = merged_df.merge(
    nhr_df, on=["Display_Name", "Dataset_Nice", "Context_Class"], how="left"
)


# To plot how often the models stick to the truth
plot_iconic_with_se_split(
    "Stickiness (%)",
    "Fig_Stickiness_Iconic.png",
    "Truth Persistence (Stickiness)",
    "Stickiness Rate (%)",
)

# To plot how often models are "hooked" by the lie
plot_iconic_with_se_split(
    "Narrative Hook Rate (%)",
    "Fig_NHR_Iconic.png",
    "Narrative Hook Rate (NHR)",
    "Hook Rate (%)",
)


# 2. Heatmaps
plot_heatmap("Synergy Collapse Rate (%)", "Fig2_SCR_Heatmap.png", "SCR Heatmap", "Reds")
plot_heatmap(
    "Phantom Knowledge Rate (PKR) (%)", "Fig3_PKR_Heatmap.png", "PKR Heatmap", "Oranges"
)
plot_heatmap(
    "Stickiness (%)", "Fig_Stickiness_Heatmap.png", "Stickiness Heatmap", "Greens"
)
plot_heatmap("Narrative Hook Rate (%)", "Fig_NHR_Heatmap.png", "NHR Heatmap", "YlOrBr")

# 3. Dumbbell Charts
create_dumbbell_grid("Synergy Collapse Rate (%)", "SCR", "Fig5_Dumbbell_SCR.png")
create_dumbbell_grid("Phantom Knowledge Rate (PKR) (%)", "PKR", "Fig5_Dumbbell_PKR.png")
create_dumbbell_grid("Stickiness (%)", "Stickiness", "Fig_Dumbbell_Stickiness.png")
create_dumbbell_grid(
    "Narrative Hook Rate (%)", "Narrative Hook Rate", "Fig_Dumbbell_NHR.png"
)

# 4. Mechanism Grid (Stickiness/Teachability)
plot_mech_full_grid()

# 5. Ablation (Context Length)
create_granular_bar_grid("Synergy Collapse Rate (%)", "SCR", "Fig6_Granular_SCR.png")
plot_bucket_ablation("SCR", "Is_Failure", "Fig_Bucket_SCR.png", "SCR (%)")
plot_bucket_ablation("PKR", "Is_Success", "Fig_Bucket_PKR.png", "PKR (%)")
plot_bucket_ablation(
    "Stickiness", "Is_Success", "Fig_Bucket_Stickiness.png", "Stickiness (%)"
)
plot_bucket_ablation(
    "Narrative Hook Rate", "Is_Failure", "Fig_Bucket_NHR.png", "NHR (%)"
)

# 6. LogProb Analysis (Explainability)
plot_synergy_friction_and_gap()
plot_narrative_hook_confidence()
plot_narrative_hook_gap()
run_mechanistic_audit_rq3_rq4()


# 7. Generate Tables
print("\nGenerating LaTeX Tables...")
run_all_tables()
# get_detailed_mechanistic_table()
generate_synhook_tables()
run_main_text_aggregate_chi2()


# 1. Synergy Friction (Synthetic Data - PK+ Ctx+)
# We expect negative deltas (Instruct is more hesitant)
synergy_mask = (
    (df["PK_Score"] == 10)
    & (df["Context_Label"] == "Ctx_correct")
    & (df["Context_Class"] == "Short / Synthetic")
)

print_dual_mechanistic_table(
    "Synergy Friction (Synthetic) ", synergy_mask, higher_is_better=True
)

# 2. Narrative Hook (Natural Data - PK+ Ctx- Accepted Lies)
# We expect positive deltas (Instruct is more confidently wrong)
hook_mask = (
    (df["PK_Score"] == 10)
    & (df["Context_Label"] == "Ctx_incorrect")
    & (df["Context_Class"] == "Short / Synthetic")
    & (df["Answer_Label"] == "Ans_Failure")
)

print_dual_mechanistic_table(
    "Narrative Hook (Lie Acceptance short/synthetic)", hook_mask, higher_is_better=False
)


hook_mask = (
    (df["PK_Score"] == 10)
    & (df["Context_Label"] == "Ctx_incorrect")
    & (df["Context_Class"] == "Long / Natural")
    & (df["Answer_Label"] == "Ans_Failure")
)

print_dual_mechanistic_table(
    "Narrative Hook (Lie Acceptance long/narrative)", hook_mask, higher_is_better=False
)


# 3. Phantom Knowledge (Natural Data - PK- Ctx- Reconstructing Truth)
# Quadrant: Model didn't "know" it (PK-), Context lied (Ctx-), but Model got it right.
# We expect positive deltas (Instruct is more decisive in its latent recall/vibe-completion)
phantom_mask = (
    (df["PK_Score"] == 0)
    & (df["Context_Label"] == "Ctx_incorrect")
    & (df["Context_Class"] == "Long / Natural")
    & (df["Is_Success"] == 1)
)  # Using Is_Success is safer than the raw string

# Use a lower threshold (MIN_N=1) for this table because Base successes are rare
print_dual_mechanistic_table(
    "Phantom Knowledge (Truth Reconstruction long)",
    phantom_mask,
    higher_is_better=False,
    min_n=1,
)

phantom_mask = (
    (df["PK_Score"] == 0)
    & (df["Context_Label"] == "Ctx_incorrect")
    & (df["Context_Class"] == "Short / Synthetic")
    & (df["Is_Success"] == 1)
)  # Using Is_Success is safer than the raw string

# Use a lower threshold (MIN_N=1) for this table because Base successes are rare
print_dual_mechanistic_table(
    "Phantom Knowledge (Truth Reconstruction synthetic)",
    phantom_mask,
    higher_is_better=False,
    min_n=1,
)


print("\n✅ All assets generated successfully.")
