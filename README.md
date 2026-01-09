# 📖 Judgmental Reader

**Judgmental Reader: How Instruction Tuning Introduces Formatting Fragility in RAG Grounding** (ACL 2026 Submission)

A large-scale analysis of 20 state-of-the-art models revealing how instruction tuning conditions reasoning capabilities on stylistic patterns, rendering factual grounding fragile when faced with non-narrative or synthetic data.

[![arXiv](https://img.shields.io/badge/arXiv-26xx.xxxxx-b31b1b.svg)](https://arxiv.org/abs/26xx.xxxxx) 
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Data-yellow)](https://huggingface.co/)
[![GitHub](https://img.shields.io/badge/GitHub-Project-blue)](https://github.com/AnonymousResearcher6/JudgementalReader)

---

## 🚀 Highlights
*   **🧩 Three Behavioral Shifts:** Characterizes **Synergy Collapse** (failure on synthetic truth), the **Narrative Hook** (suggestibility to narrative lies), and **Phantom Knowledge** (rejection-triggered recall).
*   **🧑‍🔬 Massive Scale:** Evaluates 20 model pairs (Base vs. Instruct) across **238,210** observations using NQ, SQuAD, and MuSiQue.
*   **⚖️ The Alignment Trade-off:** Quantifies how alignment makes factual grounding highly sensitive to narrative framing, creating a hidden trade-off between conversational fluidity and RAG robustness.
*   **🧠 Mechanistic Audit:** Uses internal confidence metrics ($\Delta$ LogProbs) to reveal that instruction-tuned models exhibit higher decisiveness when following narrative misinformation than when extracting facts from synthetic formats.

---

## 🛠️ Framework Overview

Our pipeline is designed for reproducible, large-scale evaluation of RAG grounding robustness:

| File | Stage | Description |
| :--- | :--- | :--- |
| `prep_data.py` | **Prepare** | Unifies raw datasets (NQ/JUICE, SQuAD, MuSiQue) into a standardized format with synthetic/narrative scaffolds. |
| `generate_data.py` | **Generate** | High-throughput inference engine using **vLLM** to collect model responses and logprobs. |
| `label.py` | **Evaluate** | Automated tri-modal classification using a large LLM (e.g., Llama-3-70B) with Chain-of-Thought (CoT) reasoning. |
| `plot.py` | **Analyze** | Full analysis suite: generates heatmaps, dumbbell plots, and performs Chi-Square statistical audits. |
| `annotate.py` | **Validate** | CLI-based interface for PhD-level gold-standard human verification. |
| `calculate_agreement.py` | **Validate** | Computes Cohen’s Kappa to ensure judge-to-human reliability. |

---

## 🚦 Usage

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
uv pip install vllm  # Required for generate_data.py and label.py
```

### 2️⃣ Data Preparation
```bash
python prep_data.py --ua_format
```

### 3️⃣ Scale Generation
```bash
python generate_data.py \
    --model_path "meta-llama/Llama-3.2-3B-Instruct" \
    --dataset_path "new_data/kc_data_ua.jsonl" \
    --output_dir "raw_results" \
    --tensor_parallel_size 1
```

### 4️⃣ Automated Labeling (LLM Judge)
Uses a high-capacity model to extract answers and judge correctness based on CoT.
```bash
python label.py \
    --judge_model_path "meta-llama/Meta-Llama-3.3-70B-Instruct" \
    --original_dataset_path "new_data/kc_data_ua.jsonl" \
    --input_dir "raw_results" \
    --output_dir "labeled_results" \
    --tensor_parallel_size 4
```

### 5️⃣ Visualization & Stats
```bash
# Set DATA_DIR in plot.py to "labeled_results"
python plot.py
```

---

## 📊 Dataset & Metrics
We examine the intersection of internal knowledge ($PK$) and retrieved context ($Ctx$):
*   **Synergy Collapse Rate (SCR):** Failure rate despite unanimous truth signals in synthetic formats ($PK^+ \cap Ctx_{cor}$).
*   **Narrative Hook Rate (NHR):** Frequency of following a narrative "lie" over internal truth ($PK^+ \cap Ctx_{inc}$).
*   **Phantom Knowledge Rate (PKR):** Frequency of latent recall triggered by the rejection of a synthetic distractor ($PK^- \cap Ctx_{inc}$).

> [!NOTE]  
> Data and pre-computed logprobs will be available on **HuggingFace** upon paper acceptance.

---

## 📖 Citation
*The BibTeX will be updated following the peer-review process.*

```bibtex
@article{anonymous2026judgmental,
  title={Judgmental Reader: How Instruction Tuning Introduces Formatting Fragility in RAG Grounding},
  author={Anonymous Authors},
  journal={arXiv preprint arXiv:26xx.xxxxx},
  year={2026}
}
```

---

## 📬 Contact
For questions regarding the methodology or the anonymous code, please use the GitHub issue tracker:  
**[https://github.com/AnonymousResearcher6/JudgementalReader](https://github.com/AnonymousResearcher6/JudgementalReader)**