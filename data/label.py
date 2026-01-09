import argparse
import asyncio
import json
import re
import glob
import sys
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# vLLM imports
from transformers import AutoTokenizer
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm import SamplingParams


LABELING_PROMPT = """You are a precise and objective AI evaluator. Your task is to analyze a model's response to a question and extract the answer it provides, then judge if that answer is correct.

--- EXAMPLE START ---
Question:
What is the hottest planet in the solar system?
Model's response:
The distinction of being the hottest planet does not belong to Mercury, despite its proximity to the Sun. Instead, the title belongs to Venus due to its dense, heat-trapping atmosphere.
Known correct answer: "Venus"
Potential distractor answer: "Mercury"
Thought process:
The model's response mentions both the distractor ("Mercury") and the correct answer ("Venus").
The response explicitly negates the distractor with the phrase "does not belong to Mercury".
It then positively asserts the correct answer with the phrase "the title belongs to Venus".
Therefore, the model's final asserted answer is "Venus".
This matches the "Known correct answer".
Therefore, the extracted answer is correct.

Final Answer (JSON object only):
{{
"extracted_answer": "Venus",
"is_correct": true
}}
--- EXAMPLE END ---

--- TASK START ---
Question:
{question}
Model's Response:
{model_response}
Known Correct Answer: "{correct_answer}"
Potential Distractor Answer: "{distractor_answer}"
Thought process:
"""


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--judge_model_path", type=str, default="meta-llama/Meta-Llama-3-70B-Instruct"
    )
    parser.add_argument("--original_dataset_path", type=str, required=True)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="labeled_results")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1024)
    args = parser.parse_args()
    return args


def find_all_spans(haystack, needle):
    spans = []
    if not needle or not isinstance(needle, str) or not haystack:
        return []
    try:
        for match in re.finditer(re.escape(needle), haystack, re.IGNORECASE):
            spans.append(
                {"start": match.start(), "end": match.end(), "text": match.group()}
            )
    except Exception:
        return []
    return spans


def get_original_data_mapping(dataset_path):
    mapping = {}
    print(f"--- Loading original dataset: {dataset_path} ---")
    try:
        with open(dataset_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                q_text = item.get("question_text", item.get("question", "N/A"))
                mapping[item["id"]] = {
                    "correct_answer": item.get("correct_answer", "N/A"),
                    "distractor_answer": item.get("distractor_answer", "N/A"),
                    "question_text": q_text,
                }
    except FileNotFoundError:
        print(f"Error: Could not find original dataset at {dataset_path}")
        sys.exit(1)
    return mapping


def extract_json_from_text(text):
    """
    Robustly extracts JSON and Thought Process.
    """
    extracted_answer = "PARSING_ERROR"
    is_correct = False
    thought_process = text  # Default to whole text if no JSON found

    if not text or not text.strip():
        return {
            "extracted_answer": "NO_OUTPUT",
            "is_correct": False,
            "judge_thought_process": "Error: Model output is empty.",
        }

    try:
        # Find the last JSON block
        matches = list(re.finditer(r"\{.*?\}", text, re.DOTALL))

        json_found = False
        if matches:
            for match in reversed(matches):
                try:
                    candidate = json.loads(match.group())
                    if "extracted_answer" in candidate:
                        extracted_answer = candidate.get(
                            "extracted_answer", "PARSING_ERROR"
                        )
                        is_correct = candidate.get("is_correct", False)
                        if isinstance(is_correct, str):
                            is_correct = is_correct.lower() == "true"

                        thought_process = text[: match.start()].strip()
                        thought_process = re.sub(
                            r"Final Answer.*?:",
                            "",
                            thought_process,
                            flags=re.IGNORECASE,
                        ).strip()

                        json_found = True
                        break
                except:
                    continue

        if not json_found:
            ans_match = re.search(r'"extracted_answer"\s*:\s*"(.*?)"', text)
            if ans_match:
                extracted_answer = ans_match.group(1)

            corr_match = re.search(
                r'"is_correct"\s*:\s*(true|false)', text, re.IGNORECASE
            )
            if corr_match:
                is_correct = corr_match.group(1).lower() == "true"

    except Exception as e:
        thought_process += f" [System Error: {e}]"

    return {
        "extracted_answer": extracted_answer,
        "is_correct": is_correct,
        "judge_thought_process": thought_process,
    }


async def process_request(engine, prompt, sampling_params, req_id):
    """Consumes vLLM async generator."""
    results_generator = engine.generate(prompt, sampling_params, req_id)
    final_output = None
    async for result in results_generator:
        final_output = result
    return final_output


def flush_item_to_disk(
    model_slug, item_id, item_results, original_info, output_handles
):
    """
    Aggregates all parts (Probes + Scenarios) for one ID and writes files.
    """
    pk_probes = []
    scenarios = []

    # Separate Probes from Scenarios
    for res in item_results:
        # Ensure we include the raw output for debugging
        res["labeled_output"]["raw_judge_output"] = res.get("raw_judge_text", "")

        s_type = res["original_item"]["scenario_type"]
        if "knowledge_probe" in s_type:
            pk_probes.append(res)
        else:
            scenarios.append(res)

    # Calculate Parametric Knowledge (PK)
    pk_probes.sort(key=lambda x: x["original_item"]["scenario_type"])
    correct_count = sum(
        1 for p in pk_probes if p["labeled_output"]["is_correct"] is True
    )
    pk_label = "PK_correct" if correct_count >= 5 else "PK_incorrect"

    formatted_probes = []
    for p in pk_probes:
        formatted_probes.append(
            {
                "original_response": p["original_item"]["response_text"],
                "labeled_info": p["labeled_output"],
            }
        )

    # Process Scenarios
    formatted_scenarios = []
    for s in scenarios:
        item = s["original_item"]
        judge = s["labeled_output"]
        scenario_name = item["scenario_type"]

        # Clean=Correct, Conflict=Incorrect
        if "Clean" in scenario_name:
            ctx_label = "Ctx_correct"
        elif "Conflict" in scenario_name or "conflict" in scenario_name.lower():
            ctx_label = "Ctx_incorrect"
        else:
            ctx_label = "Ctx_incorrect"

        ans_label = "Ans_Success" if judge["is_correct"] else "Ans_Failure"

        # The final combined label
        final_label_str = f"{pk_label}-{ctx_label}-{ans_label}"

        # Find character spans
        spans = find_all_spans(item["response_text"], judge["extracted_answer"])
        labeled_spans = []
        for span in spans:
            span["label_type"] = final_label_str
            labeled_spans.append(span)

        formatted_scenarios.append(
            {
                "source_id": item_id,
                "model": model_slug,
                "temperature": 0.0,
                "parametric_knowledge_label": pk_label,
                "parametric_knowledge_score": correct_count,
                "question_details": {
                    "correct_answer": original_info.get("correct_answer"),
                    "distractor_answer": original_info.get("distractor_answer"),
                },
                "id": item_id,
                "scenario_type": scenario_name,
                "context_label": ctx_label,
                "answer_label": ans_label,
                "final_label": final_label_str,
                "prompt": item["prompt"],
                "original_response": item["response_text"],
                "labeled_info": judge,
                "labels": labeled_spans,
            }
        )

    formatted_scenarios.sort(key=lambda x: x["scenario_type"])

    if model_slug not in output_handles:
        return
    f = output_handles[model_slug]

    if formatted_probes:
        summary_prompt = original_info.get("question_text", "N/A")

        probe_summary = {
            "source_id": item_id,
            "model": model_slug,
            "temperature": 0.0,
            "parametric_knowledge_label": pk_label,
            "parametric_knowledge_score": correct_count,
            "question_details": {
                "correct_answer": original_info.get("correct_answer"),
                "distractor_answer": original_info.get("distractor_answer"),
            },
            "id": item_id,
            "prompt": summary_prompt,
            "knowledge_probe_results": formatted_probes,
            "scenarios": [],  # Scenarios are in separate lines
        }
        f.write(json.dumps(probe_summary) + "\n")

    # Write conflict scenarios
    for scen_obj in formatted_scenarios:
        f.write(json.dumps(scen_obj) + "\n")

    f.flush()


async def run_labeling_stage(args):
    print("\n--- Initializing Judge vLLM Engine ---")

    tokenizer = AutoTokenizer.from_pretrained(
        args.judge_model_path, trust_remote_code=True
    )
    stop_token_ids = []
    if tokenizer.eos_token_id is not None:
        stop_token_ids.append(tokenizer.eos_token_id)
    if "<|eot_id|>" in tokenizer.get_vocab():
        stop_token_ids.append(tokenizer.convert_tokens_to_ids("<|eot_id|>"))

    # Stop at start of new tasks/examples
    stop_strings = ["--- TASK END ---"]

    engine_args = AsyncEngineArgs(
        model=args.judge_model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        gpu_memory_utilization=0.90,
        max_model_len=4096,
        disable_log_stats=False,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    sampling_params = SamplingParams(
        n=1,
        temperature=0.0,
        max_tokens=1024,
        stop=stop_strings,
        stop_token_ids=stop_token_ids,
    )

    original_data_map = get_original_data_mapping(args.original_dataset_path)
    source_files = glob.glob(f"{args.input_dir}/*_flat.jsonl")

    # Buffers
    request_context = {}
    expected_counts = defaultdict(lambda: defaultdict(int))
    active_results = defaultdict(lambda: defaultdict(list))
    output_files = {}
    all_requests = []

    print("\n--- Phase 1: Preparing Requests ---")
    global_req_counter = 0

    for file_path in tqdm(source_files, desc="Scanning Inputs"):
        model_slug = Path(file_path).name.replace("_flat.jsonl", "")
        out_path = Path(args.output_dir) / f"{model_slug}_labeled.jsonl"

        if not args.overwrite and out_path.exists():
            print(f"Skipping {model_slug} (exists)")
            continue

        output_files[model_slug] = open(out_path, "w")

        with open(file_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                item_id = item["id"]

                expected_counts[model_slug][item_id] += 1

                original_info = original_data_map.get(item_id, {})

                # Use clean question for the judge prompt; we don't want to confuse judge with conflict context
                clean_judge_q = original_info.get("question_text", "N/A")

                judge_prompt = LABELING_PROMPT.format(
                    question=clean_judge_q,
                    model_response=item["response_text"],
                    correct_answer=original_info.get("correct_answer", "N/A"),
                    distractor_answer=item.get("distractor_answer_for_item")
                    or original_info.get("distractor_answer", "N/A"),
                )

                req_id = f"{model_slug}|{item_id}|{global_req_counter}"
                global_req_counter += 1

                request_context[req_id] = {
                    "model_slug": model_slug,
                    "item_id": item_id,
                    "original_item": item,
                }
                all_requests.append((req_id, judge_prompt))

    print(f"--- Total Requests: {len(all_requests)} ---")
    if not all_requests:
        return

    print("\n--- Phase 2: Processing ---")
    batch_size = args.batch_size

    pbar = tqdm(total=len(all_requests), desc="Judging")

    for i in range(0, len(all_requests), batch_size):
        batch_args = all_requests[i : i + batch_size]
        tasks = [
            process_request(engine, prompt, sampling_params, req_id)
            for req_id, prompt in batch_args
        ]

        batch_results = await asyncio.gather(*tasks)

        for result in batch_results:
            if not result:
                continue

            req_id = result.request_id
            ctx = request_context.pop(req_id)

            model_slug = ctx["model_slug"]
            item_id = ctx["item_id"]

            raw_text = result.outputs[0].text
            parsed = extract_json_from_text(raw_text)

            active_results[model_slug][item_id].append(
                {
                    "original_item": ctx["original_item"],
                    "labeled_output": parsed,
                    "raw_judge_text": raw_text,
                }
            )

            # Flush when all expected items for this ID are ready
            if (
                len(active_results[model_slug][item_id])
                == expected_counts[model_slug][item_id]
            ):
                flush_item_to_disk(
                    model_slug,
                    item_id,
                    active_results[model_slug][item_id],
                    original_data_map.get(item_id, {}),
                    output_files,
                )
                del active_results[model_slug][item_id]

        pbar.update(len(batch_results))

    pbar.close()
    for f in output_files.values():
        f.close()
    print("\n--- Labeling Complete ---")


def main():
    args = config()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    try:
        asyncio.run(run_labeling_stage(args))
    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
