import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from tqdm import tqdm as std_tqdm
from tqdm.asyncio import tqdm as async_tqdm
from transformers import AutoTokenizer
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm import SamplingParams


def config():
    parser = argparse.ArgumentParser(
        description="Run LLM generation for the Priming Experiment."
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Hugging Face model identifier."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="prepared_data/priming_data_ua.jsonl",
        help="Path to the prepared .jsonl dataset from prep_data.py.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="raw_results",
        help="Directory to save the flat output files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of data points for debugging.",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism.",
    )
    parser.add_argument(
        "--max_concurrent_requests",
        type=int,
        default=256,
        help="Max concurrent vLLM requests.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, overwrite existing output file.",
    )
    args = parser.parse_args()
    return args


def create_prompts_for_item(item):
    """
    Creates a flat list of all prompts to be run for a given question object,
    based on the new "one question, one object" data structure.
    """
    prompts_to_run = []

    # Ten knowledge probes
    for i in range(10):
        prompts_to_run.append(
            {
                "prompt_text": item["prompt_clean"],
                "scenario_type": f"knowledge_probe_{i}",
                "distractor_answer": None,  # No distractor for probes
            }
        )

    # Clean context scenarios
    prompts_to_run.append(
        {
            "prompt_text": item["prompt_clean_narrative"],
            "scenario_type": "Clean Narrative",
            "distractor_answer": None,
        }
    )
    prompts_to_run.append(
        {
            "prompt_text": item["prompt_clean_repetitive"],
            "scenario_type": "Clean Repetitive",
            "distractor_answer": None,
        }
    )

    # Conflict scenarios for each of the 3 distractors
    for scenario in item["distractor_scenarios"]:
        distractor = scenario["distractor_answer"]
        prompts_to_run.append(
            {
                "prompt_text": scenario["prompt_narrative_conflict"],
                "scenario_type": "Narrative Conflict",
                "distractor_answer": distractor,  # Propagate metadata
            }
        )
        prompts_to_run.append(
            {
                "prompt_text": scenario["prompt_repetitive_conflict"],
                "scenario_type": "Repetitive Conflict",
                "distractor_answer": distractor,  # Propagate metadata
            }
        )

    return prompts_to_run


def save_results_batch(results_batch, output_file):
    """Appends a batch of results to a .jsonl file."""
    with open(output_file, "a") as f:
        for result in results_batch:
            f.write(json.dumps(result) + "\n")


async def run_generation_stage(args, dataset, output_path):
    """Runs the generation stage with a robust async pattern and graceful shutdown."""
    engine = None
    try:
        print("\n--- Initializing vLLM Engine (Asynchronous) ---")
        engine_args = AsyncEngineArgs(
            model=args.model_path,
            tensor_parallel_size=args.tensor_parallel_size,
            trust_remote_code=True,
            gpu_memory_utilization=0.95,
            max_model_len=2048,
        )
        engine = AsyncLLMEngine.from_engine_args(engine_args)

        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, trust_remote_code=True
        )
        stop_sequences = [
            "<|eot_id|>",
            "<|end_of_text|>",
            "</s>",
            "<|endoftext|>",
            "<end_of_turn>",
            "<|im_end|>",
        ]
        if tokenizer.eos_token and tokenizer.eos_token not in stop_sequences:
            stop_sequences.append(tokenizer.eos_token)

        print(f"Configured stop sequences: {stop_sequences}")
        sampling_params = SamplingParams(
            n=1, temperature=0.0, max_tokens=128, stop=stop_sequences
        )

        print("\n--- Creating Generation Requests ---")
        all_requests_info = []
        request_counter = 0
        for item in std_tqdm(dataset, desc="Creating Requests"):
            prompts = create_prompts_for_item(item)
            for prompt_info in prompts:
                all_requests_info.append(
                    {
                        "request_id": str(request_counter),
                        "item_id": item["id"],
                        "prompt_info": prompt_info,
                    }
                )
                request_counter += 1
        print(f"Created {len(all_requests_info)} total requests to process.")

        print(f"\n--- Starting Generation ---")
        start_time = time.time()

        results_buffer = []
        pbar = async_tqdm(total=len(all_requests_info), desc="Generating Responses")
        semaphore = asyncio.Semaphore(args.max_concurrent_requests)

        async def process_request(req_info):
            async with semaphore:
                prompt_info = req_info["prompt_info"]
                results_generator = engine.generate(
                    prompt_info["prompt_text"], sampling_params, req_info["request_id"]
                )
                final_output = None
                async for request_output in results_generator:
                    final_output = request_output
                response_text = final_output.outputs[0].text if final_output else ""

                return {
                    "id": req_info["item_id"],
                    "scenario_type": prompt_info["scenario_type"],
                    "distractor_answer_for_prompt": prompt_info["distractor_answer"],
                    "prompt": prompt_info["prompt_text"],
                    "response_text": response_text.strip(),
                }

        tasks = [process_request(req) for req in all_requests_info]

        for future in asyncio.as_completed(tasks):
            try:
                result = await future
                results_buffer.append(result)
            except Exception as e:
                print(f"An error occurred while processing a request: {e}")
            finally:
                pbar.update(1)

            if len(results_buffer) >= args.max_concurrent_requests:
                save_results_batch(results_buffer, output_path)
                results_buffer.clear()

        if results_buffer:
            save_results_batch(results_buffer, output_path)

        end_time = time.time()
        pbar.close()
        print(f"\nTotal generation time: {end_time - start_time:.2f} seconds.")

    finally:
        if engine is not None:
            print("\n--- Generation finished. Shutting down engine gracefully. ---")
            del engine
            await asyncio.sleep(2.0)
            print("--- Engine shut down. ---")


def main():
    args = config()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model_slug = Path(args.model_path).name
    output_path = Path(args.output_dir) / f"{model_slug}_flat.jsonl"

    if output_path.exists() and not args.overwrite:
        print(
            f"ERROR: Output file {output_path} already exists. Use --overwrite flag to replace it."
        )
        exit(1)
    elif output_path.exists() and args.overwrite:
        os.remove(output_path)

    print("--- Loading Initial Dataset ---")
    dataset = [
        json.loads(line)
        for line in std_tqdm(open(args.dataset_path, "r"), desc="Loading Dataset")
    ]
    if args.limit:
        dataset = dataset[: args.limit]
    print(f"Loaded {len(dataset)} data points.")

    print("\n" + "=" * 25 + " STAGE 1: GENERATION " + "=" * 25)
    asyncio.run(run_generation_stage(args, dataset, output_path))
    print("\n--- All stages complete! ---")
    print(f"Final flat output is at: {output_path}")


if __name__ == "__main__":
    main()
