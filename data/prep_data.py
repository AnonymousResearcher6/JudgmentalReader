import json
import random
import os
import re
import argparse


def load_data(file_path):
    data = []
    if not os.path.exists(file_path):
        return data
    _, ext = os.path.splitext(file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        if ext == ".jsonl":
            data = [json.loads(line) for line in f if line.strip()]
        elif ext == ".json":
            data = json.load(f)
    return data


def save_jsonl(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Successfully saved {len(data)} items to {file_path}")


def create_ua_prompt(
    question, context=None, is_repetitive=False, repetitive_answer=None
):
    if is_repetitive:
        return f"USER: Please answer the following question.\nQuestion: {question.strip()}\nASSISTANT: {repetitive_answer.strip()}.\nUSER: Please answer the same question again.\nQuestion: {question.strip()}\nASSISTANT:"
    elif context:
        return f"USER: Based on the following context, please answer the question.\n\nContext:\n{context.strip()}\n\nQuestion:\n{question.strip()}\n\nASSISTANT:"
    else:
        return f"USER: Please answer the following question.\n\nQuestion:\n{question.strip()}\n\nASSISTANT:"


def create_clean_version(distractor_text, correct_answer, distractor_answer):
    if not all([distractor_text, correct_answer, distractor_answer]):
        return ""
    # Use word boundaries for safety
    pattern = re.compile(r"\b" + re.escape(distractor_answer) + r"\b", re.IGNORECASE)
    return pattern.sub(correct_answer, distractor_text)


def process_juice_data(juice_data):
    """Converts raw JUICE data to the unified format using RAW COMPLETION."""
    unified = []

    for i, item in enumerate(juice_data):
        correct_answer = item.get("Answer", [""])[0]
        distractor_answer = item.get("Distracted Token", "")

        question_completion = item.get("Clean Prompt", "").strip()

        if not all([correct_answer, distractor_answer, question_completion]):
            continue

        entry = {
            "id": f"juice_{i}",
            "category": "juice",
            "question_text": question_completion,
            "correct_answer": correct_answer,
            "distractor_answer": distractor_answer,
            "prompt_clean": question_completion,
            "prompt_substitute_conflict": item.get("Substitution Conflict", ""),
            "prompt_coherent_conflict": item.get("Coherent Conflict", ""),
            "prompt_clean_substitute": create_clean_version(
                item.get("Substitution Conflict", ""), correct_answer, distractor_answer
            ),
            "prompt_clean_coherent": create_clean_version(
                item.get("Coherent Conflict", ""), correct_answer, distractor_answer
            ),
        }
        unified.append(entry)
    return unified


def process_squad_musique_data(
    golden_path, negative_path, category_name, ua_format=True
):
    """Processes SQuAD/MuSiQue data using golden and negative files."""
    golden_data, negative_data = load_data(golden_path), load_data(negative_path)
    if not golden_data or not negative_data:
        return []

    negative_map = {item["id"]: item for item in negative_data}
    prompt_creator = create_ua_prompt

    unified = []
    for golden_item in golden_data:
        item_id = golden_item["id"]
        if item_id in negative_map:
            negative_item = negative_map[item_id]
            question = golden_item["question"]
            correct_answer = golden_item["answer"].strip()
            distractor_answer = negative_item["answer"].strip()

            # Create the repetitive prompts
            repetitive_conflict_prompt = prompt_creator(
                question, is_repetitive=True, repetitive_answer=distractor_answer
            )
            clean_repetitive_prompt = prompt_creator(
                question, is_repetitive=True, repetitive_answer=correct_answer
            )

            entry = {
                "id": item_id,
                "category": category_name,
                "question_text": question,
                "correct_answer": correct_answer,
                "distractor_answer": distractor_answer,
                "prompt_clean": prompt_creator(question),
                "prompt_repetitive_conflict": repetitive_conflict_prompt,
                "prompt_narrative_conflict": prompt_creator(
                    question, context=negative_item["context"]
                ),
                "prompt_clean_repetitive": clean_repetitive_prompt,
                "prompt_clean_narrative": prompt_creator(
                    question, context=golden_item["context"]
                ),
            }
            unified.append(entry)
    return unified


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unify NQ, SQuAD, and MuSiQue datasets."
    )
    parser.add_argument(
        "--ua_format",
        action="store_true",
        help="Ensures prompts are in USER/ASSISTANT style (default).",
    )
    args = parser.parse_args()

    output_dir = "new_data"
    juice_file = "original/juice_data.jsonl"
    musique_golden = "original/musique_golden.json"
    musique_negative = "original/musique_negative.json"
    squad_golden = "original/squad_golden.json"
    squad_negative = "original/squad_negative.json"

    all_unified_data = []

    print("\n--- Processing all data sources into a unified format ---")

    juice_raw = load_data(juice_file)
    if juice_raw:
        unified_juice = process_juice_data(juice_raw)
        all_unified_data.extend(unified_juice)
        print(f"Loaded and processed {len(unified_juice)} items from JUICE.")

    unified_musique = process_squad_musique_data(
        musique_golden, musique_negative, "musique", ua_format=True
    )
    all_unified_data.extend(unified_musique)
    print(f"Loaded and processed {len(unified_musique)} items from MuSiQue.")

    unified_squad = process_squad_musique_data(
        squad_golden, squad_negative, "squad", ua_format=True
    )
    all_unified_data.extend(unified_squad)
    print(f"Loaded and processed {len(unified_squad)} items from SQuAD.")

    output_filename = "kc_data_ua.jsonl"

    if not all_unified_data:
        print("No data was generated. Please check file paths. Exiting.")
    else:
        print(f"\nTotal unified dataset size: {len(all_unified_data)} items.")
        random.seed(42)
        random.shuffle(all_unified_data)

        final_output_path = os.path.join(output_dir, output_filename)
        save_jsonl(all_unified_data, final_output_path)
        print("\nData preparation complete.")
