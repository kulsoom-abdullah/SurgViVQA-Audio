import json
import collections
import re

# --- Configuration ---
FILES_TO_ANALYZE = {
    "In-Template (Robotic)": "in_template.jsonl",
    "Out-Template (Natural)": "out_template.jsonl"
}
# Simple regex to catch potential acronyms (2+ capital letters together)
ACRONYM_RE = re.compile(r'\b[A-Z]{2,}\b')

def analyze_file(filepath, label):
    print(f"\n--- Analyzing: {label} ---")
    try:
        with open(filepath, 'r') as f:
            data = [json.loads(line) for line in f]
    except FileNotFoundError:
        print(f"Error: File {filepath} not found. Skipping.")
        return

    total_count = len(data)
    print(f"Total samples: {total_count}")

    # 1. Answer Balance Analysis
    answers = [item['short_answer'].lower() for item in data if 'short_answer' in item]
    answer_counts = collections.Counter(answers)
    print("\nAnswer Distribution (Short Answer):")
    for ans, count in answer_counts.most_common(5):
        percentage = (count / total_count) * 100
        print(f"  '{ans}': {count} ({percentage:.1f}%)")

    # 2. Question Category Analysis
    categories = [item['question_type'] for item in data if 'question_type' in item]
    category_counts = collections.Counter(categories)
    print("\nTop 5 Question Categories:")
    for cat, count in category_counts.most_common(5):
        print(f"  {cat}: {count}")

    # 3. Length Analysis (for Audio duration estimation)
    # Estimate: ~3 words per second for clear medical speech
    question_lengths = [len(item['question'].split()) for item in data if 'question' in item]
    max_len = max(question_lengths)
    avg_len = sum(question_lengths) / len(question_lengths)
    print(f"\nQuestion Length Estimation:")
    print(f"  Max words: {max_len} (approx {max_len/3:.1f} seconds audio)")
    print(f"  Avg words: {avg_len:.1f} (approx {avg_len/3:.1f} seconds audio)")

    # 4. Vocabulary & Acronym Spotting (Crucial for TTS)
    all_questions_text = " ".join([item['question'] for item in data if 'question' in item])
    # Find potential acronyms
    acronyms = collections.Counter(ACRONYM_RE.findall(all_questions_text))
    print("\nPotential Acronyms/Abbreviations (Check pronunciation!):")
    if not acronyms:
        print("  None found with simple regex.")
    for acro, count in acronyms.most_common(15):
        print(f"  {acro}: {count} times")

    # Look at a few random samples to see the "flavor" of the text
    print(f"\nRandom Sample Questions ({label}):")
    import random
    sample_indices = random.sample(range(total_count), min(3, total_count))
    for i in sample_indices:
        print(f"  - Q: {data[i]['question']}")
        print(f"    A: {data[i]['answer']}")

# --- Run Analysis ---
# Make sure you download the .jsonl files from the repo first!
# You only need the jsonl files for this step, not the images.
for label, filename in FILES_TO_ANALYZE.items():
    analyze_file(filename, label)
