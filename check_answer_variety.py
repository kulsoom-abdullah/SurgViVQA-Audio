#!/usr/bin/env python3
"""
Check actual answer variety for each question type
Helps distinguish truly open-ended vs limited-choice questions
"""

import json
from collections import Counter, defaultdict

def analyze_answer_variety(filepath):
    """Check unique answers for each question type"""
    with open(filepath) as f:
        data = [json.loads(line) for line in f]

    # Group by question type
    qtype_answers = defaultdict(list)
    for sample in data:
        qt = sample['question_type']
        answer = sample.get('short_answer', sample.get('answer', '')).strip()
        qtype_answers[qt].append(answer)

    print(f"\n{'='*80}")
    print(f"Answer Variety Analysis: {filepath}")
    print(f"{'='*80}\n")

    # Analyze each question type
    for qt in sorted(qtype_answers.keys()):
        answers = qtype_answers[qt]
        unique_answers = Counter(answers)
        total = len(answers)

        # Classify
        yes_no = unique_answers.get('yes', 0) + unique_answers.get('no', 0) + \
                 unique_answers.get('Yes', 0) + unique_answers.get('No', 0)

        if yes_no > total * 0.8:
            category = "Yes/No"
        elif len(unique_answers) <= 10:
            category = f"Limited Choice ({len(unique_answers)} options)"
        else:
            category = f"Open-Ended ({len(unique_answers)} unique)"

        print(f"{qt:35} | {total:4} samples | {category}")

        # Show top 5 most common answers
        top_answers = unique_answers.most_common(5)
        for ans, count in top_answers:
            pct = count / total * 100
            print(f"  {'':35}   - {ans:20} ({count:3}, {pct:5.1f}%)")

        print()

def main():
    for split in ['train_multivideo.jsonl', 'eval_multivideo.jsonl', 'test_multivideo.jsonl']:
        try:
            analyze_answer_variety(split)
        except FileNotFoundError:
            print(f"Skipping {split} (not found)")

if __name__ == '__main__':
    main()
